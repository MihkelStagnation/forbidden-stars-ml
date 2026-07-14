"""Self-play rollout + a simplified, PPO-style training update.

Design choices (kept deliberately simple for a first scaffold):
* One shared policy plays both sides. Each transition records which player
  made it so we can assign the zero-sum outcome with the right sign.
* Advantages use GAE over each player's own trajectory (reward is 0 until the
  terminal outcome for that player). lam=1, gamma=1 recovers plain
  Monte-Carlo returns; the defaults trade a little bias for less variance —
  which matters here because setup/dice luck dominates raw outcomes.
* Recurrence uses *stored* GRU input states (no backprop-through-time): the
  memory fed at action time is saved and replayed during the update, so each
  transition's forward pass is independent and batchable. TODO: true BPTT.
* The update is a clipped surrogate (PPO-style) with a value loss and an
  entropy bonus.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import obs_to_tensors, stack_obs


def collect_episode(env, model, device="cpu", compositions=None, factions=None,
                    opponent=None, learner_player=0, reg_model=None):
    """Play one episode. Returns (transitions, winner).

    opponent None -> mirror self-play: the model plays both sides and BOTH
    sides' transitions are recorded (each tagged with its player, so the
    trainer signs outcomes per side). Otherwise `opponent(env, obs, player)`
    acts for the non-learner side (a frozen league member or scripted bot)
    and only `learner_player`'s transitions are recorded.

    reg_model (R-NaD): a frozen regularization net evaluated on the SAME
    observations/memory-thread; each transition also records that net's
    log-prob of the taken action, so the trainer can apply the DeepNash-
    style reward transform -eta*log(pi/pi_reg).
    """
    model.eval()
    obs, info = env.reset(compositions=compositions, factions=factions)
    memory = {0: model.initial_memory(1, device), 1: model.initial_memory(1, device)}
    if reg_model is not None:
        reg_model.eval()
        reg_memory = {0: reg_model.initial_memory(1, device),
                      1: reg_model.initial_memory(1, device)}
    transitions = []

    done = False
    while not done:
        player = info["current_player"]
        if opponent is not None and player != learner_player:
            action = opponent(env, obs, player)
            obs, reward, done, info = env.step(action)
            continue

        tensors = obs_to_tensors(obs, device)
        mem_in = memory[player]

        with torch.no_grad():
            logits, value, mem_out = model(tensors, mem_in)
            probs = F.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())
            logp = float(torch.log(probs[0, action] + 1e-12).item())

        memory[player] = mem_out

        t = {
            "obs": {k: np.asarray(v) for k, v in obs.items() if k != "phase"},
            "action": action,
            "logp": logp,
            "value": float(value.item()),
            "player": player,
            "mem_in": mem_in.squeeze(0).detach().cpu().numpy(),
        }
        if reg_model is not None:
            with torch.no_grad():
                rlogits, _, rmem = reg_model(tensors, reg_memory[player])
                t["logp_reg"] = float(
                    F.log_softmax(rlogits, dim=-1)[0, action].item())
            reg_memory[player] = rmem
        transitions.append(t)

        obs, reward, done, info = env.step(action)

    return transitions, env.winner


def _outcome(winner, player):
    if winner == -1 or winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def _step_rewards(traj, winner, p, eta):
    """Per-step rewards: the zero-sum outcome at the terminal step, plus —
    when eta > 0 (R-NaD) — the regularized transform -eta*log(pi/pi_reg)
    at EVERY step. Transitions collected without a reg_model contribute no
    transform (logp_reg defaults to logp)."""
    rewards = []
    for i, t in enumerate(traj):
        r = -eta * (t["logp"] - t.get("logp_reg", t["logp"])) if eta else 0.0
        if i == len(traj) - 1:
            r += _outcome(winner, p)
        rewards.append(r)
    return rewards


def build_batch(episodes, device="cpu", gamma=0.99, lam=0.95, eta=0.0):
    """Flatten episodes into update tensors with GAE advantages.

    Each player's transitions form their own trajectory: reward is 0 at every
    step except the last (the +1/-1/0 outcome for that player), and the value
    bootstrap after the terminal step is 0. Value targets are the lambda-
    returns (advantage + value). eta > 0 adds the R-NaD per-step transform.
    """
    obs_list, actions, old_logps, advantages, returns, mem_ins = [], [], [], [], [], []
    for transitions, winner in episodes:
        for p in (0, 1):
            traj = [t for t in transitions if t["player"] == p]
            if not traj:
                continue
            n = len(traj)
            rewards = _step_rewards(traj, winner, p, eta)
            vals = [t["value"] for t in traj] + [0.0]
            gae = 0.0
            advs = [0.0] * n
            for i in reversed(range(n)):
                delta = rewards[i] + gamma * vals[i + 1] - vals[i]
                gae = delta + gamma * lam * gae
                advs[i] = gae
            for i, t in enumerate(traj):
                obs_list.append(t["obs"])
                actions.append(t["action"])
                old_logps.append(t["logp"])
                advantages.append(advs[i])
                returns.append(advs[i] + vals[i])
                mem_ins.append(t["mem_in"])

    batch = stack_obs(obs_list, device)
    batch["action"] = torch.tensor(actions, dtype=torch.long, device=device)
    batch["old_logp"] = torch.tensor(old_logps, dtype=torch.float32, device=device)
    batch["advantage"] = torch.tensor(advantages, dtype=torch.float32, device=device)
    batch["return"] = torch.tensor(returns, dtype=torch.float32, device=device)
    batch["mem_in"] = torch.tensor(np.stack(mem_ins), dtype=torch.float32, device=device)
    return batch


OBS_KEYS = ("units_self", "units_enemy", "hand", "scalars", "action_mask")


def build_sequences(episodes, device="cpu", gamma=0.99, lam=0.95, chunk=32,
                    eta=0.0):
    """build_batch, but for truncated BPTT: each player's trajectory keeps
    its temporal order and is split into chunks of <= `chunk` steps. Returns
    (N, L, ...) tensors plus a validity mask; a chunk's initial memory is the
    ROLLOUT-time memory of its first step (stored state, no burn-in), and
    gradient flows through the GRU within the chunk."""
    raw = []
    for transitions, winner in episodes:
        for p in (0, 1):
            traj = [t for t in transitions if t["player"] == p]
            if not traj:
                continue
            n = len(traj)
            rewards = _step_rewards(traj, winner, p, eta)
            vals = [t["value"] for t in traj] + [0.0]
            gae = 0.0
            advs = [0.0] * n
            for i in reversed(range(n)):
                delta = rewards[i] + gamma * vals[i + 1] - vals[i]
                gae = delta + gamma * lam * gae
                advs[i] = gae
            for start in range(0, n, chunk):
                sub = traj[start:start + chunk]
                raw.append({
                    "steps": sub,
                    "adv": advs[start:start + chunk],
                    "ret": [advs[start + i] + vals[start + i]
                            for i in range(len(sub))],
                    "mem0": sub[0]["mem_in"],
                })

    N, L = len(raw), chunk
    batch = {}
    for key in OBS_KEYS:
        shape = raw[0]["steps"][0]["obs"][key].shape
        dtype = bool if key == "action_mask" else np.float32
        arr = np.zeros((N, L) + shape, dtype=dtype)
        for i, c in enumerate(raw):
            for t, s in enumerate(c["steps"]):
                arr[i, t] = s["obs"][key]
        batch[key] = torch.as_tensor(
            arr, dtype=torch.bool if key == "action_mask" else torch.float32,
            device=device)

    def pad(vals_per_chunk, dtype):
        arr = np.zeros((N, L), dtype=dtype)
        for i, vs in enumerate(vals_per_chunk):
            arr[i, :len(vs)] = vs
        return torch.as_tensor(arr, device=device)

    batch["action"] = pad([[s["action"] for s in c["steps"]] for c in raw],
                          np.int64)
    batch["old_logp"] = pad([[s["logp"] for s in c["steps"]] for c in raw],
                            np.float32)
    batch["advantage"] = pad([c["adv"] for c in raw], np.float32)
    batch["return"] = pad([c["ret"] for c in raw], np.float32)
    batch["valid"] = pad([[1.0] * len(c["steps"]) for c in raw], np.float32)
    batch["mem_in"] = torch.as_tensor(
        np.stack([c["mem0"] for c in raw]), dtype=torch.float32, device=device)
    return batch


def ppo_update_bptt(model, optimizer, batch, epochs=4, clip=0.2,
                    value_coef=0.5, entropy_coef=0.01):
    """Clipped-surrogate updates with gradient flowing through the GRU
    across each chunk's steps (truncated BPTT). Padded steps sit at chunk
    tails, so their memory garbage never feeds a valid step."""
    model.train()
    valid = batch["valid"]
    n_valid = valid.sum()
    adv = batch["advantage"]
    mean = (adv * valid).sum() / n_valid
    std = torch.sqrt(((adv - mean) ** 2 * valid).sum() / n_valid) + 1e-8
    adv = (adv - mean) / std

    L = valid.shape[1]
    stats = {}
    for _ in range(epochs):
        mem = batch["mem_in"]
        policy_loss = value_loss = entropy_sum = 0.0
        for t in range(L):
            obs_t = {k: batch[k][:, t] for k in OBS_KEYS}
            logits, value, mem = model(obs_t, mem)
            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(
                1, batch["action"][:, t].unsqueeze(1)).squeeze(1)
            v = valid[:, t]

            ratio = torch.exp(logp - batch["old_logp"][:, t])
            unclipped = ratio * adv[:, t]
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[:, t]
            policy_loss = policy_loss - (torch.min(unclipped, clipped) * v).sum()

            value_loss = value_loss + (
                (value - batch["return"][:, t]) ** 2 * v).sum()

            probs = torch.exp(logp_all)
            entropy_sum = entropy_sum - ((probs * logp_all).sum(dim=-1) * v).sum()

        policy_loss = policy_loss / n_valid
        value_loss = value_loss / n_valid
        entropy = entropy_sum / n_valid
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
    return stats


def ppo_update(model, optimizer, batch, epochs=4, clip=0.2,
               value_coef=0.5, entropy_coef=0.01):
    """A few epochs of clipped-surrogate updates over one batch."""
    model.train()
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {}
    for _ in range(epochs):
        logits, value, _ = model(
            {
                "units_self": batch["units_self"],
                "units_enemy": batch["units_enemy"],
                "hand": batch["hand"],
                "scalars": batch["scalars"],
                "action_mask": batch["action_mask"],
            },
            batch["mem_in"],
        )
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)

        ratio = torch.exp(logp - batch["old_logp"])
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()

        value_loss = F.mse_loss(value, batch["return"])

        probs = torch.exp(logp_all)
        entropy = -(probs * logp_all).sum(dim=-1).mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
    return stats
