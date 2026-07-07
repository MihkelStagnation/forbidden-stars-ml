"""Self-play LEAGUE training entry point.

    python -m scripts.train --iterations 1000

Each iteration collects a mix of episodes:
  * mirror self-play (the learner plays both sides — both sides recorded),
  * vs a frozen past snapshot from checkpoints/league/ (one sampled per iter),
  * vs the scripted heuristic bot.
Only the learner's transitions train. The learner's side (attacker/defender)
and the faction-role assignment (SM or Orks as attacker) are randomized per
episode, so one network learns all four faction-role combinations against
diverse opponents — the guard against brittle mirror-self-play strategies.

Snapshots are saved to the league every --snapshot-every iterations.
Saves the model to checkpoints/model.pt (every 5 iterations and at the end).
"""

import argparse
import os

import numpy as np
import torch

from fsneural.combat_env import CombatEnv
from fsneural import game_env as ge
from fsneural import board_env as be
from fsneural.model import CombatNet
from fsneural.agent import PolicyAgent
from fsneural.heuristic import (
    heuristic_action, campaign_heuristic_action, board_heuristic_action,
)
from fsneural.selfplay import collect_episode, build_batch, ppo_update

MIX_MIRROR, MIX_SNAPSHOT, MIX_HEURISTIC = 0.5, 0.35, 0.15


def make_env_and_model(kind, seed, device):
    if kind == "campaign":
        env = ge.GameEnv(seed=seed)
        model = CombatNet(scalar_feats=ge.SCALAR_FEATS,
                          n_build_actions=ge.N_BUILD_ACTIONS).to(device)
        return env, model, campaign_heuristic_action
    if kind == "board":
        env = be.BoardEnv(seed=seed)
        # flat actions 17..36: 14 build + 5 destinations + move-pass
        model = CombatNet(scalar_feats=be.SCALAR_FEATS,
                          unit_feats=be.UNIT_FEATS,
                          n_build_actions=ge.N_BUILD_ACTIONS + be.N_PLANETS + 1
                          ).to(device)
        return env, model, board_heuristic_action
    env = CombatEnv(seed=seed)
    return env, CombatNet().to(device), heuristic_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--episodes-per-iter", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/model.pt")
    ap.add_argument("--league-dir", type=str, default="checkpoints/league")
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--no-league", action="store_true",
                    help="pure mirror self-play (the pre-league behaviour)")
    ap.add_argument("--exploit", type=str, default=None, metavar="CHECKPOINT",
                    help="exploitability probe: train a fresh adversary with "
                         "EVERY episode against this frozen checkpoint")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--env", default="combat",
                    choices=("combat", "campaign", "board"))
    args = ap.parse_args()
    if args.env != "combat":  # separate artefacts unless overridden
        if args.out == "checkpoints/model.pt":
            args.out = f"checkpoints/{args.env}.pt"
        if args.league_dir == "checkpoints/league":
            args.league_dir = f"checkpoints/league_{args.env}"

    import sys
    sys.stdout.reconfigure(line_buffering=True)  # keep redirected logs live

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env, model, heuristic_fn = make_env_and_model(args.env, args.seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.league_dir, exist_ok=True)

    # reused buffer for frozen opponents (same dims as the learner)
    _, snapshot_model, _ = make_env_and_model(args.env, args.seed, device)
    snapshot_agent = PolicyAgent(snapshot_model, device)

    def net_opponent(env_, obs, player):
        action, _, _ = snapshot_agent.act(obs, player, deterministic=False)
        return action

    def heuristic_opponent(env_, obs, player):
        return heuristic_fn(env_, player)

    if args.exploit:
        snapshot_model.load_state_dict(
            torch.load(args.exploit, map_location=device))
        print(f"Exploitability probe: adversary vs frozen {args.exploit}")

    for it in range(1, args.iterations + 1):
        if not args.exploit:
            snapshots = sorted(
                f for f in os.listdir(args.league_dir) if f.endswith(".pt"))
            if snapshots and not args.no_league:
                pick = str(rng.choice(snapshots))  # one frozen opponent per iter
                snapshot_model.load_state_dict(
                    torch.load(os.path.join(args.league_dir, pick),
                               map_location=device))

        episodes, opp_counts = [], {"mirror": 0, "snapshot": 0, "heuristic": 0}
        for _ in range(args.episodes_per_iter):
            factions = ("SM", "Orks") if rng.random() < 0.5 else ("Orks", "SM")
            if args.exploit:
                opp_counts["snapshot"] += 1
                snapshot_agent.reset_memory()
                episodes.append(collect_episode(env, model, device,
                                                factions=factions,
                                                opponent=net_opponent,
                                                learner_player=int(rng.integers(2))))
                continue
            r = rng.random()
            if args.no_league or (r < MIX_MIRROR) or (not snapshots and r < MIX_MIRROR + MIX_SNAPSHOT):
                opp_counts["mirror"] += 1
                episodes.append(collect_episode(env, model, device,
                                                factions=factions))
                continue
            learner = int(rng.integers(2))
            if r < MIX_MIRROR + MIX_SNAPSHOT:
                opp_counts["snapshot"] += 1
                snapshot_agent.reset_memory()
                opponent = net_opponent
            else:
                opp_counts["heuristic"] += 1
                opponent = heuristic_opponent
            episodes.append(collect_episode(env, model, device,
                                            factions=factions,
                                            opponent=opponent,
                                            learner_player=learner))

        batch = build_batch(episodes, device,
                            gamma=args.gamma, lam=args.gae_lambda)
        stats = ppo_update(model, optimizer, batch, epochs=args.epochs)

        if it % args.snapshot_every == 0 and not (args.no_league or args.exploit):
            torch.save(model.state_dict(),
                       os.path.join(args.league_dir, f"snap_{it:05d}.pt"))

        p0_wins = sum(1 for _, w in episodes if w == 0)
        draws = sum(1 for _, w in episodes if w == -1)
        if it % 5 == 0 or it == 1:
            print(f"iter {it:4d} | loss {stats['loss']:+.3f} "
                  f"(pi {stats['policy_loss']:+.3f}, v {stats['value_loss']:.3f}, "
                  f"H {stats['entropy']:.3f}) | "
                  f"P0 wins {p0_wins}/{len(episodes)} draws {draws} | "
                  f"mix m{opp_counts['mirror']}/s{opp_counts['snapshot']}"
                  f"/h{opp_counts['heuristic']}")
            torch.save(model.state_dict(), args.out)

    torch.save(model.state_dict(), args.out)
    print(f"Done. Saved {args.out}")


if __name__ == "__main__":
    main()
