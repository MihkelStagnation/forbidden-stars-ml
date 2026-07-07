"""Evaluate a trained policy against a baseline.

    python -m scripts.evaluate --games 2000
    python -m scripts.evaluate --agent-player 1 --opponent heuristic

The trained agent plays `--agent-player` (default 0); the other side plays
uniformly at random over legal moves, or the scripted heuristic bot with
`--opponent heuristic`. Interpret win rates against the per-side
random-vs-random baseline from `scripts.calibrate` (the game is not
symmetric: the attacker/defender and faction matter).
"""

import argparse
import os

import numpy as np
import torch

from fsneural.combat_env import CombatEnv
from fsneural import game_env as ge
from fsneural.model import CombatNet
from fsneural.agent import PolicyAgent
from fsneural.heuristic import heuristic_action, campaign_heuristic_action


def random_action(obs, rng):
    legal = np.where(obs["action_mask"])[0]
    return int(rng.choice(legal))


def play_game(env, agent, rng, agent_player=0, deterministic=True,
              opponent="random"):
    heuristic_fn = (campaign_heuristic_action if isinstance(env, ge.GameEnv)
                    else heuristic_action)
    obs, info = env.reset()
    agent.reset_memory()
    done = False
    while not done:
        p = info["current_player"]
        if p == agent_player:
            action, _, _ = agent.act(obs, p, deterministic=deterministic)
        elif opponent == "heuristic":
            action = heuristic_fn(env, p)
        else:
            action = random_action(obs, rng)
        obs, _, done, info = env.step(action)
    return env.winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/model.pt")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--agent-player", type=int, default=0, choices=(0, 1))
    ap.add_argument("--opponent", default="random", choices=("random", "heuristic"))
    ap.add_argument("--factions", nargs=2, default=["SM", "Orks"],
                    metavar=("ATTACKER", "DEFENDER"))
    ap.add_argument("--env", default="combat", choices=("combat", "campaign"))
    args = ap.parse_args()
    if args.env == "campaign" and args.model == "checkpoints/model.pt":
        args.model = "checkpoints/campaign.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    if args.env == "campaign":
        env = ge.GameEnv(factions=tuple(args.factions), seed=args.seed)
        model = CombatNet(scalar_feats=ge.SCALAR_FEATS,
                          n_build_actions=ge.N_BUILD_ACTIONS).to(device)
    else:
        env = CombatEnv(factions=tuple(args.factions), seed=args.seed)
        model = CombatNet().to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded {args.model}")
    else:
        print(f"WARNING: {args.model} not found — evaluating an untrained net.")
    agent = PolicyAgent(model, device)

    me = args.agent_player
    wins = draws = losses = 0
    for _ in range(args.games):
        w = play_game(env, agent, rng, agent_player=me, opponent=args.opponent)
        if w == me:
            wins += 1
        elif w == -1:
            draws += 1
        else:
            losses += 1

    n = args.games
    print(f"agent (P{me}/{args.factions[me]}) vs {args.opponent} over {n} games"
          f" [{args.factions[0]} attacks {args.factions[1]}]:")
    print(f"  win {wins/n:.1%} | draw {draws/n:.1%} | loss {losses/n:.1%}")


if __name__ == "__main__":
    main()
