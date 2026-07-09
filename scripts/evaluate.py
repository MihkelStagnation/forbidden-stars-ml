"""Evaluate a trained policy against a baseline.

    python -m scripts.evaluate --games 2000
    python -m scripts.evaluate --agent-player 1 --opponent heuristic

The trained agent plays `--agent-player` (default 0); the other side plays
uniformly at random over legal moves, the scripted heuristic bot with
`--opponent heuristic`, or another checkpoint with
`--opponent model:checkpoints/other.pt` (dims must match `--env`). Interpret win rates against the per-side
random-vs-random baseline from `scripts.calibrate` (the game is not
symmetric: the attacker/defender and faction matter).
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


def random_action(obs, rng):
    legal = np.where(obs["action_mask"])[0]
    return int(rng.choice(legal))


def play_game(env, agent, rng, agent_player=0, deterministic=True,
              opponent="random", opponent_agent=None):
    heuristic_fn = (campaign_heuristic_action if isinstance(env, ge.GameEnv)
                    else board_heuristic_action if isinstance(env, be.BoardEnv)
                    else heuristic_action)
    obs, info = env.reset()
    agent.reset_memory()
    if opponent_agent is not None:
        opponent_agent.reset_memory()
    done = False
    while not done:
        p = info["current_player"]
        if p == agent_player:
            action, _, _ = agent.act(obs, p, deterministic=deterministic)
        elif opponent_agent is not None:
            action, _, _ = opponent_agent.act(obs, p,
                                              deterministic=deterministic)
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
    ap.add_argument("--opponent", default="random",
                    help="'random', 'heuristic', or 'model:<checkpoint path>'")
    ap.add_argument("--factions", nargs=2, default=["SM", "Orks"],
                    metavar=("ATTACKER", "DEFENDER"))
    ap.add_argument("--env", default="combat",
                    choices=("combat", "campaign", "board"))
    args = ap.parse_args()
    if args.env != "combat" and args.model == "checkpoints/model.pt":
        args.model = f"checkpoints/{args.env}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    if args.env == "campaign":
        env = ge.GameEnv(factions=tuple(args.factions), seed=args.seed)
        def make_model():
            return CombatNet(scalar_feats=ge.SCALAR_FEATS,
                             n_build_actions=ge.N_BUILD_ACTIONS).to(device)
    elif args.env == "board":
        env = be.BoardEnv(factions=tuple(args.factions), seed=args.seed)
        def make_model():
            return CombatNet(scalar_feats=be.SCALAR_FEATS,
                             unit_feats=be.UNIT_FEATS,
                             n_build_actions=(ge.N_BUILD_ACTIONS
                                              + be.N_PLANETS + 1)).to(device)
    else:
        env = CombatEnv(factions=tuple(args.factions), seed=args.seed)
        def make_model():
            return CombatNet().to(device)
    model = make_model()
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"Loaded {args.model}")
    else:
        print(f"WARNING: {args.model} not found — evaluating an untrained net.")
    agent = PolicyAgent(model, device)

    opponent_agent = None
    if args.opponent.startswith("model:"):
        opp_path = args.opponent.split(":", 1)[1]
        opp_model = make_model()
        opp_model.load_state_dict(torch.load(opp_path, map_location=device))
        print(f"Loaded opponent {opp_path}")
        opponent_agent = PolicyAgent(opp_model, device)
    elif args.opponent not in ("random", "heuristic"):
        ap.error(f"unknown opponent {args.opponent!r}")

    me = args.agent_player
    wins = draws = losses = 0
    for _ in range(args.games):
        w = play_game(env, agent, rng, agent_player=me, opponent=args.opponent,
                      opponent_agent=opponent_agent)
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
