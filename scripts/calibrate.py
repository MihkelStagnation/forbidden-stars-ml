"""Random-vs-random calibration of the combat model.

    python -m scripts.calibrate --games 20000

Plays uniform-random-vs-uniform-random battles and reports attacker (P0) win
rates per stage and overall. Compare against the ForbiddenStarsFight reference
figures (docs/reference_combat_model.md §5): SM vs Orks should land within a
few points of 50-50; "early Orks vs SM" sat near the reference's upper
balance threshold (57%).

Differences from the reference that shift the numbers slightly: uniform card
choice (reference plays a fixed weighted order) and uniform damage-target
choice (reference uses a tier-preference heuristic).
"""

import argparse

import numpy as np

from fsneural.combat_env import CombatEnv
from fsneural.game_data import STAGES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--factions", nargs=2, default=["SM", "Orks"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = CombatEnv(factions=tuple(args.factions), seed=args.seed)
    per_stage = {s: {"win": 0, "draw": 0, "loss": 0, "n": 0} for s in STAGES}

    per_game = max(1, args.games // len(STAGES))
    for stage in STAGES:
        for _ in range(per_game):
            obs, info = env.reset(stage=stage)
            done = False
            while not done:
                legal = np.where(obs["action_mask"])[0]
                obs, _, done, info = env.step(int(rng.choice(legal)))
            bucket = per_stage[stage]
            bucket["n"] += 1
            key = {0: "win", 1: "loss", -1: "draw"}[env.winner]
            bucket[key] += 1

    fa, fd = args.factions
    print(f"random {fa} (attacker/P0) vs random {fd} (defender/P1):")
    tot = {"win": 0, "draw": 0, "loss": 0, "n": 0}
    for stage, b in per_stage.items():
        for k in tot:
            tot[k] += b[k]
        print(f"  {stage:5s}: win {b['win']/b['n']:.1%} | draw {b['draw']/b['n']:.1%}"
              f" | loss {b['loss']/b['n']:.1%}  (n={b['n']})")
    print(f"  TOTAL: win {tot['win']/tot['n']:.1%} | draw {tot['draw']/tot['n']:.1%}"
          f" | loss {tot['loss']/tot['n']:.1%}  (n={tot['n']})")


if __name__ == "__main__":
    main()
