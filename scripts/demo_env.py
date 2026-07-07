"""Sanity check: play a random-vs-random battle and print it.

    python -m scripts.demo_env

No network involved — this just proves the combat env runs end to end and the
action masking is consistent.
"""

import numpy as np

from fsneural.combat_env import CombatEnv


def main():
    rng = np.random.default_rng(0)
    env = CombatEnv(seed=0)
    obs, info = env.reset()

    print(env.render(), "\n")
    steps = 0
    done = False
    while not done:
        mask = obs["action_mask"]
        legal = np.where(mask)[0]
        assert len(legal) > 0, "no legal actions but episode not done"
        action = int(rng.choice(legal))
        obs, reward, done, info = env.step(action)
        steps += 1

    print(env.render())
    print(f"\nfinished in {steps} steps | winner: {info['winner']} | morale: {info['morale']}")


if __name__ == "__main__":
    main()
