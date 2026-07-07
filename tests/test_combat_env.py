"""Env invariants for the ported combat model. Run with: pytest

These don't need torch — they exercise the pure-python combat env so you can
validate the rules layer independently of the learning stack.
"""

import numpy as np

from fsneural.combat_env import (
    CombatEnv, ACTION_DIM, MAX_HAND, PHASE_PLAY, PHASE_ASSIGN,
)
from fsneural.utils import DICE_CAP, pool_size


def _random_rollout(seed):
    rng = np.random.default_rng(seed)
    env = CombatEnv(seed=seed)
    obs, info = env.reset()
    steps = 0
    done = False
    while not done:
        legal = np.where(obs["action_mask"])[0]
        # Invariant: a live, non-terminal state always offers a legal action.
        assert len(legal) > 0
        # Invariant: legal actions respect the phase partition.
        if obs["phase"] == PHASE_PLAY:
            assert legal.max() < MAX_HAND
        elif obs["phase"] == PHASE_ASSIGN:
            assert legal.min() >= MAX_HAND
        action = int(rng.choice(legal))
        obs, reward, done, info = env.step(action)
        steps += 1
        assert steps < 500, "episode failed to terminate"
    return env, info, reward


def test_episode_terminates_and_has_winner():
    for seed in range(50):
        env, info, reward = _random_rollout(seed)
        assert env.done
        assert env.winner in (0, 1, -1)
        assert reward in (-1.0, 0.0, 1.0)
        # Draws only on a simultaneous wipe (defender wins morale ties).
        if env.winner == -1:
            assert not env.alive(0) and not env.alive(1)


def test_action_mask_shape():
    env = CombatEnv(seed=0)
    obs, _ = env.reset()
    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert obs["action_mask"].dtype == bool


def test_reset_is_repeatable_with_seed():
    e1 = CombatEnv(seed=7); o1, _ = e1.reset()
    e2 = CombatEnv(seed=7); o2, _ = e2.reset()
    assert np.array_equal(o1["units_self"], o2["units_self"])
    assert np.array_equal(o1["hand"], o2["hand"])
    assert np.array_equal(o1["scalars"], o2["scalars"])
    assert np.array_equal(o1["action_mask"], o2["action_mask"])


def test_dice_pool_capped_and_deck_composition():
    for seed in range(20):
        env = CombatEnv(seed=seed)
        env.reset()
        for p in (0, 1):
            assert pool_size(env.dice[p]) <= DICE_CAP
            cards = [c for c in env.hands[p] if c is not None] + env.undrawn[p]
            assert len(cards) == 10
            ids = sorted(c["id"] for c in cards)
            assert len(set(ids)) == 5           # 5 distinct card ids
            assert all(ids.count(i) == 2 for i in set(ids))  # exactly 2 copies


def test_faction_role_swap():
    env = CombatEnv(seed=3)
    obs, _ = env.reset(factions=("Orks", "SM"))
    assert env.factions == ("Orks", "SM")
    # P0 (viewer) is now Orks -> the is-SM unit feature must be 0 for own units.
    present = obs["units_self"][:, -1] > 0
    assert present.any() and (obs["units_self"][present, -2] == 0.0).all()
    # A full rollout still terminates under swapped roles.
    rng = np.random.default_rng(3)
    done, steps = False, 0
    while not done:
        legal = np.where(obs["action_mask"])[0]
        obs, _, done, _ = env.step(int(rng.choice(legal)))
        steps += 1
        assert steps < 500


def test_defender_wins_morale_ties():
    env = CombatEnv(seed=0)
    env.reset(compositions=[["Scouts"], ["Ork Boyz", "Ork Boyz"]])
    # Scouts morale 2 vs 2x Ork Boyz morale 1+1 -> tied; icons tied too.
    env._last_morale_icons = {0: 0, 1: 0}
    env._finish()
    assert env.morale_score(0) == env.morale_score(1) == 2
    assert env.winner == 1


def test_damage_assignment_kills_or_routs():
    env = CombatEnv(seed=0)
    env.reset(compositions=[["Space Marines", "Scouts"], ["Ork Boyz"]])
    env.phase = PHASE_ASSIGN
    env.current_player = 0
    # Damage >= hp kills outright.
    env._pending[0] = 3
    env.step(MAX_HAND + 0)  # Space Marines, hp 3
    assert env.units[0][0]["killed"]
    # A smaller final hit only routs.
    env.phase = PHASE_ASSIGN
    env.current_player = 0
    env._pending[0] = 1
    env.step(MAX_HAND + 1)  # Scouts, hp 2
    assert env.units[0][1]["routed"] and not env.units[0][1]["killed"]
