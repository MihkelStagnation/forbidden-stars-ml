"""Board env invariants (Phase 4 rung C). Pure python, no torch."""

import numpy as np

from fsneural.board_env import (
    BoardEnv, ACTION_DIM, SCALAR_FEATS, N_PLANETS, N_ROUNDS, HOME, ADJACENT,
    PHASE_BUILD, PHASE_SELECT, PHASE_DEST, UNIT_SELECT_BASE, DEST_BASE,
    MOVE_PASS,
)
from fsneural.combat_env import ACTION_DIM as COMBAT_ACTION_DIM, MAX_HAND


def _random_game(seed):
    rng = np.random.default_rng(seed)
    env = BoardEnv(seed=seed)
    obs, info = env.reset()
    steps = 0
    done = False
    while not done:
        legal = np.where(obs["action_mask"])[0]
        assert len(legal) > 0
        if env._battle is not None:
            assert legal.max() < COMBAT_ACTION_DIM
        elif obs["phase"] == PHASE_BUILD:
            assert legal.min() >= COMBAT_ACTION_DIM
        elif obs["phase"] == PHASE_SELECT:
            assert all(UNIT_SELECT_BASE <= a < UNIT_SELECT_BASE + 12
                       or a == MOVE_PASS for a in legal)
        elif obs["phase"] == PHASE_DEST:
            assert all(DEST_BASE <= a < DEST_BASE + N_PLANETS for a in legal)
        obs, reward, done, info = env.step(int(rng.choice(legal)))
        steps += 1
        assert steps < 6000, "board game failed to terminate"
    return env, info, reward


def test_board_game_terminates():
    for seed in range(20):
        env, info, reward = _random_game(seed)
        assert env.done
        assert env.winner in (0, 1, -1)
        assert reward in (-1.0, 0.0, 1.0)
        assert env.round <= N_ROUNDS


def test_obs_shapes():
    env = BoardEnv(seed=0)
    obs, _ = env.reset()
    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert obs["scalars"].shape == (SCALAR_FEATS,)


def test_reset_is_repeatable_with_seed():
    e1 = BoardEnv(seed=13); o1, _ = e1.reset()
    e2 = BoardEnv(seed=13); o2, _ = e2.reset()
    assert np.array_equal(o1["units_self"], o2["units_self"])
    assert np.array_equal(o1["scalars"], o2["scalars"])


def test_units_start_at_home_and_moves_are_adjacent():
    env = BoardEnv(seed=2)
    env.reset()
    for p in (0, 1):
        assert all(u["planet"] == HOME[p] for u in env.units[p])
    # Walk to the movement phase: both players pass the build.
    from fsneural.game_env import PASS_ACTION
    env.step(PASS_ACTION)
    env.step(PASS_ACTION)
    assert env.phase == PHASE_SELECT and env.current_player == 0
    env.step(UNIT_SELECT_BASE + 0)
    assert env.phase == PHASE_DEST
    mask = env.action_mask()
    legal_dests = [a - DEST_BASE for a in np.where(mask)[0]]
    assert set(legal_dests) == set(ADJACENT[HOME[0]])
    env.step(DEST_BASE + legal_dests[0])
    assert env.units[0][0]["planet"] == legal_dests[0]
    assert env.units[0][0]["moved"]


def test_initiative_rotates():
    env = BoardEnv(seed=4)
    env.reset()
    assert env.current_player == 0        # round 1: P0 acts first
    env.units[1] = [dict(u) for u in env.units[1]]  # no battles: stay home
    from fsneural.game_env import PASS_ACTION
    env.step(PASS_ACTION); env.step(PASS_ACTION)    # both pass build
    assert env.current_player == 0        # round 1 movement: P0 first
    env.step(MOVE_PASS); env.step(MOVE_PASS)        # both pass movement
    assert env.round == 2
    assert env.current_player == 1        # round 2: P1 acts first


def test_home_capture_wins():
    env = BoardEnv(seed=3)
    env.reset()
    # Teleport P0's army onto P1's empty home and end the round.
    for u in env.units[0]:
        u["planet"] = HOME[1]
    env.units[1] = [{"name": env.units[1][0]["name"], "planet": HOME[0],
                     "moved": False}]
    env.round = 1
    env._end_round()
    assert env.done and env.winner == 0
