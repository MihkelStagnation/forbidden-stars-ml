"""Objective env invariants (Phase 4 rung E). Pure python, no torch."""

import numpy as np

from fsneural.objective_env import (
    ObjectiveEnv, SCALAR_FEATS, OBJ_SCALARS, N_OBJECTIVES, OBJECTIVE_SPOTS,
)
from fsneural.order_env import (
    ACTION_DIM, N_PLANETS, N_ROUNDS, HOME,
)


def _random_game(seed):
    rng = np.random.default_rng(seed)
    env = ObjectiveEnv(seed=seed)
    obs, info = env.reset()
    steps = 0
    done = False
    while not done:
        legal = np.where(obs["action_mask"])[0]
        assert len(legal) > 0
        obs, reward, done, info = env.step(int(rng.choice(legal)))
        steps += 1
        assert steps < 8000, "objective game failed to terminate"
    return env, info, reward


def test_objective_game_terminates():
    for seed in range(20):
        env, info, reward = _random_game(seed)
        assert env.done
        assert env.winner in (0, 1, -1)
        assert reward in (-1.0, 0.0, 1.0)
        assert env.round <= N_ROUNDS
        for p in (0, 1):
            assert env.claimed[p] + len(env.objectives[p]) >= N_OBJECTIVES


def test_obs_shapes():
    env = ObjectiveEnv(seed=0)
    obs, _ = env.reset()
    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert obs["scalars"].shape == (SCALAR_FEATS,)


def test_setup_is_symmetric_under_map_automorphism():
    # the map automorphism is H0<->H1 (0<->4), A<->B (1<->3), M fixed
    swap = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
    mirrored = sorted(swap[i] for i in OBJECTIVE_SPOTS[0])
    assert mirrored == sorted(OBJECTIVE_SPOTS[1])


def test_claim_one_token_per_controlled_planet_per_round():
    env = ObjectiveEnv(seed=1)
    env.reset()
    # P0 solely occupies H1 (which carries TWO of its tokens) and M (one).
    for u in env.units[0]:
        u["planet"] = HOME[1]
    env.units[0][0]["planet"] = 2
    # P1 parks on flank A, which carries a P0 token but none of P1's
    env.units[1] = [{"name": env.units[1][0]["name"], "planet": 1,
                     "moved": False}]
    env.stacks = {i: [] for i in range(N_PLANETS)}
    env.round = 1
    env._end_round()
    # one claim from H1 (not both) + one from M = 2; game continues
    assert env.claimed[0] == 2 and not env.done
    assert env.objectives[0].count(HOME[1]) == 1
    # P1 controls A but holds no tokens there — and P0's token at A is
    # safe from being claimed by its owner while the enemy sits on it
    assert env.claimed[1] == 0
    assert env.objectives[0].count(1) == 1


def test_collecting_all_tokens_wins_immediately():
    env = ObjectiveEnv(seed=2)
    env.reset()
    env.objectives[0] = [2]          # one token left, on the center
    env.claimed[0] = N_OBJECTIVES - 1
    for u in env.units[0]:
        u["planet"] = 2
    env.units[1] = [{"name": env.units[1][0]["name"], "planet": HOME[0],
                     "moved": False}]
    env.stacks = {i: [] for i in range(N_PLANETS)}
    env.round = 1
    env._end_round()
    assert env.done and env.winner == 0


def test_home_capture_alone_no_longer_wins():
    env = ObjectiveEnv(seed=3)
    env.reset()
    # P0 sits on the enemy home (rung C/D instant win) with no prior claims:
    # it claims ONE token there but the game continues.
    for u in env.units[0]:
        u["planet"] = HOME[1]
    env.units[1] = [{"name": env.units[1][0]["name"], "planet": HOME[0],
                     "moved": False}]
    env.stacks = {i: [] for i in range(N_PLANETS)}
    env.round = 1
    env._end_round()
    assert not env.done
    assert env.claimed[0] == 1 and env.round == 2


def test_timeout_scores_by_tokens_then_value():
    env = ObjectiveEnv(seed=4)
    env.reset()
    env.claimed = {0: 2, 1: 1}
    env.stacks = {i: [] for i in range(N_PLANETS)}
    env.round = N_ROUNDS
    env._end_round()
    assert env.done and env.winner == 0   # more tokens claimed wins


def test_obs_encodes_objective_state():
    env = ObjectiveEnv(seed=5)
    obs, _ = env.reset()
    block = obs["scalars"][-OBJ_SCALARS:]
    # P0 to act: its own remaining tokens are the first 5 entries
    mine = block[:N_PLANETS] * 2.0
    for i in range(N_PLANETS):
        assert mine[i] == OBJECTIVE_SPOTS[0].count(i)
    assert block[-2] == 0.0 and block[-1] == 0.0   # nothing claimed yet
