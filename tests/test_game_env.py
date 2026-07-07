"""Campaign env invariants (Phase 4 rungs A+B). Pure python, no torch."""

import numpy as np

from fsneural.game_env import (
    GameEnv, ACTION_DIM, SCALAR_FEATS, PHASE_BUILD, PASS_ACTION,
    BUY_UNIT_BASE, BUY_CARD_BASE, N_BATTLES,
)
from fsneural.combat_env import ACTION_DIM as COMBAT_ACTION_DIM


def _random_campaign(seed):
    rng = np.random.default_rng(seed)
    env = GameEnv(seed=seed)
    obs, info = env.reset()
    steps = 0
    done = False
    while not done:
        legal = np.where(obs["action_mask"])[0]
        assert len(legal) > 0
        # Phase partition: build actions during build, combat actions in battle.
        if obs["phase"] == PHASE_BUILD:
            assert legal.min() >= COMBAT_ACTION_DIM
        else:
            assert legal.max() < COMBAT_ACTION_DIM
        obs, reward, done, info = env.step(int(rng.choice(legal)))
        steps += 1
        assert steps < 2000, "campaign failed to terminate"
    return env, info, reward


def test_campaign_terminates_and_scores():
    for seed in range(25):
        env, info, reward = _random_campaign(seed)
        assert env.done
        assert env.winner in (0, 1, -1)
        assert env.battle_idx == N_BATTLES
        assert env.score[0] + env.score[1] <= N_BATTLES
        # Winner consistency with the score.
        if env.score[0] > env.score[1]:
            assert env.winner == 0
        elif env.score[1] > env.score[0]:
            assert env.winner == 1
        else:
            assert env.winner == -1
        assert reward in (-1.0, 0.0, 1.0)


def test_obs_shapes():
    env = GameEnv(seed=0)
    obs, _ = env.reset()
    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert obs["scalars"].shape == (SCALAR_FEATS,)


def test_build_economy():
    env = GameEnv(seed=1)
    env.reset()
    p0_before = env.points[0]
    # Buying a tier-0 unit costs 1 and grows the roster.
    n_before = len(env.roster[0])
    assert env.current_player == 0
    env.step(BUY_UNIT_BASE + 0)
    assert env.points[0] == p0_before - 1
    assert len(env.roster[0]) == n_before + 1
    # A bought upgrade card is owned and cannot be bought twice.
    env.step(PASS_ACTION)          # P1 passes
    env.step(BUY_CARD_BASE + 0)    # P0 buys upgrade id 5 (cost 1)
    assert 5 in env.owned[0]
    assert not env.action_mask()[BUY_CARD_BASE + 0]


def test_reset_is_repeatable_with_seed():
    e1 = GameEnv(seed=11); o1, _ = e1.reset()
    e2 = GameEnv(seed=11); o2, _ = e2.reset()
    assert np.array_equal(o1["units_self"], o2["units_self"])
    assert np.array_equal(o1["scalars"], o2["scalars"])
    assert np.array_equal(o1["action_mask"], o2["action_mask"])


def test_casualties_persist_between_battles():
    # Play random campaigns and verify rosters never grow during a battle
    # and shrink exactly by the number of killed (non-spawned) units.
    rng = np.random.default_rng(5)
    env = GameEnv(seed=5)
    obs, info = env.reset()
    done = False
    while not done:
        was_battle = env.in_battle
        legal = np.where(obs["action_mask"])[0]
        obs, _, done, info = env.step(int(rng.choice(legal)))
        if was_battle and not env.in_battle:  # battle just ended
            attacker = (env.battle_idx - 1) % 2  # the battle that just finished
            for seat in (0, 1):
                p = attacker if seat == 0 else 1 - attacker
                comp = env._battle_comps[seat]
                killed = sum(1 for i in range(len(comp))
                             if env.combat.units[seat][i]["killed"])
                assert len(env.roster[p]) == len(comp) - killed
