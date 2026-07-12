"""Order-stack env invariants (Phase 4 rung D). Pure python, no torch."""

import numpy as np

from fsneural.order_env import (
    OrderEnv, ACTION_DIM, SCALAR_FEATS, ORDER_SCALARS, N_PLANETS, N_ROUNDS,
    HOME, ADJACENT, PLANET_VALUE, ORDER_TYPE_BASE, N_ORDER_TYPES,
    TOKENS_PER_TYPE, ORDERS_PER_ROUND, DOMINATE_MULT, STACK_ENC_DEPTH,
    ADVANCE, DEPLOY, STRATEGIZE, DOMINATE,
    PHASE_ORDER_TYPE, PHASE_ORDER_PLANET, PHASE_EXECUTE, PHASE_ADVANCE,
    PHASE_DEPLOY, PHASE_STRAT, DEST_BASE, MOVE_PASS, UNIT_SELECT_BASE,
)
from fsneural.combat_env import ACTION_DIM as COMBAT_ACTION_DIM
from fsneural.game_env import (
    BUY_UNIT_BASE, BUY_CARD_BASE, PASS_ACTION as BUILD_PASS,
)


def _random_game(seed):
    rng = np.random.default_rng(seed)
    env = OrderEnv(seed=seed)
    obs, info = env.reset()
    steps = 0
    done = False
    while not done:
        legal = np.where(obs["action_mask"])[0]
        assert len(legal) > 0
        if env._battle is not None:
            assert legal.max() < COMBAT_ACTION_DIM
        elif obs["phase"] == PHASE_ORDER_TYPE:
            assert all(ORDER_TYPE_BASE <= a < ORDER_TYPE_BASE + N_ORDER_TYPES
                       for a in legal)
        elif obs["phase"] in (PHASE_ORDER_PLANET, PHASE_EXECUTE):
            assert all(DEST_BASE <= a < DEST_BASE + N_PLANETS for a in legal)
        elif obs["phase"] == PHASE_ADVANCE:
            assert all(UNIT_SELECT_BASE <= a < UNIT_SELECT_BASE + 12
                       or a == MOVE_PASS for a in legal)
        elif obs["phase"] == PHASE_DEPLOY:
            assert all(BUY_UNIT_BASE <= a < BUY_CARD_BASE or a == BUILD_PASS
                       for a in legal)
        elif obs["phase"] == PHASE_STRAT:
            assert all(BUY_CARD_BASE <= a <= BUILD_PASS for a in legal)
        obs, reward, done, info = env.step(int(rng.choice(legal)))
        steps += 1
        assert steps < 8000, "order game failed to terminate"
    return env, info, reward


def test_order_game_terminates():
    for seed in range(20):
        env, info, reward = _random_game(seed)
        assert env.done
        assert env.winner in (0, 1, -1)
        assert reward in (-1.0, 0.0, 1.0)
        assert env.round <= N_ROUNDS
        assert all(not s for s in env.stacks.values())


def test_obs_shapes():
    env = OrderEnv(seed=0)
    obs, _ = env.reset()
    assert obs["action_mask"].shape == (ACTION_DIM,)
    assert obs["scalars"].shape == (SCALAR_FEATS,)


def _place(env, t, planet):
    env.step(ORDER_TYPE_BASE + t)
    env.step(DEST_BASE + planet)


def test_placement_alternates_and_respects_pool():
    env = OrderEnv(seed=1)
    env.reset()
    assert env.phase == PHASE_ORDER_TYPE and env.current_player == 0
    # P0 uses both Advance tokens; the third Advance must be masked out.
    _place(env, ADVANCE, 2)
    assert env.current_player == 1
    _place(env, ADVANCE, 2)              # P1 stacks on the same planet
    _place(env, ADVANCE, HOME[0])        # P0's second (last) Advance
    _place(env, DEPLOY, HOME[1])
    assert env.current_player == 0
    mask = env.action_mask()
    assert not mask[ORDER_TYPE_BASE + ADVANCE]      # pool exhausted
    assert mask[ORDER_TYPE_BASE + DEPLOY]
    # Stack on planet 2 is [P0:A, P1:A] bottom->top: P1 owns the top.
    assert env.stacks[2] == [(0, ADVANCE), (1, ADVANCE)]


def _fill_placements(env, spec):
    """spec: list of (type, planet) consumed alternately from current state."""
    for t, planet in spec:
        _place(env, t, planet)


def test_lifo_execution_and_turn_order():
    env = OrderEnv(seed=2)
    env.reset()
    # Round 1: P0 has initiative. Both bury everything on the center planet.
    _fill_placements(env, [
        (DOMINATE, 2), (DOMINATE, 2),    # P0 bottom, P1 above it
        (STRATEGIZE, 2), (STRATEGIZE, 2),
        (DEPLOY, 2), (DEPLOY, 2),
        (ADVANCE, 2), (ADVANCE, 2),      # P1's Advance ends on top
    ])
    # Operations: P0 (initiative) has no top token anywhere -> P1 executes.
    assert env.phase == PHASE_EXECUTE and env.current_player == 1
    mask = env.action_mask()
    assert list(np.where(mask)[0]) == [DEST_BASE + 2]
    env.step(DEST_BASE + 2)              # P1 reveals its top Advance
    assert env.phase == PHASE_ADVANCE and env.current_player == 1
    env.step(MOVE_PASS)                  # moves nothing: the safe bluff
    # Now P0's Advance is on top; alternation offers P0 the turn.
    assert env.phase == PHASE_EXECUTE and env.current_player == 0


def test_deploy_spawns_units_at_planet_and_fizzles_elsewhere():
    env = OrderEnv(seed=3)
    env.reset()
    env.points[0] = 10
    _fill_placements(env, [
        (DEPLOY, HOME[0]), (DEPLOY, 2),  # P1's deploy at empty center fizzles
        (DOMINATE, 1), (DOMINATE, HOME[1]),
        (STRATEGIZE, 1), (STRATEGIZE, 1),
        (ADVANCE, 3), (ADVANCE, 3),
    ])
    before = len(env.units[0])
    assert env.current_player == 0
    env.step(DEST_BASE + HOME[0])        # reveal Deploy at own home
    assert env.phase == PHASE_DEPLOY
    mask = env.action_mask()
    tiers = [a - BUY_UNIT_BASE for a in np.where(mask)[0] if a != BUILD_PASS]
    assert 0 in tiers and 3 not in tiers          # early stage: tiers 0-1
    env.step(BUY_UNIT_BASE + 0)
    assert len(env.units[0]) == before + 1
    new = env.units[0][-1]
    assert new["planet"] == HOME[0] and new["moved"]
    env.step(BUILD_PASS)
    # P1 reveals its Deploy at center planet 2 where it has no units: fizzle.
    assert env.current_player == 1
    n1 = len(env.units[1])
    env.step(DEST_BASE + 2)
    assert len(env.units[1]) == n1 and env.phase == PHASE_EXECUTE


def test_dominate_income_requires_control():
    env = OrderEnv(seed=4)
    env.reset()
    _fill_placements(env, [
        (DOMINATE, HOME[0]), (DOMINATE, 2),   # P1 dominates empty center
        (STRATEGIZE, 1), (STRATEGIZE, 1),
        (ADVANCE, 3), (ADVANCE, 3),
        (DEPLOY, HOME[0]), (DEPLOY, HOME[1]),
    ])
    # Walk the whole operations phase; track points around the dominates.
    p0_before, p1_before = env.points[0], env.points[1]
    rng = np.random.default_rng(0)
    while env.round == 1 and not env.done:
        legal = np.where(env.action_mask())[0]
        # Never buy anything so point deltas come from Dominate alone.
        passes = [a for a in legal if a in (BUILD_PASS, MOVE_PASS)]
        env.step(int(passes[0]) if passes else int(rng.choice(legal)))
    gained0 = env.points[0] - p0_before - 3   # minus round-2 base income
    gained1 = env.points[1] - p1_before - 3
    assert gained0 == PLANET_VALUE[HOME[0]] * DOMINATE_MULT   # controlled home
    assert gained1 == 0                                       # empty center


def test_advance_moves_adjacent_units_and_triggers_battle():
    env = OrderEnv(seed=5)
    env.reset()
    # March P1's army to the center so P0's advance into it is an attack.
    for u in env.units[1]:
        u["planet"] = 2
    _fill_placements(env, [
        (ADVANCE, 2), (DOMINATE, 2),         # P1 covers P0's Advance
        (DEPLOY, HOME[0]), (STRATEGIZE, HOME[0]),
        (STRATEGIZE, 1), (DOMINATE, 1),      # every P0 token gets covered,
        (DOMINATE, 3), (ADVANCE, 3),         # so P1 must execute first
    ])
    # P0 must dig its Advance out from under P1's Dominate: P1 topped it.
    assert env.current_player == 1
    # ... P1 executes first; force it to clear planet 2's top (its Dominate).
    env.step(DEST_BASE + 2)
    assert env.phase == PHASE_EXECUTE and env.current_player == 0
    env.step(DEST_BASE + 2)              # P0 reveals Advance at center
    assert env.phase == PHASE_ADVANCE
    movable = [a for a in np.where(env.action_mask())[0] if a != MOVE_PASS]
    assert movable, "home units are adjacent to the center"
    env.step(int(movable[0]))
    moved = [u for u in env.units[0] if u["planet"] == 2]
    assert len(moved) == 1 and moved[0]["moved"]
    env.step(MOVE_PASS)
    assert env._battle is not None       # contested center -> battle
    assert env._battle[1][0] == 0        # advancer is the attacker


def test_enemy_order_types_are_hidden():
    env = OrderEnv(seed=6)
    env.reset()
    _place(env, DOMINATE, 2)             # P0 places at the center
    # P1 to act: its scalar block must show planet 2's top slot as an enemy
    # token with NO type bits set.
    obs = env.observe()
    block = obs["scalars"][-ORDER_SCALARS:]
    stack_off = (ORDER_SCALARS - 2 * N_PLANETS
                 - N_PLANETS * STACK_ENC_DEPTH * (2 + N_ORDER_TYPES))
    slot = block[stack_off + 2 * STACK_ENC_DEPTH * (2 + N_ORDER_TYPES):][:6]
    assert slot[0] == 1.0                # a token exists on planet 2
    assert slot[1] == 0.0                # it is not P1's
    assert not slot[2:].any()            # and its type is invisible
    # The owner, by contrast, sees the type.
    env2 = OrderEnv(seed=6)
    env2.reset()
    _place(env2, DOMINATE, 2)
    _place(env2, ADVANCE, 3)             # P1 places elsewhere; P0 to act
    obs2 = env2.observe()
    block2 = obs2["scalars"][-ORDER_SCALARS:]
    slot2 = block2[stack_off + 2 * STACK_ENC_DEPTH * (2 + N_ORDER_TYPES):][:6]
    assert slot2[0] == 1.0 and slot2[1] == 1.0
    assert slot2[2 + DOMINATE] == 1.0


def test_home_capture_wins():
    env = OrderEnv(seed=7)
    env.reset()
    for u in env.units[0]:
        u["planet"] = HOME[1]
    env.units[1] = [{"name": env.units[1][0]["name"], "planet": HOME[0],
                     "moved": False}]
    env.stacks = {i: [] for i in range(N_PLANETS)}
    env._end_round()
    assert env.done and env.winner == 0


def test_initiative_rotates():
    env = OrderEnv(seed=8)
    env.reset()
    assert env.current_player == 0
    rng = np.random.default_rng(1)
    while env.round == 1 and not env.done:
        legal = np.where(env.action_mask())[0]
        env.step(int(rng.choice(legal)))
    if not env.done:
        assert env.phase == PHASE_ORDER_TYPE
        assert env.current_player == 1   # round 2: P1 places first
