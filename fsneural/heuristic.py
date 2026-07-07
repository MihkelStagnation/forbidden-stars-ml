"""A scripted baseline bot — the measuring stick for learned policies.

Beats uniform-random by roughly +12 points on both sides (2026-07-07 probe),
which bounds from below how much edge decisions offer in this game. A trained
agent should first beat random, then approach/beat this.

PLAY: attacker leans Bolter, defender leans Shield, everyone leans Morale in
the final round. ASSIGN: absorb the whole pending pool with the least-morale
unit big enough to soak it (rout-only); otherwise sacrifice the cheapest
morale-per-hp unit.
"""

import numpy as np

from .combat_env import MAX_HAND, PHASE_PLAY


def campaign_heuristic_action(env, p):
    """Scripted campaign bot: greedy army-first build, then the combat
    heuristic (translated to the player's combat seat this battle).

    Build: buy the highest-tier affordable unit while the roster is small,
    then the most expensive affordable deck upgrade, else pass."""
    from .game_env import (
        BUY_UNIT_BASE, BUY_CARD_BASE, PASS_ACTION, UPGRADE_IDS, CARD_COST,
        _GROUP_OF,
    )
    if not env.in_battle:
        mask = env.action_mask()
        if len(env.roster[p]) < 6:
            for t in (3, 2, 1, 0):
                if mask[BUY_UNIT_BASE + t]:
                    return BUY_UNIT_BASE + t
        best, best_cost = None, -1
        for i, cid in enumerate(UPGRADE_IDS):
            if mask[BUY_CARD_BASE + i] and CARD_COST[_GROUP_OF[cid]] > best_cost:
                best, best_cost = BUY_CARD_BASE + i, CARD_COST[_GROUP_OF[cid]]
        if best is not None:
            return best
        if mask[PASS_ACTION]:
            return PASS_ACTION
        return int(np.where(mask)[0][0])  # forced rebuild of an empty roster
    return heuristic_action(env.combat, env.combat_seat(p))


def board_heuristic_action(env, p):
    """Scripted board bot: greedy build, defend home when invaded, otherwise
    march everything toward the enemy home; combat heuristic in battles."""
    from .board_env import (
        PHASE_BUILD, PHASE_SELECT, PHASE_DEST, UNIT_SELECT_BASE, DEST_BASE,
        MOVE_PASS, ADJACENT, DIST_TO_HOME, HOME, PLANET_VALUE,
    )
    from .game_env import (
        BUY_UNIT_BASE, BUY_CARD_BASE, PASS_ACTION, UPGRADE_IDS, CARD_COST,
        _GROUP_OF,
    )
    if env._battle is not None:
        seat = next(s for s, q in env._battle[1].items() if q == p)
        return heuristic_action(env.combat, seat)
    mask = env.action_mask()
    if env.phase == PHASE_BUILD:
        if len(env.units[p]) < 8:
            for t in (3, 2, 1, 0):
                if mask[BUY_UNIT_BASE + t]:
                    return BUY_UNIT_BASE + t
        best, best_cost = None, -1
        for i, cid in enumerate(UPGRADE_IDS):
            if mask[BUY_CARD_BASE + i] and CARD_COST[_GROUP_OF[cid]] > best_cost:
                best, best_cost = BUY_CARD_BASE + i, CARD_COST[_GROUP_OF[cid]]
        return best if best is not None else PASS_ACTION
    invaded = any(u["planet"] == HOME[p] for u in env.units[1 - p])
    dist = DIST_TO_HOME[p] if invaded else DIST_TO_HOME[1 - p]
    if env.phase == PHASE_SELECT:
        # Move units not already at the objective (a selected unit MUST move).
        movable = [j for j in range(len(env.units[p]))
                   if mask[UNIT_SELECT_BASE + j]
                   and dist[env.units[p][j]["planet"]] > 0]
        if not movable:
            return MOVE_PASS
        return UNIT_SELECT_BASE + movable[0]
    assert env.phase == PHASE_DEST
    unit = env.units[p][env._selected]
    options = [d for d in ADJACENT[unit["planet"]] if mask[DEST_BASE + d]]
    dest = min(options, key=lambda d: (dist[d], -PLANET_VALUE[d]))
    return DEST_BASE + dest


def heuristic_action(env, p):
    if env.phase == PHASE_PLAY:
        best, best_v = None, -1e9
        for i, c in enumerate(env.hands[p]):
            if c is None:
                continue
            if p == 0:
                v = 1.0 * c["Bolter"] + 0.6 * c["Shield"] + 0.4 * c["Morale"]
            else:
                v = 0.6 * c["Bolter"] + 1.0 * c["Shield"] + 0.4 * c["Morale"]
            if env.round == 3:
                v += 0.8 * c["Morale"]
            if v > best_v:
                best, best_v = i, v
        return best

    pend = env._pending[p]
    targets = [(j, u) for j, u in enumerate(env.units[p]) if not u["killed"]]
    unrouted = [(j, u) for j, u in targets if not u["routed"]]
    pool = unrouted or targets
    soakers = [(j, u) for j, u in pool if u["hp"] > pend]
    if soakers:
        j, _ = min(soakers, key=lambda t: t[1]["morale"])
        return MAX_HAND + j
    j, _ = min(pool, key=lambda t: t[1]["morale"] / t[1]["hp"])
    return MAX_HAND + j
