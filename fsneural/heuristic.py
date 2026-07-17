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


def order_heuristic_action(env, p):
    """Scripted order-stack bot: purposeful placement (defend home, deploy,
    press toward the enemy home, dominate for income), execute economy
    before attacks, feed advances only when not outnumbered."""
    from .order_env import (
        ADVANCE, DEPLOY, STRATEGIZE, DOMINATE, ORDER_TYPE_BASE,
        PHASE_ORDER_TYPE, PHASE_ORDER_PLANET, PHASE_EXECUTE, PHASE_ADVANCE,
        PHASE_DEPLOY, PHASE_STRAT, DEST_BASE, MOVE_PASS, UNIT_SELECT_BASE,
        ADJACENT, DIST_TO_HOME, HOME, PLANET_VALUE, N_PLANETS,
    )
    from .game_env import (
        BUY_UNIT_BASE, BUY_CARD_BASE, PASS_ACTION, UPGRADE_IDS, CARD_COST,
        _GROUP_OF,
    )
    if env._battle is not None:
        seat = next(s for s, q in env._battle[1].items() if q == p)
        return heuristic_action(env.combat, seat)
    opp = 1 - p
    mask = env.action_mask()

    def occupied(q, i):
        return sum(1 for u in env.units[q] if u["planet"] == i)

    home_threat = occupied(opp, HOME[p]) or any(
        occupied(opp, d) for d in ADJACENT[HOME[p]])
    my_planets = [i for i in range(N_PLANETS) if env._controller(i) == p]

    def attack_target():
        """The adjacent-to-my-army planet closest to the enemy home."""
        reachable = {d for u in env.units[p] for d in ADJACENT[u["planet"]]}
        reachable |= {u["planet"] for u in env.units[p]}
        if not reachable:
            return HOME[p]
        return min(reachable, key=lambda i: (DIST_TO_HOME[opp][i],
                                             -PLANET_VALUE[i]))

    if env.phase == PHASE_ORDER_TYPE:
        already = {t for s in env.stacks.values() for o, t in s if o == p}
        prefs = []
        if home_threat:
            prefs.append(ADVANCE)                      # reinforce home first
        if DEPLOY not in already and env.points[p] >= 2:
            prefs.append(DEPLOY)
        prefs += [ADVANCE, DOMINATE, DOMINATE, STRATEGIZE, ADVANCE, DEPLOY]
        for t in prefs:
            if mask[ORDER_TYPE_BASE + t]:
                return ORDER_TYPE_BASE + t
        return int(np.where(mask)[0][0])

    if env.phase == PHASE_ORDER_PLANET:
        t = env._pending_type
        if t == ADVANCE:
            mine_on_home = any(o == p and ot == ADVANCE
                               for o, ot in env.stacks[HOME[p]])
            dest = (HOME[p] if home_threat and not mine_on_home
                    else attack_target())
        elif t == DEPLOY:
            dest = (HOME[p] if not occupied(opp, HOME[p])
                    else max(my_planets, key=lambda i: PLANET_VALUE[i],
                             default=HOME[p]))
        elif t == DOMINATE:
            dest = max(my_planets, key=lambda i: PLANET_VALUE[i],
                       default=HOME[p])
        else:  # STRATEGIZE: park it on an enemy-topped stack to delay them
            enemy_tops = [i for i in range(N_PLANETS)
                          if env.stacks[i] and env.stacks[i][-1][0] == opp]
            dest = enemy_tops[0] if enemy_tops else HOME[p]
        return DEST_BASE + dest

    if env.phase == PHASE_EXECUTE:
        prio = {DOMINATE: 0, STRATEGIZE: 1, DEPLOY: 2, ADVANCE: 3}
        tops = [i for i in range(N_PLANETS) if mask[DEST_BASE + i]]
        best = min(tops, key=lambda i: prio[env.stacks[i][-1][1]])
        return DEST_BASE + best

    if env.phase == PHASE_ADVANCE:
        target = env._executing[1]
        movable = [a for a in np.where(mask)[0] if a != MOVE_PASS]
        if not movable:
            return MOVE_PASS
        enemies = occupied(opp, target)
        committed = occupied(p, target)
        if enemies and committed + len(movable) <= enemies:
            return MOVE_PASS            # outnumbered: don't feed the battle
        return int(movable[0])

    if env.phase == PHASE_DEPLOY:
        if len(env.units[p]) < 8:
            for t in (3, 2, 1, 0):
                if mask[BUY_UNIT_BASE + t]:
                    return BUY_UNIT_BASE + t
        return PASS_ACTION

    assert env.phase == PHASE_STRAT
    best, best_cost = PASS_ACTION, -1
    for i, cid in enumerate(UPGRADE_IDS):
        if mask[BUY_CARD_BASE + i] and CARD_COST[_GROUP_OF[cid]] > best_cost:
            best, best_cost = BUY_CARD_BASE + i, CARD_COST[_GROUP_OF[cid]]
    return best


def objective_heuristic_action(env, p):
    """Rung E bot: the order-stack bot with objective-aware targeting —
    march toward planets still holding own objective tokens, defend home
    (it carries the enemy's tokens), dominate for income."""
    from .order_env import (
        ADVANCE, PHASE_ORDER_PLANET, DEST_BASE, ADJACENT, HOME, PLANET_VALUE,
        N_PLANETS,
    )
    # hop distance on this map is 0 (same), 1 (adjacent) or 2 (rest)
    def dist(a, b):
        return 0 if a == b else (1 if b in ADJACENT[a] else 2)

    if (env._battle is None and env.phase == PHASE_ORDER_PLANET
            and env._pending_type == ADVANCE):
        opp = 1 - p
        home_threat = any(u["planet"] == HOME[p] for u in env.units[opp]) or any(
            u["planet"] in ADJACENT[HOME[p]] for u in env.units[opp])
        mine_on_home = any(o == p and ot == ADVANCE
                           for o, ot in env.stacks[HOME[p]])
        if home_threat and not mine_on_home:
            return DEST_BASE + HOME[p]
        targets = set(env.objectives[p])
        reachable = {d for u in env.units[p] for d in ADJACENT[u["planet"]]}
        reachable |= {u["planet"] for u in env.units[p]}
        if targets and reachable:
            # nearest objective planet; if unreachable this round, step
            # toward it via the reachable planet closest to it
            best_t = min(targets, key=lambda t: (
                min(dist(r, t) for r in reachable), -PLANET_VALUE[t]))
            if best_t in reachable:
                return DEST_BASE + best_t
            step = min(reachable, key=lambda r: (dist(r, best_t),
                                                 -PLANET_VALUE[r]))
            return DEST_BASE + step
    return order_heuristic_action(env, p)


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
