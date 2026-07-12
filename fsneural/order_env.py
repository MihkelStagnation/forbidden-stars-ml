r"""Phase 4 rung D: ORDER STACKS — the strategic heart of Forbidden Stars.

Same 5-planet map as rung C, but the build/move phases are replaced by the
real game's planning/operations structure:

  1. PLANNING: starting with the initiative player, players alternate placing
     ORDERS_PER_ROUND face-down order tokens (from a pool of TOKENS_PER_TYPE
     each of Advance / Deploy / Strategize / Dominate) on ANY planet. Tokens
     STACK: a later token goes on top of whatever is already there — including
     the enemy's. The opponent sees WHERE each token sits and WHOSE it is,
     never WHICH order it is. This is the Stratego-like bluffing core:
     covering an enemy stack delays it (LIFO), and every placement is a
     signal that may be a lie.
  2. OPERATIONS: alternating (initiative first), each player must reveal and
     execute one of THEIR OWN tokens that is on top of a stack; a player with
     no own top token is skipped. All tokens resolve every round.
       * ADVANCE at X: pull any of your unmoved units from planets adjacent
         to X into X, one at a time, then done. If X is then contested, a
         combat-env battle resolves immediately (advancer = attacker).
         Revealing an Advance and moving nothing is legal — the safe bluff.
       * DEPLOY at X: buy units (they arrive AT X, exhausted) while you can
         afford them. Requires sole occupancy of X (or X = your empty home);
         otherwise the order fizzles.
       * STRATEGIZE: buy at most ONE deck-upgrade card (planet irrelevant —
         which makes it the natural pure-bluff/deny token).
       * DOMINATE at X: if you solely occupy X, gain PLANET_VALUE[X] *
         DOMINATE_MULT resources. Planet income flows ONLY through Dominate
         now — base income is flat — so economy competes with tempo for
         order slots.
  3. Round end: capture of the enemy home wins immediately; after N_ROUNDS
     the higher controlled value wins (equal = draw). Initiative rotates.

Deliberate simplifications vs the rulebook (rung D scope):
  * Advance pulls from ALL adjacent planets, not exactly one system.
  * No ships/transport capacity; ground units move freely along edges.
  * Tokens return to the full 2-of-each pool every round (as in FS refresh).

ACTIONS (single discrete index, masked by phase):
    0 .. 16      combat actions (battles only)
    5 .. 16      ALSO: select own unit j to advance (ADVANCE phase)
    17 .. 30     build actions: buy unit tier (DEPLOY) / buy card
                 (STRATEGIZE) / pass = done with the sub-phase
    31 .. 35     pick planet: order placement target (ORDER_PLANET phase)
                 or which own top-of-stack token to execute (EXECUTE phase)
    36           done advancing units (ADVANCE phase)
    37 .. 40     order type to place (ORDER_TYPE phase)
"""

import numpy as np

from . import game_data as gd
from .combat_env import (
    CombatEnv, ACTION_DIM as COMBAT_ACTION_DIM, MAX_HAND, MAX_UNITS,
    SCALAR_FEATS as COMBAT_SCALAR_FEATS, CARD_FEATS,
)
from .game_env import (
    UNIT_COST, CARD_COST, UPGRADE_IDS, MAX_OWNED_UPGRADES, _GROUP_OF,
    BUY_UNIT_BASE, BUY_CARD_BASE, PASS_ACTION as BUILD_PASS,
    N_BUILD_ACTIONS,
)
from .board_env import (
    PLANETS, N_PLANETS, ADJACENT, PLANET_VALUE, HOME, DIST_TO_HOME,
    N_ROUNDS, BASE_INCOME, ROSTER_CAP, START_ROSTER_TIERS, STAGE_BY_ROUND,
    MAX_UNIT_TIER_BY_STAGE, CARD_GROUPS_BY_STAGE,
    DEST_BASE, MOVE_PASS, UNIT_SELECT_BASE, UNIT_FEATS,
)

ADVANCE, DEPLOY, STRATEGIZE, DOMINATE = 0, 1, 2, 3
ORDER_NAMES = ("Advance", "Deploy", "Strategize", "Dominate")
N_ORDER_TYPES = 4
TOKENS_PER_TYPE = 2
ORDERS_PER_ROUND = 4
DOMINATE_MULT = 2
STACK_ENC_DEPTH = 4       # stack slots encoded per planet, top-down

ORDER_TYPE_BASE = MOVE_PASS + 1                          # 37
ACTION_DIM = ORDER_TYPE_BASE + N_ORDER_TYPES             # 41
# actions with stable indices (everything past the combat pointer slots),
# served by the model's plain flat head
N_FLAT_ACTIONS = ACTION_DIM - COMBAT_ACTION_DIM          # 24

PHASE_ORDER_TYPE = 5
PHASE_ORDER_PLANET = 6
PHASE_EXECUTE = 7
PHASE_ADVANCE = 8
PHASE_DEPLOY = 9
PHASE_STRAT = 10

# order scalar block appended to the combat scalars:
# [phase one-hot x7 (order_type, order_planet, execute, advance, deploy,
#  strat, battle), round/8, has_initiative, points self, points enemy,
#  owned bits self x9, enemy x9, controlled-by-self x5, by-enemy x5,
#  per-planet unit counts self x5, enemy x5, roster self, roster enemy,
#  own pool per type x4, own left to place, enemy left to place,
#  pending placement type one-hot x4 (own eyes only),
#  executing order type one-hot x4 + planet one-hot x5 (public: revealed),
#  per planet x5: STACK_ENC_DEPTH slots top-down x [exists, is_own,
#  own-type one-hot x4] (enemy token types stay hidden),
#  per planet own token count x5, enemy token count x5]
ORDER_SCALARS = (7 + 2 + 2 + 2 * len(UPGRADE_IDS) + 4 * N_PLANETS + 2
                 + N_ORDER_TYPES + 2 + N_ORDER_TYPES
                 + N_ORDER_TYPES + N_PLANETS
                 + N_PLANETS * STACK_ENC_DEPTH * (2 + N_ORDER_TYPES)
                 + 2 * N_PLANETS)
SCALAR_FEATS = COMBAT_SCALAR_FEATS + ORDER_SCALARS


class OrderEnv:
    def __init__(self, factions=("SM", "Orks"), seed=None):
        self.factions = factions
        self.rng = np.random.default_rng(seed)
        self.combat = CombatEnv(factions=factions)
        self.combat.rng = self.rng

    # ------------------------------------------------------------------ reset
    def reset(self, compositions=None, stage=None, factions=None):
        if factions is not None:
            self.factions = tuple(factions)
        self.units = {}
        for p in (0, 1):
            self.units[p] = [
                {"name": gd.UNIT_ROSTERS[self.factions[p]][t]["name"],
                 "planet": HOME[p], "moved": False}
                for t in START_ROSTER_TIERS
            ]
        self.points = {0: 0, 1: 0}
        self.owned = {0: set(), 1: set()}
        self.round = 1
        self.done = False
        self.winner = None
        self._battle = None              # (planet, {seat: player}, comps)
        self._executing = None           # (order type, planet) being resolved
        self._pending_type = None        # chosen type awaiting its planet
        self.stacks = {i: [] for i in range(N_PLANETS)}  # bottom -> top
        self._start_planning()
        self._grant_income()
        return self.observe(), self._info()

    def _start_planning(self):
        self.pool = {p: [TOKENS_PER_TYPE] * N_ORDER_TYPES for p in (0, 1)}
        self._placed = {0: 0, 1: 0}
        self.phase = PHASE_ORDER_TYPE
        self.current_player = self._initiative()

    def _grant_income(self):
        for p in (0, 1):
            self.points[p] += BASE_INCOME   # planet income only via Dominate

    def _controller(self, planet):
        occ = [p for p in (0, 1)
               if any(u["planet"] == planet for u in self.units[p])]
        return occ[0] if len(occ) == 1 else None

    def _initiative(self):
        return (self.round - 1) % 2

    def _stage(self):
        return STAGE_BY_ROUND[min(self.round, N_ROUNDS) - 1]

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        actor = self.current_player
        if self._battle is not None:
            self._step_battle(action)
        elif self.phase == PHASE_ORDER_TYPE:
            self._step_order_type(action)
        elif self.phase == PHASE_ORDER_PLANET:
            self._step_order_planet(action)
        elif self.phase == PHASE_EXECUTE:
            self._step_execute(action)
        elif self.phase == PHASE_ADVANCE:
            self._step_advance(action)
        elif self.phase == PHASE_DEPLOY:
            self._step_deploy(action)
        elif self.phase == PHASE_STRAT:
            self._step_strat(action)
        else:  # pragma: no cover
            raise AssertionError("bad phase")

        reward = 0.0
        if self.done and self.winner != -1:
            reward = 1.0 if self.winner == actor else -1.0
        return self.observe(), reward, self.done, self._info(extra_actor=actor)

    # --------------------------------------------------------------- planning
    def _step_order_type(self, action):
        t = action - ORDER_TYPE_BASE
        assert 0 <= t < N_ORDER_TYPES and self.pool[self.current_player][t] > 0, \
            "illegal order type"
        self._pending_type = t
        self.phase = PHASE_ORDER_PLANET

    def _step_order_planet(self, action):
        p = self.current_player
        planet = action - DEST_BASE
        assert 0 <= planet < N_PLANETS, "illegal placement planet"
        self.stacks[planet].append((p, self._pending_type))
        self.pool[p][self._pending_type] -= 1
        self._placed[p] += 1
        self._pending_type = None
        if self._placed[0] >= ORDERS_PER_ROUND and self._placed[1] >= ORDERS_PER_ROUND:
            self._seek_executor((self._initiative(), 1 - self._initiative()))
            return
        self.phase = PHASE_ORDER_TYPE
        other = 1 - p
        self.current_player = other if self._placed[other] < ORDERS_PER_ROUND else p

    # -------------------------------------------------------------- operations
    def _top_planets(self, p):
        return [i for i in range(N_PLANETS)
                if self.stacks[i] and self.stacks[i][-1][0] == p]

    def _seek_executor(self, order_pair):
        """Hand the turn to the first player in order_pair holding a top
        token; if neither does, all orders are resolved -> end the round."""
        for q in order_pair:
            if self._top_planets(q):
                self.phase = PHASE_EXECUTE
                self.current_player = q
                return
        self._end_round()

    def _step_execute(self, action):
        p = self.current_player
        planet = action - DEST_BASE
        assert (0 <= planet < N_PLANETS and self.stacks[planet]
                and self.stacks[planet][-1][0] == p), "not your top token"
        _, order = self.stacks[planet].pop()
        self._executing = (order, planet)
        if order == ADVANCE:
            self.phase = PHASE_ADVANCE
        elif order == DEPLOY:
            if self._can_deploy_at(p, planet):
                self.phase = PHASE_DEPLOY
            else:
                self._finish_execution()    # fizzles: planet not yours
        elif order == STRATEGIZE:
            self.phase = PHASE_STRAT
        else:  # DOMINATE
            if self._controller(planet) == p:
                self.points[p] += PLANET_VALUE[planet] * DOMINATE_MULT
            self._finish_execution()

    def _finish_execution(self):
        p = self.current_player
        self._executing = None
        self._seek_executor((1 - p, p))

    # ---------------------------------------------------------------- advance
    def _advanceable(self, p, target):
        return [j for j, u in enumerate(self.units[p][:MAX_UNITS])
                if not u["moved"] and u["planet"] in ADJACENT[target]]

    def _step_advance(self, action):
        p = self.current_player
        target = self._executing[1]
        if action == MOVE_PASS:
            if (any(u["planet"] == target for u in self.units[p])
                    and any(u["planet"] == target for u in self.units[1 - p])):
                self._start_battle(target, attacker=p)
            else:
                self._finish_execution()
            return
        idx = action - UNIT_SELECT_BASE
        assert idx in self._advanceable(p, target), "illegal advance"
        self.units[p][idx]["planet"] = target
        self.units[p][idx]["moved"] = True

    # ----------------------------------------------------------------- deploy
    def _can_deploy_at(self, p, planet):
        if any(u["planet"] == planet for u in self.units[1 - p]):
            return False
        return (planet == HOME[p]
                or any(u["planet"] == planet for u in self.units[p]))

    def _can_buy_unit(self, p, tier):
        unit = gd.UNIT_ROSTERS[self.factions[p]][tier]
        return (tier <= MAX_UNIT_TIER_BY_STAGE[self._stage()]
                and self.points[p] >= UNIT_COST[tier]
                and len(self.units[p]) < ROSTER_CAP
                and sum(1 for u in self.units[p]
                        if u["name"] == unit["name"]) < unit["unit_count"])

    def _step_deploy(self, action):
        p = self.current_player
        if action == BUILD_PASS:
            self._finish_execution()
            return
        tier = action - BUY_UNIT_BASE
        assert BUY_UNIT_BASE <= action < BUY_CARD_BASE and \
            self._can_buy_unit(p, tier), "illegal deploy purchase"
        self.points[p] -= UNIT_COST[tier]
        self.units[p].append(
            {"name": gd.UNIT_ROSTERS[self.factions[p]][tier]["name"],
             "planet": self._executing[1], "moved": True})

    # ------------------------------------------------------------- strategize
    def _can_buy_card(self, p, cid):
        group = _GROUP_OF[cid]
        return (group in CARD_GROUPS_BY_STAGE[self._stage()]
                and cid not in self.owned[p]
                and len(self.owned[p]) < MAX_OWNED_UPGRADES
                and self.points[p] >= CARD_COST[group])

    def _step_strat(self, action):
        p = self.current_player
        if action != BUILD_PASS:
            cid = UPGRADE_IDS[action - BUY_CARD_BASE]
            assert BUY_CARD_BASE <= action < BUILD_PASS and \
                self._can_buy_card(p, cid), "illegal strategize purchase"
            self.points[p] -= CARD_COST[_GROUP_OF[cid]]
            self.owned[p].add(cid)
        self._finish_execution()     # one card max, then done

    # ----------------------------------------------------------------- battle
    def _start_battle(self, planet, attacker):
        def_ = 1 - attacker
        comps = {0: [u["name"] for u in self.units[attacker]
                     if u["planet"] == planet][:MAX_UNITS],
                 1: [u["name"] for u in self.units[def_]
                     if u["planet"] == planet][:MAX_UNITS]}
        self.combat.reset(compositions=[comps[0], comps[1]],
                          stage=self._stage(),
                          factions=(self.factions[attacker], self.factions[def_]))
        for seat, q in ((0, attacker), (1, def_)):
            hand, undrawn = gd.build_deck_from(
                self.factions[q], self.owned[q], self.rng)
            self.combat.hands[seat], self.combat.undrawn[seat] = hand, undrawn
        self._battle = (planet, {0: attacker, 1: def_}, comps)
        self.phase = self.combat.phase
        self.current_player = {0: attacker, 1: def_}[self.combat.current_player]

    def _step_battle(self, action):
        assert action < COMBAT_ACTION_DIM, "non-combat action during battle"
        planet, seat_of, comps = self._battle
        _, _, cdone, _ = self.combat.step(action)
        if not cdone:
            self.phase = self.combat.phase
            self.current_player = seat_of[self.combat.current_player]
            return
        for seat in (0, 1):
            q = seat_of[seat]
            killed_names = [comps[seat][i] for i in range(len(comps[seat]))
                            if self.combat.units[seat][i]["killed"]]
            for name in killed_names:
                k = next(j for j, u in enumerate(self.units[q])
                         if u["planet"] == planet and u["name"] == name)
                self.units[q].pop(k)
        if self.combat.winner in (0, 1):
            loser = seat_of[1 - self.combat.winner]
            self._retreat(loser, planet)
        self._battle = None
        self.current_player = seat_of[0]     # the advancing player
        self._finish_execution()

    def _retreat(self, p, planet):
        survivors = [u for u in self.units[p] if u["planet"] == planet]
        if not survivors:
            return
        options = [d for d in ADJACENT[planet]
                   if not any(u["planet"] == d for u in self.units[1 - p])]
        if not options:
            self.units[p] = [u for u in self.units[p] if u["planet"] != planet]
            return
        dest = min(options, key=lambda d: (DIST_TO_HOME[p][d], d))
        for u in survivors:
            u["planet"] = dest

    # ------------------------------------------------------------- round flow
    def _end_round(self):
        assert all(not s for s in self.stacks.values())
        for p in (0, 1):
            enemy_home = HOME[1 - p]
            if (any(u["planet"] == enemy_home for u in self.units[p])
                    and not any(u["planet"] == enemy_home
                                for u in self.units[1 - p])):
                self.done = True
                self.winner = p
                return
        if self.round >= N_ROUNDS:
            self.done = True
            s = {q: sum(PLANET_VALUE[i] for i in range(N_PLANETS)
                        if self._controller(i) == q) for q in (0, 1)}
            self.winner = 0 if s[0] > s[1] else (1 if s[1] > s[0] else -1)
            return
        self.round += 1
        for p in (0, 1):
            for u in self.units[p]:
                u["moved"] = False
        self._start_planning()
        self._grant_income()

    # ------------------------------------------------------------- observation
    def action_mask(self):
        mask = np.zeros(ACTION_DIM, dtype=bool)
        if self.done:
            return mask
        p = self.current_player
        if self._battle is not None:
            mask[:COMBAT_ACTION_DIM] = self.combat.action_mask()
            return mask
        if self.phase == PHASE_ORDER_TYPE:
            for t in range(N_ORDER_TYPES):
                mask[ORDER_TYPE_BASE + t] = self.pool[p][t] > 0
        elif self.phase == PHASE_ORDER_PLANET:
            mask[DEST_BASE:DEST_BASE + N_PLANETS] = True
        elif self.phase == PHASE_EXECUTE:
            for i in self._top_planets(p):
                mask[DEST_BASE + i] = True
        elif self.phase == PHASE_ADVANCE:
            for j in self._advanceable(p, self._executing[1]):
                mask[UNIT_SELECT_BASE + j] = True
            mask[MOVE_PASS] = True
        elif self.phase == PHASE_DEPLOY:
            for tier in range(4):
                mask[BUY_UNIT_BASE + tier] = self._can_buy_unit(p, tier)
            mask[BUILD_PASS] = True
        elif self.phase == PHASE_STRAT:
            for i, cid in enumerate(UPGRADE_IDS):
                mask[BUY_CARD_BASE + i] = self._can_buy_card(p, cid)
            mask[BUILD_PASS] = True
        return mask

    def _encode_units(self, owner, viewer):
        arr = np.zeros((MAX_UNITS, UNIT_FEATS), dtype=np.float32)
        for j, rec in enumerate(self.units[owner][:MAX_UNITS]):
            u = gd.make_unit(self.factions[owner], rec["name"])
            feats = [
                u["tier"] / gd.MAX_TIER,
                u["hp"] / gd.MAX_HP,
                u["morale"] / 4.0,
                u["dice"] / 3.0,
                float(rec["moved"]), 0.0,
                1.0 if owner == viewer else 0.0,
                1.0 if u["faction"] == "SM" else 0.0,
                1.0,
                0.0,                       # rung C 'selected' slot, unused here
            ]
            planet_onehot = [0.0] * N_PLANETS
            planet_onehot[rec["planet"]] = 1.0
            arr[j] = feats + planet_onehot
        return arr

    def _encode_battle_units(self, seat_units):
        planet = self._battle[0]
        arr = np.zeros((MAX_UNITS, UNIT_FEATS), dtype=np.float32)
        arr[:, :9] = seat_units
        present = seat_units[:, -1] > 0
        arr[present, 10 + planet] = 1.0
        return arr

    def _order_block(self, p):
        opp = 1 - p
        in_battle = self._battle is not None
        block = [
            1.0 if (self.phase == PHASE_ORDER_TYPE and not in_battle) else 0.0,
            1.0 if self.phase == PHASE_ORDER_PLANET else 0.0,
            1.0 if self.phase == PHASE_EXECUTE else 0.0,
            1.0 if self.phase == PHASE_ADVANCE else 0.0,
            1.0 if self.phase == PHASE_DEPLOY else 0.0,
            1.0 if self.phase == PHASE_STRAT else 0.0,
            1.0 if in_battle else 0.0,
            self.round / N_ROUNDS,
            1.0 if self._initiative() == p else 0.0,
            self.points[p] / 15.0, self.points[opp] / 15.0,
        ]
        block += [float(cid in self.owned[p]) for cid in UPGRADE_IDS]
        block += [float(cid in self.owned[opp]) for cid in UPGRADE_IDS]
        block += [1.0 if self._controller(i) == p else 0.0 for i in range(N_PLANETS)]
        block += [1.0 if self._controller(i) == opp else 0.0 for i in range(N_PLANETS)]
        for q in (p, opp):
            block += [sum(1 for u in self.units[q] if u["planet"] == i) / 6.0
                      for i in range(N_PLANETS)]
        block += [len(self.units[p]) / ROSTER_CAP, len(self.units[opp]) / ROSTER_CAP]
        block += [self.pool[p][t] / TOKENS_PER_TYPE for t in range(N_ORDER_TYPES)]
        block += [(ORDERS_PER_ROUND - self._placed[p]) / ORDERS_PER_ROUND,
                  (ORDERS_PER_ROUND - self._placed[opp]) / ORDERS_PER_ROUND]
        pend = [0.0] * N_ORDER_TYPES
        if self._pending_type is not None and p == self.current_player:
            pend[self._pending_type] = 1.0
        block += pend
        exe_t, exe_pl = [0.0] * N_ORDER_TYPES, [0.0] * N_PLANETS
        if self._executing is not None:      # revealed -> public to both
            exe_t[self._executing[0]] = 1.0
            exe_pl[self._executing[1]] = 1.0
        block += exe_t + exe_pl
        for i in range(N_PLANETS):
            top_down = self.stacks[i][::-1]
            for d in range(STACK_ENC_DEPTH):
                slot = [0.0] * (2 + N_ORDER_TYPES)
                if d < len(top_down):
                    owner, otype = top_down[d]
                    slot[0] = 1.0
                    if owner == p:
                        slot[1] = 1.0
                        slot[2 + otype] = 1.0   # enemy types stay hidden
                block += slot
        for q in (p, opp):
            block += [sum(1 for o, _ in self.stacks[i] if o == q) / ORDERS_PER_ROUND
                      for i in range(N_PLANETS)]
        assert len(block) == ORDER_SCALARS
        return np.array(block, dtype=np.float32)

    def observe(self):
        p = self.current_player
        opp = 1 - p
        if self._battle is not None:
            obs = self.combat.observe()
            obs["units_self"] = self._encode_battle_units(obs["units_self"])
            obs["units_enemy"] = self._encode_battle_units(obs["units_enemy"])
            obs["scalars"] = np.concatenate([obs["scalars"], self._order_block(p)])
            mask = np.zeros(ACTION_DIM, dtype=bool)
            mask[:COMBAT_ACTION_DIM] = obs["action_mask"]
            obs["action_mask"] = mask
            obs["phase"] = self.combat.phase
            return obs
        combat_scalars = np.zeros(COMBAT_SCALAR_FEATS, dtype=np.float32)
        combat_scalars[5] = 1.0 if self.factions[p] == "SM" else 0.0
        for i, s in enumerate(gd.STAGES):
            combat_scalars[6 + i] = float(self._stage() == s)
        return {
            "units_self": self._encode_units(p, p),
            "units_enemy": self._encode_units(opp, p),
            "hand": np.zeros((MAX_HAND, CARD_FEATS), dtype=np.float32),
            "scalars": np.concatenate([combat_scalars, self._order_block(p)]),
            "action_mask": self.action_mask(),
            "phase": self.phase,
        }

    def _info(self, extra_actor=None):
        info = {
            "current_player": self.current_player,
            "round": self.round,
            "phase": self.phase,
            "winner": self.winner,
            "control": {q: [PLANETS[i] for i in range(N_PLANETS)
                            if self._controller(i) == q] for q in (0, 1)},
        }
        if extra_actor is not None:
            info["acting_player"] = extra_actor
        return info

    # ------------------------------------------------------------- text render
    def render(self):
        lines = [f"== round {self.round}/{N_ROUNDS} | phase {self.phase}"
                 f" | to act: P{self.current_player} =="]
        for i in range(N_PLANETS):
            stacks = []
            for p in (0, 1):
                here = [u["name"] for u in self.units[p] if u["planet"] == i]
                if here:
                    stacks.append(f"P{p}: {', '.join(here)}")
            owner = self._controller(i)
            tag = f" (P{owner})" if owner is not None else ""
            tok = "".join(f" <P{o}:{ORDER_NAMES[t][0]}>"
                          for o, t in self.stacks[i])
            lines.append(f"  {PLANETS[i]}[{PLANET_VALUE[i]}]{tag}: "
                         + ("; ".join(stacks) or "-") + tok)
        for p in (0, 1):
            lines.append(f"P{p} [{self.factions[p]}] pts {self.points[p]}"
                         f" | pool {self.pool[p]} | upgrades {sorted(self.owned[p])}")
        if self._executing is not None:
            lines.append(f"EXECUTING {ORDER_NAMES[self._executing[0]]}"
                         f" at {PLANETS[self._executing[1]]}")
        if self._battle is not None:
            lines.append(f"BATTLE at {PLANETS[self._battle[0]]}:")
            lines.append(self.combat.render())
        if self.done:
            lines.append(f"WINNER: {'draw' if self.winner == -1 else 'P' + str(self.winner)}")
        return "\n".join(lines)
