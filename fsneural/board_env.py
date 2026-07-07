r"""Phase 4 rung C: a small BOARD environment — position, territory, maneuver.

A war on a fixed 5-planet map over N_ROUNDS rounds. Same Gym-style interface
as CombatEnv/GameEnv; the combat env resolves battles wherever opposing
forces share a planet.

Map (symmetric diamond; planet values in brackets):

        A[1]
       /    \
  H0[2] - M[2] - H1[2]
       \    /
        B[1]

H0/H1 are the players' home planets; every planet pair shown is adjacent,
plus A-M and M-B. Controlling a planet (sole occupancy) yields its value as
income each round.

Round structure:
  1. BUILD: income = 3 + controlled planet values; players alternate public
     purchases exactly as in the campaign env (units deploy AT HOME — a
     deliberate simplification; deploy is blocked while the enemy holds your
     home). Pass is binding.
  2. MOVE: players alternate moving ONE unit at a time — an AUTOREGRESSIVE
     compound action taken as two sequenced decisions: SELECT one of your
     units (reusing the unit-pointer action slots), then pick an adjacent
     DESTINATION planet. Each unit moves at most once per round. Moving onto
     an enemy-held planet is an attack. Pass is binding for the round.
  3. RESOLVE: every contested planet hosts a combat-env battle (attacker =
     whoever moved in most recently this round; battle decks come from each
     player's owned upgrades). Killed units are gone; the loser's survivors
     retreat to an adjacent enemy-free planet nearest their home (destroyed
     if none exists); routed survivors rally.

Victory: solely occupying the enemy home at the end of any round wins
immediately. Otherwise after N_ROUNDS rounds the higher total controlled
planet value wins; equal value is a draw. Zero-sum terminal reward only.

ACTIONS (single discrete index, masked by phase):
    0 .. 16      combat actions (battles only)
    5 .. 16      ALSO: select own unit j to move (SELECT phase — the unit
                 slots are reused; phases are disjoint so masks disambiguate)
    17 .. 30     build actions (buy unit tier / buy upgrade card / pass)
    31 .. 35     move destination planet (DEST phase)
    36           pass moving for this round
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

PLANETS = ("H0", "A", "M", "B", "H1")
N_PLANETS = 5
ADJACENT = {0: (1, 2, 3), 1: (0, 2, 4), 2: (0, 1, 3, 4), 3: (0, 2, 4), 4: (1, 2, 3)}
PLANET_VALUE = (2, 1, 2, 1, 2)
HOME = {0: 0, 1: 4}
# hop-distance to each player's home (this map: home 0, its neighbours 1,
# the far home 2)
DIST_TO_HOME = {p: tuple(0 if i == HOME[p] else (1 if i in ADJACENT[HOME[p]] else 2)
                         for i in range(N_PLANETS)) for p in (0, 1)}

N_ROUNDS = 8
BASE_INCOME = 3
ROSTER_CAP = 12
START_ROSTER_TIERS = (0, 0, 1)
STAGE_BY_ROUND = ("early",) * 3 + ("mid",) * 3 + ("late",) * 2
MAX_UNIT_TIER_BY_STAGE = {"early": 1, "mid": 2, "late": 3}
CARD_GROUPS_BY_STAGE = {"early": ("t0",), "mid": ("t0", "t2"),
                        "late": ("t0", "t2", "t3")}

DEST_BASE = COMBAT_ACTION_DIM + N_BUILD_ACTIONS          # 31
MOVE_PASS = DEST_BASE + N_PLANETS                        # 36
ACTION_DIM = MOVE_PASS + 1                               # 37
UNIT_SELECT_BASE = MAX_HAND                              # unit slots 5..16

# Board unit features = combat's 9 + [selected] + planet one-hot.
UNIT_FEATS = 9 + 1 + N_PLANETS

# board scalar block appended to the combat scalars:
# [in_build, in_select, in_dest, in_battle, round/8,
#  points self, points enemy, owned bits self x9, enemy x9,
#  per-planet: controlled-by-self x5, by-enemy x5,
#  per-planet unit counts self x5, enemy x5,
#  selected-unit planet one-hot x5, roster self, roster enemy]
BOARD_SCALARS = 4 + 1 + 2 + 2 * len(UPGRADE_IDS) + 4 * N_PLANETS + N_PLANETS + 2
SCALAR_FEATS = COMBAT_SCALAR_FEATS + BOARD_SCALARS

PHASE_BUILD = 2
PHASE_SELECT = 3
PHASE_DEST = 4


class BoardEnv:
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
        self.phase = PHASE_BUILD
        self.current_player = 0
        self._passed = {0: False, 1: False}
        self._selected = None            # roster index during DEST phase
        self._last_entrant = {}          # planet -> player who moved in last
        self._battle_queue = []
        self._battle = None              # (planet, {seat: campaign player}, comps)
        self.done = False
        self.winner = None
        self._grant_income()
        return self.observe(), self._info()

    def _grant_income(self):
        for p in (0, 1):
            self.points[p] += BASE_INCOME + sum(
                PLANET_VALUE[i] for i in range(N_PLANETS)
                if self._controller(i) == p)

    def _controller(self, planet):
        """Sole occupant of a planet, or None."""
        occ = [p for p in (0, 1)
               if any(u["planet"] == planet for u in self.units[p])]
        return occ[0] if len(occ) == 1 else None

    def _stage(self):
        return STAGE_BY_ROUND[min(self.round, N_ROUNDS) - 1]

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        actor = self.current_player
        if self._battle is not None:
            self._step_battle(action)
        elif self.phase == PHASE_BUILD:
            self._step_build(action)
        elif self.phase == PHASE_SELECT:
            self._step_select(action)
        elif self.phase == PHASE_DEST:
            self._step_dest(action)
        else:  # pragma: no cover
            raise AssertionError("bad phase")

        reward = 0.0
        if self.done and self.winner != -1:
            reward = 1.0 if self.winner == actor else -1.0
        return self.observe(), reward, self.done, self._info(extra_actor=actor)

    # ------------------------------------------------------------------ build
    def _can_buy_unit(self, p, tier):
        unit = gd.UNIT_ROSTERS[self.factions[p]][tier]
        home_free = self._controller(HOME[p]) in (p, None) and not any(
            u["planet"] == HOME[p] for u in self.units[1 - p])
        return (tier <= MAX_UNIT_TIER_BY_STAGE[self._stage()]
                and self.points[p] >= UNIT_COST[tier]
                and len(self.units[p]) < ROSTER_CAP
                and home_free
                and sum(1 for u in self.units[p]
                        if u["name"] == unit["name"]) < unit["unit_count"])

    def _can_buy_card(self, p, cid):
        group = _GROUP_OF[cid]
        return (group in CARD_GROUPS_BY_STAGE[self._stage()]
                and cid not in self.owned[p]
                and len(self.owned[p]) < MAX_OWNED_UPGRADES
                and self.points[p] >= CARD_COST[group])

    def _step_build(self, action):
        p = self.current_player
        if action == BUILD_PASS:
            self._passed[p] = True
        elif BUY_UNIT_BASE <= action < BUY_CARD_BASE:
            tier = action - BUY_UNIT_BASE
            assert self._can_buy_unit(p, tier), "illegal unit purchase"
            self.points[p] -= UNIT_COST[tier]
            self.units[p].append(
                {"name": gd.UNIT_ROSTERS[self.factions[p]][tier]["name"],
                 "planet": HOME[p], "moved": False})
        elif BUY_CARD_BASE <= action < BUILD_PASS:
            cid = UPGRADE_IDS[action - BUY_CARD_BASE]
            assert self._can_buy_card(p, cid), "illegal card purchase"
            self.points[p] -= CARD_COST[_GROUP_OF[cid]]
            self.owned[p].add(cid)
        else:
            raise AssertionError("bad build action")

        if self._passed[0] and self._passed[1]:
            self.phase = PHASE_SELECT
            self._passed = {0: False, 1: False}
            self.current_player = 0
            return
        other = 1 - p
        self.current_player = other if not self._passed[other] else p

    # --------------------------------------------------------------- movement
    def _movable(self, p):
        return [j for j, u in enumerate(self.units[p]) if not u["moved"]]

    def _step_select(self, action):
        p = self.current_player
        if action == MOVE_PASS:
            self._passed[p] = True
            if self._passed[0] and self._passed[1]:
                self._queue_battles()
                return
            other = 1 - p
            self.current_player = other if not self._passed[other] else p
            return
        idx = action - UNIT_SELECT_BASE
        assert idx in self._movable(p), "selected an immovable unit"
        self._selected = idx
        self.phase = PHASE_DEST     # same player picks the destination next

    def _step_dest(self, action):
        p = self.current_player
        dest = action - DEST_BASE
        unit = self.units[p][self._selected]
        assert dest in ADJACENT[unit["planet"]], "illegal destination"
        unit["planet"] = dest
        unit["moved"] = True
        self._selected = None
        self._last_entrant[dest] = p
        self.phase = PHASE_SELECT
        other = 1 - p
        self.current_player = other if not self._passed[other] else p

    # ----------------------------------------------------------------- battle
    def _queue_battles(self):
        self._battle_queue = [
            i for i in range(N_PLANETS)
            if any(u["planet"] == i for u in self.units[0])
            and any(u["planet"] == i for u in self.units[1])
        ]
        self._next_battle_or_round()

    def _next_battle_or_round(self):
        while self._battle_queue:
            planet = self._battle_queue.pop(0)
            att = self._last_entrant.get(planet, 0)
            def_ = 1 - att
            comps = {0: [u["name"] for u in self.units[att] if u["planet"] == planet],
                     1: [u["name"] for u in self.units[def_] if u["planet"] == planet]}
            if not comps[0] or not comps[1]:
                continue  # a retreat emptied this planet before its battle
            self.combat.reset(compositions=[comps[0][:MAX_UNITS], comps[1][:MAX_UNITS]],
                              stage=self._stage(),
                              factions=(self.factions[att], self.factions[def_]))
            for seat, q in ((0, att), (1, def_)):
                hand, undrawn = gd.build_deck_from(
                    self.factions[q], self.owned[q], self.rng)
                self.combat.hands[seat], self.combat.undrawn[seat] = hand, undrawn
            self._battle = (planet, {0: att, 1: def_}, comps)
            self.phase = self.combat.phase
            self.current_player = {0: att, 1: def_}[self.combat.current_player]
            return
        self._end_round()

    def _step_battle(self, action):
        assert action < COMBAT_ACTION_DIM, "non-combat action during battle"
        planet, seat_of, comps = self._battle
        _, _, cdone, _ = self.combat.step(action)
        if not cdone:
            self.phase = self.combat.phase
            self.current_player = seat_of[self.combat.current_player]
            return
        # Apply casualties: remove killed units (spawned ones never joined).
        for seat in (0, 1):
            q = seat_of[seat]
            killed_names = [comps[seat][i] for i in range(len(comps[seat]))
                            if self.combat.units[seat][i]["killed"]]
            for name in killed_names:
                k = next(j for j, u in enumerate(self.units[q])
                         if u["planet"] == planet and u["name"] == name)
                self.units[q].pop(k)
        # Loser's survivors retreat (winner -1 = mutual wipe, nobody retreats).
        if self.combat.winner in (0, 1):
            loser = seat_of[1 - self.combat.winner]
            self._retreat(loser, planet)
        self._battle = None
        self._next_battle_or_round()

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
        # Immediate win: solely occupying the enemy home.
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
        self.phase = PHASE_BUILD
        self.current_player = 0
        self._passed = {0: False, 1: False}
        self._last_entrant = {}
        for p in (0, 1):
            for u in self.units[p]:
                u["moved"] = False
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
        if self.phase == PHASE_BUILD:
            for tier in range(4):
                mask[BUY_UNIT_BASE + tier] = self._can_buy_unit(p, tier)
            for i, cid in enumerate(UPGRADE_IDS):
                mask[BUY_CARD_BASE + i] = self._can_buy_card(p, cid)
            mask[BUILD_PASS] = True
        elif self.phase == PHASE_SELECT:
            for j in self._movable(p)[:MAX_UNITS]:
                mask[UNIT_SELECT_BASE + j] = True
            mask[MOVE_PASS] = True
        elif self.phase == PHASE_DEST:
            for d in ADJACENT[self.units[p][self._selected]["planet"]]:
                mask[DEST_BASE + d] = True
        return mask

    def _encode_units(self, owner, viewer):
        arr = np.zeros((MAX_UNITS, UNIT_FEATS), dtype=np.float32)
        selected = (self._selected if (owner == viewer == self.current_player
                                       and self.phase == PHASE_DEST) else None)
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
                1.0 if j == selected else 0.0,
            ]
            planet_onehot = [0.0] * N_PLANETS
            planet_onehot[rec["planet"]] = 1.0
            arr[j] = feats + planet_onehot
        return arr

    def _encode_battle_units(self, seat_units):
        """Pad combat's 9-feature unit rows to board width; all units of the
        current battle share its planet."""
        planet = self._battle[0]
        arr = np.zeros((MAX_UNITS, UNIT_FEATS), dtype=np.float32)
        arr[:, :9] = seat_units
        present = seat_units[:, -1] > 0
        arr[present, 10 + planet] = 1.0
        return arr

    def _board_block(self, p):
        opp = 1 - p
        block = [
            1.0 if (self.phase == PHASE_BUILD and self._battle is None) else 0.0,
            1.0 if self.phase == PHASE_SELECT else 0.0,
            1.0 if self.phase == PHASE_DEST else 0.0,
            1.0 if self._battle is not None else 0.0,
            self.round / N_ROUNDS,
            self.points[p] / 15.0, self.points[opp] / 15.0,
        ]
        block += [float(cid in self.owned[p]) for cid in UPGRADE_IDS]
        block += [float(cid in self.owned[opp]) for cid in UPGRADE_IDS]
        block += [1.0 if self._controller(i) == p else 0.0 for i in range(N_PLANETS)]
        block += [1.0 if self._controller(i) == opp else 0.0 for i in range(N_PLANETS)]
        for q in (p, opp):
            block += [sum(1 for u in self.units[q] if u["planet"] == i) / 6.0
                      for i in range(N_PLANETS)]
        sel = [0.0] * N_PLANETS
        if self._selected is not None and p == self.current_player:
            sel[self.units[p][self._selected]["planet"]] = 1.0
        block += sel
        block += [len(self.units[p]) / ROSTER_CAP, len(self.units[opp]) / ROSTER_CAP]
        return np.array(block, dtype=np.float32)

    def observe(self):
        p = self.current_player
        opp = 1 - p
        if self._battle is not None:
            obs = self.combat.observe()
            obs["units_self"] = self._encode_battle_units(obs["units_self"])
            obs["units_enemy"] = self._encode_battle_units(obs["units_enemy"])
            obs["scalars"] = np.concatenate([obs["scalars"], self._board_block(p)])
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
            "scalars": np.concatenate([combat_scalars, self._board_block(p)]),
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
            lines.append(f"  {PLANETS[i]}[{PLANET_VALUE[i]}]{tag}: "
                         + ("; ".join(stacks) or "-"))
        for p in (0, 1):
            lines.append(f"P{p} [{self.factions[p]}] pts {self.points[p]}"
                         f" | upgrades {sorted(self.owned[p])}")
        if self._battle is not None:
            lines.append(f"BATTLE at {PLANETS[self._battle[0]]}:")
            lines.append(self.combat.render())
        if self.done:
            lines.append(f"WINNER: {'draw' if self.winner == -1 else 'P' + str(self.winner)}")
        return "\n".join(lines)
