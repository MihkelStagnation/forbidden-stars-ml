"""Phase 4 rungs A+B: a CAMPAIGN environment composing the combat env.

A campaign is a war of N_BATTLES battles at fixed stages (early -> mid ->
late) between two persistent forces. Same Gym-style interface as CombatEnv
(`reset()/step()`, one acting player at a time), so the model, agent and
trainer carry over unchanged apart from dimensions.

Structure per battle:
  1. BUILD phase: both players receive INCOME build points, then alternate
     purchases (public): buy a unit (cost tier+1, gated by stage, supply and
     ROSTER_CAP) or a deck-upgrade card (cost by tier group; each id once,
     added as x2 copies displacing a start card — rung A's "hands drawn from
     decks": each battle draws a fresh 5-card hand from the player's CURRENT
     10-card deck). Passing is binding; the phase ends when both have passed.
     A player with an empty roster cannot pass while a unit is affordable.
  2. BATTLE: the combat env runs as a subroutine with the current rosters and
     decks. The campaign attacker alternates (P0, P1, P0); combat seat 0 is
     always the combat attacker, so seats are remapped per battle.
  3. Aftermath: killed units leave the roster permanently; routed units rally.
     Battle winner scores 1 campaign point (combat draws score nothing).

Terminal after the last battle: most points wins (+1/-1), equal points is a
draw (0). There is NO per-battle reward shaping — credit must flow through
the value function across ~40-70 decisions, which is what GAE is for.

Simplifications (deliberate, documented):
  * Purchases are public and sequential (real FS builds are board-visible).
  * Battle-spawned units (Drop Pods, Sea of Green) are battle-only.
  * The whole roster fights every battle (no army-selection decision yet).
  * Which start cards remain in a part-upgraded deck is random per battle.

ACTIONS (single discrete index; combat actions masked during build and vice
versa):
    0 .. 16                combat actions (see combat_env)
    17 + t                 buy own-faction unit of tier t (0..3)
    21 + i                 buy deck upgrade card id 5+i (i = 0..8)
    30                     pass (build phase only)
"""

import numpy as np

from . import game_data as gd
from .combat_env import (
    CombatEnv, ACTION_DIM as COMBAT_ACTION_DIM, MAX_HAND, MAX_UNITS,
    SCALAR_FEATS as COMBAT_SCALAR_FEATS, UNIT_FEATS, CARD_FEATS,
)
from .utils import ICONS

N_BATTLES = 3
STAGE_BY_BATTLE = ("early", "mid", "late")
ROSTER_CAP = 8
INCOME = 5
START_ROSTER_TIERS = (0, 0, 1)

UNIT_COST = {0: 1, 1: 2, 2: 3, 3: 4}          # cost = tier + 1
CARD_COST = {"t0": 1, "t2": 2, "t3": 3}
UPGRADE_IDS = tuple(range(5, 14))
MAX_OWNED_UPGRADES = 5                         # a fully-upgraded 10-card deck
MAX_UNIT_TIER_BY_BATTLE = (1, 2, 3)
CARD_GROUPS_BY_BATTLE = (("t0",), ("t0", "t2"), ("t0", "t2", "t3"))

N_BUILD_ACTIONS = 4 + len(UPGRADE_IDS) + 1
ACTION_DIM = COMBAT_ACTION_DIM + N_BUILD_ACTIONS
BUY_UNIT_BASE = COMBAT_ACTION_DIM
BUY_CARD_BASE = BUY_UNIT_BASE + 4
PASS_ACTION = BUY_CARD_BASE + len(UPGRADE_IDS)

# campaign scalar block appended to the combat scalars:
# [in_build, battle one-hot x3, score self, score enemy, points self,
#  points enemy, owned-upgrade bits self x9, enemy x9, roster self, enemy]
CAMPAIGN_SCALARS = 1 + N_BATTLES + 2 + 2 + 2 * len(UPGRADE_IDS) + 2
SCALAR_FEATS = COMBAT_SCALAR_FEATS + CAMPAIGN_SCALARS

PHASE_BUILD = 2

_GROUP_OF = {cid: ("t0" if cid <= 8 else "t2" if cid <= 11 else "t3")
             for cid in UPGRADE_IDS}


class GameEnv:
    def __init__(self, factions=("SM", "Orks"), seed=None):
        self.factions = factions
        self.rng = np.random.default_rng(seed)
        self.combat = CombatEnv(factions=factions)
        self.combat.rng = self.rng  # shared stream keeps campaigns seed-repeatable

    # ------------------------------------------------------------------ reset
    def reset(self, compositions=None, stage=None, factions=None):
        # compositions/stage accepted for interface parity; unused here.
        if factions is not None:
            self.factions = tuple(factions)
        rosters = {}
        for p in (0, 1):
            roster_names = [gd.UNIT_ROSTERS[self.factions[p]][t]["name"]
                            for t in START_ROSTER_TIERS]
            rosters[p] = roster_names
        self.roster = rosters
        self.points = {0: INCOME, 1: INCOME}
        self.owned = {0: set(), 1: set()}
        self.score = {0: 0, 1: 0}
        self.battle_idx = 0
        self.in_battle = False
        self.phase = PHASE_BUILD
        self.current_player = 0
        self._passed = {0: False, 1: False}
        self.done = False
        self.winner = None
        return self.observe(), self._info()

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        actor = self.current_player
        if self.in_battle:
            self._step_battle(action)
        else:
            self._step_build(action)

        reward = 0.0
        if self.done and self.winner != -1:
            reward = 1.0 if self.winner == actor else -1.0
        return self.observe(), reward, self.done, self._info(extra_actor=actor)

    # ------------------------------------------------------------------ build
    def _step_build(self, action):
        p = self.current_player
        if action == PASS_ACTION:
            self._passed[p] = True
        elif BUY_UNIT_BASE <= action < BUY_CARD_BASE:
            tier = action - BUY_UNIT_BASE
            unit = gd.UNIT_ROSTERS[self.factions[p]][tier]
            assert self._can_buy_unit(p, tier), "illegal unit purchase"
            self.points[p] -= UNIT_COST[tier]
            self.roster[p].append(unit["name"])
        elif BUY_CARD_BASE <= action < PASS_ACTION:
            cid = UPGRADE_IDS[action - BUY_CARD_BASE]
            assert self._can_buy_card(p, cid), "illegal card purchase"
            self.points[p] -= CARD_COST[_GROUP_OF[cid]]
            self.owned[p].add(cid)
        else:
            raise AssertionError("combat action during build phase")

        if self._passed[0] and self._passed[1]:
            self._start_battle()
            return
        other = 1 - p
        self.current_player = other if not self._passed[other] else p

    def _can_buy_unit(self, p, tier):
        unit = gd.UNIT_ROSTERS[self.factions[p]][tier]
        return (tier <= MAX_UNIT_TIER_BY_BATTLE[self.battle_idx]
                and self.points[p] >= UNIT_COST[tier]
                and len(self.roster[p]) < ROSTER_CAP
                and self.roster[p].count(unit["name"]) < unit["unit_count"])

    def _can_buy_card(self, p, cid):
        group = _GROUP_OF[cid]
        return (group in CARD_GROUPS_BY_BATTLE[self.battle_idx]
                and cid not in self.owned[p]
                and len(self.owned[p]) < MAX_OWNED_UPGRADES
                and self.points[p] >= CARD_COST[group])

    # ----------------------------------------------------------------- battle
    def combat_seat(self, campaign_player):
        """Combat seat (0 = attacker) of a campaign player this battle."""
        attacker = self.battle_idx % 2
        return 0 if campaign_player == attacker else 1

    def _campaign_player(self, combat_seat):
        attacker = self.battle_idx % 2
        return attacker if combat_seat == 0 else 1 - attacker

    def _start_battle(self):
        att = self.battle_idx % 2
        def_ = 1 - att
        stage = STAGE_BY_BATTLE[self.battle_idx]
        self._battle_comps = {0: list(self.roster[att]), 1: list(self.roster[def_])}
        self.combat.reset(
            compositions=[self._battle_comps[0], self._battle_comps[1]],
            stage=stage,
            factions=(self.factions[att], self.factions[def_]),
        )
        # Decks come from each player's owned upgrades, not stage sampling.
        for seat, p in ((0, att), (1, def_)):
            hand, undrawn = gd.build_deck_from(
                self.factions[p], self.owned[p], self.rng)
            self.combat.hands[seat], self.combat.undrawn[seat] = hand, undrawn
        self.in_battle = True
        self.phase = self.combat.phase
        self.current_player = self._campaign_player(self.combat.current_player)

    def _step_battle(self, action):
        assert action < COMBAT_ACTION_DIM, "build action during battle"
        _, _, cdone, _ = self.combat.step(action)
        if not cdone:
            self.phase = self.combat.phase
            self.current_player = self._campaign_player(self.combat.current_player)
            return
        # Aftermath: score and permanent casualties (spawned units excluded).
        if self.combat.winner in (0, 1):
            self.score[self._campaign_player(self.combat.winner)] += 1
        for seat in (0, 1):
            p = self._campaign_player(seat)
            comp = self._battle_comps[seat]
            self.roster[p] = [name for i, name in enumerate(comp)
                              if not self.combat.units[seat][i]["killed"]]
        self.in_battle = False
        self.battle_idx += 1
        if self.battle_idx >= N_BATTLES:
            self.done = True
            if self.score[0] > self.score[1]:
                self.winner = 0
            elif self.score[1] > self.score[0]:
                self.winner = 1
            else:
                self.winner = -1
            return
        self.phase = PHASE_BUILD
        self.current_player = 0
        self._passed = {0: False, 1: False}
        for p in (0, 1):
            self.points[p] += INCOME

    # ------------------------------------------------------------- observation
    def action_mask(self):
        mask = np.zeros(ACTION_DIM, dtype=bool)
        if self.done:
            return mask
        if self.in_battle:
            mask[:COMBAT_ACTION_DIM] = self.combat.action_mask()
            return mask
        p = self.current_player
        for tier in range(4):
            mask[BUY_UNIT_BASE + tier] = self._can_buy_unit(p, tier)
        for i, cid in enumerate(UPGRADE_IDS):
            mask[BUY_CARD_BASE + i] = self._can_buy_card(p, cid)
        # An army-less player must rebuild before passing if they can afford to.
        must_buy = (not self.roster[p]
                    and any(mask[BUY_UNIT_BASE + t] for t in range(4)))
        mask[PASS_ACTION] = not must_buy
        return mask

    def _encode_roster(self, owner, viewer):
        arr = np.zeros((MAX_UNITS, UNIT_FEATS), dtype=np.float32)
        for j, name in enumerate(self.roster[owner][:MAX_UNITS]):
            u = gd.make_unit(self.factions[owner], name)
            arr[j] = [
                u["tier"] / gd.MAX_TIER,
                u["hp"] / gd.MAX_HP,
                u["morale"] / 4.0,
                u["dice"] / 3.0,
                0.0, 0.0,
                1.0 if owner == viewer else 0.0,
                1.0 if u["faction"] == "SM" else 0.0,
                1.0,
            ]
        return arr

    def _campaign_block(self, p):
        opp = 1 - p
        block = [1.0 if not self.in_battle else 0.0]
        block += [float(self.battle_idx == b) for b in range(N_BATTLES)]
        block += [self.score[p] / N_BATTLES, self.score[opp] / N_BATTLES,
                  self.points[p] / 15.0, self.points[opp] / 15.0]
        block += [float(cid in self.owned[p]) for cid in UPGRADE_IDS]
        block += [float(cid in self.owned[opp]) for cid in UPGRADE_IDS]
        block += [len(self.roster[p]) / ROSTER_CAP,
                  len(self.roster[opp]) / ROSTER_CAP]
        return np.array(block, dtype=np.float32)

    def observe(self):
        p = self.current_player
        if self.in_battle:
            obs = self.combat.observe()
            obs["scalars"] = np.concatenate(
                [obs["scalars"], self._campaign_block(p)])
            mask = np.zeros(ACTION_DIM, dtype=bool)
            mask[:COMBAT_ACTION_DIM] = obs["action_mask"]
            obs["action_mask"] = mask
            return obs
        # Build phase: pseudo-combat scalars carry only the static context of
        # the upcoming battle (faction, role, stage).
        opp = 1 - p
        combat_scalars = np.zeros(COMBAT_SCALAR_FEATS, dtype=np.float32)
        combat_scalars[4] = 1.0 if self.combat_seat(p) == 0 else 0.0
        combat_scalars[5] = 1.0 if self.factions[p] == "SM" else 0.0
        # (clamped: after the final battle there is no upcoming stage)
        stage = STAGE_BY_BATTLE[min(self.battle_idx, N_BATTLES - 1)]
        for i, s in enumerate(gd.STAGES):
            combat_scalars[6 + i] = float(stage == s)
        return {
            "units_self": self._encode_roster(p, p),
            "units_enemy": self._encode_roster(opp, p),
            "hand": np.zeros((MAX_HAND, CARD_FEATS), dtype=np.float32),
            "scalars": np.concatenate([combat_scalars, self._campaign_block(p)]),
            "action_mask": self.action_mask(),
            "phase": PHASE_BUILD,
        }

    def _info(self, extra_actor=None):
        info = {
            "current_player": self.current_player,
            "battle": self.battle_idx,
            "phase": self.phase if self.in_battle else PHASE_BUILD,
            "score": dict(self.score),
            "winner": self.winner,
        }
        if extra_actor is not None:
            info["acting_player"] = extra_actor
        return info

    # ------------------------------------------------------------- text render
    def render(self):
        lines = [f"== battle {self.battle_idx + 1}/{N_BATTLES}"
                 f" | {'BATTLE' if self.in_battle else 'BUILD'}"
                 f" | score {self.score[0]}-{self.score[1]}"
                 f" | to act: P{self.current_player} =="]
        for p in (0, 1):
            lines.append(
                f"P{p} [{self.factions[p]}] pts {self.points[p]}"
                f" | upgrades {sorted(self.owned[p])}"
                f" | roster: {', '.join(self.roster[p]) or '(none)'}")
        if self.in_battle:
            lines.append(self.combat.render())
        if self.done:
            lines.append(f"CAMPAIGN WINNER: "
                         f"{'draw' if self.winner == -1 else 'P' + str(self.winner)}")
        return "\n".join(lines)
