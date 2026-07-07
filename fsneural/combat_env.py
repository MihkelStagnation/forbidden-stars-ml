"""A step-based, 2-player Forbidden Stars *combat* environment.

Gym-style env for self-play RL. One acting player at a time
(`info['current_player']`); in self-play the same policy controls both sides.

--------------------------------------------------------------------------
COMBAT MODEL — ported from the ForbiddenStarsFight simulator
(docs/reference_combat_model.md has the full spec and provenance)
--------------------------------------------------------------------------
* Setup: a battle stage (early/mid/late) is sampled; armies (1-5 units plus
  dice-less tier-0 reinforcements) and 10-card decks (5 distinct ids x 2,
  stage-based upgrades) are sampled per the reference priors. Each side draws
  a 5-card hand and rolls its dice pool ONCE: min(8, sum of unit dice). The
  pool persists all battle and is mutated by card effects.
* Each of up to 3 rounds, BOTH players commit one hand card hidden, then the
  cards resolve sequentially attacker-first (general ability, then unit
  ability if the required unit type is alive and unrouted).
* Icons per side = dice pool + this round's temp tokens + printed icons of
  ALL face-up cards (played cards keep scoring every later round unless
  discarded by an effect). Temp tokens expire each round.
* damage = max(0, enemy Bolter - own Shield). The damaged player ASSIGNS it
  to their own units one target at a time (the agent's decision): a target
  absorbs its full hp from the pending damage — killed outright if pending
  >= hp, merely ROUTED on a smaller final hit. Routed units give no morale,
  can be targeted only when no unrouted unit remains (then they are killed),
  and can be rallied by cards.
* Terminal on a wipe or after 3 rounds. Both wiped = draw; otherwise the
  survivor wins; if both survive, compare morale = sum of unrouted units'
  morale + final-round Morale icons — the attacker (P0) needs strictly more,
  DEFENDER WINS TIES.

Reward is zero-sum, revealed at the terminal step from the perspective of the
player who just acted (+1 / -1 / 0). `info['winner']` + `info['acting_player']`
let the trainer sign both sides correctly.

ACTIONS (single discrete index, masked by phase):
    0 .. MAX_HAND-1                  -> play hand card i
    MAX_HAND .. MAX_HAND+MAX_UNITS-1 -> assign pending damage to own unit j

Known observation simplification: opposing face-up cards are encoded only as
printed-icon sums in the scalars (not as an entity set), so the agent can't
see WHICH enemy abilities are face-up — only their icon contribution.
"""

import numpy as np

from . import game_data as gd
from .card_abilities import apply_card
from .utils import ICONS, DICE_CAP, roll_dice, pool_size

MAX_UNITS = 12
MAX_HAND = 5
ROUNDS = 3
ACTION_DIM = MAX_HAND + MAX_UNITS

UNIT_FEATS = 9    # [tier, hp, morale, dice, routed, killed, is_self, is_sm, present]
CARD_FEATS = 26   # [B, S, M, att, def, both, 4 group one-hot, 14 id one-hot, is_sm, present]
SCALAR_FEATS = 40  # (see observe(); includes an is-SM faction flag — factions
                   # can be assigned to either role via reset(factions=...))

PHASE_PLAY = 0
PHASE_ASSIGN = 1

_GROUPS = ("start", "t0", "t2", "t3")


class CombatEnv:
    def __init__(self, factions=("SM", "Orks"), seed=None):
        self.factions = factions
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ reset
    def reset(self, compositions=None, stage=None, factions=None):
        if factions is not None:
            self.factions = tuple(factions)
        self.stage = stage or str(self.rng.choice(gd.STAGES))

        if compositions is not None:
            self.units = {
                p: [gd.make_unit(self.factions[p], name) for name in compositions[p]]
                for p in (0, 1)
            }
        else:
            a, d = gd.sample_armies(self.factions[0], self.factions[1],
                                    self.stage, self.rng)
            self.units = {0: a[:MAX_UNITS], 1: d[:MAX_UNITS]}

        self.dice, self.hands, self.undrawn = {}, {}, {}
        for p in (0, 1):
            n = min(DICE_CAP, sum(u["dice"] for u in self.units[p]))
            self.dice[p] = roll_dice(n, self.rng)
            self.hands[p], self.undrawn[p] = gd.build_deck(
                self.factions[p], self.stage, self.rng)

        self.faceup = {0: [], 1: []}
        self.temp = {p: {ic: 0 for ic in ICONS} for p in (0, 1)}
        self.flags = {0: set(), 1: set()}
        self.continuous = {0: set(), 1: set()}
        self._last_morale_icons = {0: 0, 1: 0}

        self.round = 1
        self.phase = PHASE_PLAY
        self.current_player = 0            # P0 is the attacker
        self._played = {}                  # player -> committed card this round
        self._pending = {0: 0, 1: 0}       # unassigned damage per player
        self.done = False
        self.winner = None                 # 0, 1, or -1 (draw)
        return self.observe(), self._info()

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")

        actor = self.current_player
        if self.phase == PHASE_PLAY:
            self._apply_play(action)
        elif self.phase == PHASE_ASSIGN:
            self._apply_assign(action)
        else:  # pragma: no cover
            raise AssertionError("bad phase")

        reward = 0.0
        if self.done and self.winner != -1:
            reward = 1.0 if self.winner == actor else -1.0
        return self.observe(), reward, self.done, self._info(extra_actor=actor)

    # ------------------------------------------------------- phase transitions
    def _apply_play(self, action):
        assert action < MAX_HAND, "play-phase action must index the hand"
        card = self.hands[self.current_player][action]
        assert card is not None, "tried to play an empty hand slot"
        self._played[self.current_player] = card
        self.hands[self.current_player][action] = None

        if len(self._played) < 2:
            self.current_player = 1 - self.current_player
        else:
            self._resolve_round()

    def _resolve_round(self):
        for p in (0, 1):
            self.temp[p] = {ic: 0 for ic in ICONS}
            self.flags[p] = set()

        # Cards resolve sequentially, attacker first (both were committed
        # hidden, so no information leaks into the play decision).
        for p in (0, 1):
            card = self._played[p]
            self.faceup[p].append(card)
            apply_card(self, p, card, self.round)

        for p in (0, 1):
            if "fury_strip" in self.flags[p]:
                self._fury_strip(1 - p)

        icons = {p: self._icons(p) for p in (0, 1)}
        self._last_morale_icons = {p: icons[p]["Morale"] for p in (0, 1)}
        for p in (0, 1):
            dmg = max(0, icons[1 - p]["Bolter"] - icons[p]["Shield"])
            if "double_damage" in self.flags[1 - p]:
                dmg *= 2
            self._pending[p] = dmg

        self._played = {}
        self._begin_assignment_or_advance()

    def _apply_assign(self, action):
        idx = action - MAX_HAND
        assert 0 <= idx < MAX_UNITS, "assign-phase action must index own units"
        p = self.current_player
        unit = self.units[p][idx]
        assert not unit["killed"], "assigned to a dead unit"

        hp, pend = unit["hp"], self._pending[p]
        if "cannot_rout" in self.flags[p]:
            if pend < hp:
                self._pending[p] = 0      # too small to kill -> discarded
            else:
                unit["routed"] = unit["killed"] = True
                self._pending[p] = pend - hp
        elif unit["routed"]:
            unit["killed"] = True
            self._pending[p] = max(0, pend - hp)
        elif pend >= hp:
            unit["routed"] = unit["killed"] = True
            self._pending[p] = pend - hp
        else:
            unit["routed"] = True
            self._pending[p] = 0
            if "ambush" in self.flags[1 - p]:  # rout upgrades to kill unless paid off
                if self.dice[p]["Morale"] >= 1 and self.rng.random() < 0.5:
                    self.dice[p]["Morale"] -= 1
                else:
                    unit["killed"] = True

        self._advance_after_assign()

    def _begin_assignment_or_advance(self):
        for p in (0, 1):
            if self._pending[p] > 0 and self.alive(p):
                self.phase = PHASE_ASSIGN
                self.current_player = p
                return
            self._pending[p] = 0
        self._end_round()

    def _advance_after_assign(self):
        p = self.current_player
        if self._pending[p] > 0 and self.alive(p):
            return
        self._pending[p] = 0
        other = 1 - p
        if self._pending[other] > 0 and self.alive(other):
            self.phase = PHASE_ASSIGN
            self.current_player = other
            return
        self._pending[other] = 0
        self._end_round()

    def _end_round(self):
        if not self.alive(0) or not self.alive(1) or self.round >= ROUNDS:
            self._finish()
            return
        self.round += 1
        self.phase = PHASE_PLAY
        self.current_player = 0

    def _finish(self):
        self.done = True
        alive0, alive1 = bool(self.alive(0)), bool(self.alive(1))
        if not alive0 and not alive1:
            self.winner = -1
        elif not alive1:
            self.winner = 0
        elif not alive0:
            self.winner = 1
        else:  # both survive 3 rounds -> morale; defender wins ties
            self.winner = 0 if self.morale_score(0) > self.morale_score(1) else 1

    # ------------------------------------------------------ ability primitives
    def alive(self, p):
        return [u for u in self.units[p] if not u["killed"]]

    def unrouted(self, p):
        return [u for u in self.units[p] if not u["killed"] and not u["routed"]]

    def routed_units(self, p):
        return [u for u in self.units[p] if not u["killed"] and u["routed"]]

    def gain_temp(self, p, bolter=0, shield=0, morale=0, _copied=False):
        self.temp[p]["Bolter"] += bolter
        self.temp[p]["Shield"] += shield
        self.temp[p]["Morale"] += morale
        if (not _copied and (bolter or shield or morale)
                and "copy_temp" in self.continuous[1 - p]):
            self.gain_temp(1 - p, bolter, shield, morale, _copied=True)

    def rally(self, p, all_units=False):
        routed = sorted(self.routed_units(p), key=lambda u: -u["tier"])
        for u in routed[: len(routed) if all_units else 1]:
            u["routed"] = False

    def rout_lowest(self, p):
        pool = self.unrouted(p)
        if pool:
            min(pool, key=lambda u: u["tier"])["routed"] = True

    def spawn(self, p, name):
        if len(self.units[p]) < MAX_UNITS:
            self.units[p].append(gd.make_unit(self.factions[p], name, dice=0))

    def discard_faceup(self, victim):
        if not self.faceup[victim]:
            return
        card = self.faceup[victim].pop(int(self.rng.integers(len(self.faceup[victim]))))
        if (self.factions[victim] == "Orks" and card["id"] == 10
                and not any(c["id"] == 10 for c in self.faceup[victim])):
            self.continuous[victim].discard("copy_temp")

    def _fury_strip(self, victim):
        if self.temp[victim]["Shield"] > 0 and self.rng.random() < 0.5:
            self.temp[victim]["Shield"] = max(0, self.temp[victim]["Shield"] - 2)
        elif self.dice[victim]["Shield"] > 0:
            self.dice[victim]["Shield"] -= 1

    def _icons(self, p):
        out = {ic: self.dice[p][ic] + self.temp[p][ic] for ic in ICONS}
        for card in self.faceup[p]:
            for ic in ICONS:
                out[ic] += card[ic]
        return out

    def morale_score(self, p):
        return (sum(u["morale"] for u in self.unrouted(p))
                + self._last_morale_icons[p])

    # ------------------------------------------------------------- observation
    def action_mask(self):
        mask = np.zeros(ACTION_DIM, dtype=bool)
        if self.done:
            return mask
        p = self.current_player
        if self.phase == PHASE_PLAY:
            for i, card in enumerate(self.hands[p][:MAX_HAND]):
                if card is not None:
                    mask[i] = True
        else:  # ASSIGN: unrouted targets first; routed only when none unrouted
            targets = self.unrouted(p) or self.routed_units(p)
            for j, u in enumerate(self.units[p][:MAX_UNITS]):
                if any(u is t for t in targets):
                    mask[MAX_HAND + j] = True
        return mask

    def _encode_units(self, owner, viewer):
        arr = np.zeros((MAX_UNITS, UNIT_FEATS), dtype=np.float32)
        for j, u in enumerate(self.units[owner][:MAX_UNITS]):
            arr[j] = [
                u["tier"] / gd.MAX_TIER,
                u["hp"] / gd.MAX_HP,
                u["morale"] / 4.0,
                u["dice"] / 3.0,
                float(u["routed"]),
                float(u["killed"]),
                1.0 if owner == viewer else 0.0,
                1.0 if u["faction"] == "SM" else 0.0,
                1.0,
            ]
        return arr

    def _encode_hand(self, viewer):
        arr = np.zeros((MAX_HAND, CARD_FEATS), dtype=np.float32)
        for i, c in enumerate(self.hands[viewer][:MAX_HAND]):
            if c is None:
                continue
            feats = [
                c["Bolter"] / 4.0, c["Shield"] / 4.0, c["Morale"] / 4.0,
                float(c["card_type"] == "att"),
                float(c["card_type"] == "def"),
                float(c["card_type"] == "both"),
            ]
            feats += [float(c["group"] == g) for g in _GROUPS]
            id_onehot = [0.0] * gd.N_CARD_IDS
            id_onehot[c["id"]] = 1.0
            feats += id_onehot
            feats += [1.0 if self.factions[viewer] == "SM" else 0.0, 1.0]
            arr[i] = feats
        return arr

    def observe(self):
        p = self.current_player
        opp = 1 - p
        faceup = {q: {ic: sum(c[ic] for c in self.faceup[q]) for ic in ICONS}
                  for q in (p, opp)}
        scalars = np.array(
            [
                self.round / ROUNDS,
                1.0 if self.phase == PHASE_PLAY else 0.0,
                1.0 if self.phase == PHASE_ASSIGN else 0.0,
                self._pending[p] / 10.0,
                1.0 if p == 0 else 0.0,  # is_attacker
                1.0 if self.factions[p] == "SM" else 0.0,
            ]
            + [float(self.stage == s) for s in gd.STAGES]
            + [self.dice[p][ic] / DICE_CAP for ic in ICONS]
            + [self.dice[opp][ic] / DICE_CAP for ic in ICONS]
            + [self.temp[p][ic] / 6.0 for ic in ICONS]
            + [self.temp[opp][ic] / 6.0 for ic in ICONS]
            + [faceup[p][ic] / 8.0 for ic in ICONS]
            + [faceup[opp][ic] / 8.0 for ic in ICONS]
            + [
                len(self.alive(p)) / MAX_UNITS,
                len(self.alive(opp)) / MAX_UNITS,
                len(self.unrouted(p)) / MAX_UNITS,
                len(self.unrouted(opp)) / MAX_UNITS,
                self.morale_score(p) / 25.0,
                self.morale_score(opp) / 25.0,
                float("cannot_rout" in self.flags[p]),
                float("ambush" in self.flags[opp]),
                float("double_damage" in self.flags[opp]),
                float("copy_temp" in self.continuous[p]),
                float("copy_temp" in self.continuous[opp]),
                len(self.undrawn[p]) / 5.0,
                len(self.undrawn[opp]) / 5.0,
            ],
            dtype=np.float32,
        )
        assert scalars.shape == (SCALAR_FEATS,)
        return {
            "units_self": self._encode_units(p, p),
            "units_enemy": self._encode_units(opp, p),
            "hand": self._encode_hand(p),
            "scalars": scalars,
            "action_mask": self.action_mask(),
            "phase": self.phase,
        }

    def _info(self, extra_actor=None):
        info = {
            "current_player": self.current_player,
            "round": self.round,
            "phase": self.phase,
            "stage": self.stage,
            "winner": self.winner,
            "morale": {p: self.morale_score(p) for p in (0, 1)},
        }
        if extra_actor is not None:
            info["acting_player"] = extra_actor
        return info

    # ------------------------------------------------------------- text render
    def render(self):
        lines = [f"-- {self.stage} | round {self.round} | phase {self.phase}"
                 f" | to act: P{self.current_player} --"]
        for p in (0, 1):
            us = []
            for u in self.units[p]:
                tag = "DEAD" if u["killed"] else ("rout" if u["routed"] else "ok")
                us.append(f"{u['name']}({tag})")
            d = self.dice[p]
            lines.append(
                f"P{p} [{self.factions[p]}] dice B{d['Bolter']}/S{d['Shield']}"
                f"/M{d['Morale']} | faceup {len(self.faceup[p])}"
                f" | morale~{self.morale_score(p)}: " + ", ".join(us))
        if self.done:
            lines.append(f"WINNER: {'draw' if self.winner == -1 else 'P' + str(self.winner)}")
        return "\n".join(lines)
