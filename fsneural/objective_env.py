r"""Phase 4 rung E: OBJECTIVES & VICTORY on top of the order-stack game.

Rules delta vs OrderEnv (everything else — planning/operations, LIFO
stacks, battles, economy — is inherited unchanged):

  * Each player owns N_OBJECTIVES = 4 objective tokens at fixed, symmetric
    setup spots: TWO on the enemy home, one on the center, one on a flank
    (mirrored under the map automorphism H0<->H1, A<->B). You must venture
    out and HOLD those planets to score.
  * CLAIM: at each round end, every planet you control (sole occupancy)
    that still carries >= 1 of your tokens yields exactly ONE of them.
    (Holding the enemy home two round-ends running claims both its tokens.)
  * WIN: claiming your 4th token at a round end wins immediately. If both
    players complete in the same round: higher controlled planet value,
    else a draw. After N_ROUNDS: most tokens claimed, then controlled
    value, then draw.
  * The rung C/D home-capture instant win is RETIRED — objectives are the
    victory driver now (the enemy home still carries two of your tokens,
    so conquest remains a winning path, just not an instant one).

No new actions (claiming is automatic): ACTION_DIM stays 41. The scalar
block grows by OBJ_SCALARS: per-planet remaining-token counts for both
players plus claimed-so-far counters.
"""

import numpy as np

from .order_env import (
    OrderEnv, ACTION_DIM, UNIT_FEATS, SCALAR_FEATS as ORDER_SCALAR_FEATS,
    N_PLANETS, PLANETS, PLANET_VALUE, HOME, N_ROUNDS,
)

N_OBJECTIVES = 4
# setup spots per player (planet indices; duplicates = multiple tokens)
OBJECTIVE_SPOTS = {0: (HOME[1], HOME[1], 2, 1),   # P0 scores at H1, H1, M, A
                   1: (HOME[0], HOME[0], 2, 3)}   # P1 scores at H0, H0, M, B

# appended to the order scalars:
# [my remaining tokens per planet x5 (/2), enemy remaining per planet x5,
#  my claimed /4, enemy claimed /4]
OBJ_SCALARS = 2 * N_PLANETS + 2
SCALAR_FEATS = ORDER_SCALAR_FEATS + OBJ_SCALARS


class ObjectiveEnv(OrderEnv):
    def reset(self, compositions=None, stage=None, factions=None):
        self.objectives = {p: list(OBJECTIVE_SPOTS[p]) for p in (0, 1)}
        self.claimed = {0: 0, 1: 0}
        return super().reset(compositions=compositions, stage=stage,
                             factions=factions)

    # ------------------------------------------------------------- round flow
    def _end_round(self):
        assert all(not s for s in self.stacks.values())
        for p in (0, 1):
            for planet in set(self.objectives[p]):
                if self._controller(planet) == p:
                    self.objectives[p].remove(planet)   # one per planet/round
                    self.claimed[p] += 1
        done_players = [p for p in (0, 1) if self.claimed[p] >= N_OBJECTIVES]
        if done_players or self.round >= N_ROUNDS:
            self.done = True
            if len(done_players) == 1:
                self.winner = done_players[0]
                return
            # both completed, or the war timed out: most tokens, then value
            if self.claimed[0] != self.claimed[1]:
                self.winner = 0 if self.claimed[0] > self.claimed[1] else 1
                return
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

    # ------------------------------------------------------------ observation
    def _order_block(self, p):
        opp = 1 - p
        block = super()._order_block(p)
        obj = [self.objectives[p].count(i) / 2.0 for i in range(N_PLANETS)]
        obj += [self.objectives[opp].count(i) / 2.0 for i in range(N_PLANETS)]
        obj += [self.claimed[p] / N_OBJECTIVES, self.claimed[opp] / N_OBJECTIVES]
        return np.concatenate([block, np.array(obj, dtype=np.float32)])

    def _info(self, extra_actor=None):
        info = super()._info(extra_actor=extra_actor)
        info["claimed"] = dict(self.claimed)
        info["objectives"] = {q: [PLANETS[i] for i in self.objectives[q]]
                              for q in (0, 1)}
        return info

    # ------------------------------------------------------------ text render
    def render(self):
        lines = [super().render()]
        for p in (0, 1):
            spots = ", ".join(PLANETS[i] for i in self.objectives[p]) or "-"
            lines.append(f"P{p} objectives: {self.claimed[p]}/{N_OBJECTIVES}"
                         f" claimed | remaining at: {spots}")
        return "\n".join(lines)
