# Forbidden Stars Combat Sub-Game — Reference Implementation Spec

Extracted 2026-07-07 from the [ForbiddenStarsFight](https://github.com/riik-db/ForbiddenStarsFight)
simulator (`new_ver/`), scoped to Space Marines and Orks. This is the porting
target for Phase 2. **Note:** values are the fan-rebalanced ("SoW") set, not
base-game printed components — we port these because the reference balance
figures (our calibration target) were produced with them.

## 1. Dice model

`utils.rollDice()`: **P(Bolter) = 3/6, P(Shield) = 2/6, P(Morale) = 1/6** —
a d6 with 3 bolter, 2 shield, 1 morale face. One icon per face, no blanks.

- **Hard dice-pool cap of 8**: initial roll is `min(8, sum of unit dice)`;
  "roll extra dice" effects roll `min(n, 8 - pool)` — overflow silently lost.
- Dice are counts `{Bolter, Shield, Morale}`, rolled **once at battle start**;
  the same pool persists (mutated by card effects) across all 3 rounds. No
  whole-pool re-roll between rounds.

## 2. Unit rosters

All ground units. `unit_count` = supply cap for army sampling; `dice` = dice
contributed to the shared pool.

### Space Marines

| Unit | tier | unit_count | dice | hp | morale |
|---|---|---|---|---|---|
| Scouts | 0 | 6 | 1 | 2 | 2 |
| Space Marines | 1 | 6 | 2 | 3 | 3 |
| Land Raiders | 2 | 6 | 3 | 4 | 3 |
| Warlord Titans | 3 | 3 | 3 | 5 | 4 |

### Orks

| Unit | tier | unit_count | dice | hp | morale |
|---|---|---|---|---|---|
| Ork Boyz | 0 | 9 | 2 | 2 | 1 |
| Nobz | 1 | 6 | 2 | 4 | 2 |
| Battlewagons | 2 | 3 | 3 | 5 | 2 |
| Gargants | 3 | 3 | 3 | 6 | 3 |

## 3. Combat decks

14 cards per faction, ids 0–13: `start` = 0–4, `tier_0` upgrades = 5–8,
`tier_2` = 9–11, `tier_3` = 12–13. Each card: printed B/S/M icons, a
**general** ability (always fires) and a **unit** ability (fires only if the
named unit type is alive and unrouted).

### Space Marines

| id | Name | B/S/M | General ability | Unit ability (requires) |
|---|---|---|---|---|
| 0 | Reconnaissance | 0/1/0 | Gain 2 temp tokens, all-Bolter or all-Shield (50/50) | — |
| 1 | Faith in the Emperor | 0/0/1 | Roll +1 die (cap 8) | (Scouts or Space Marines) If pool at 8, or (routed unit exists and 50%): rally one routed unit (highest tier first); else add 1 Morale die |
| 2 | Ambush | 1/0/0 | Gain 2 temp Bolter | (Scouts) This round, enemy units routed-not-killed are destroyed unless enemy spends 1 Morale die |
| 3 | Fury of the Ultramar | 1/0/0 | Force reroll of 1 enemy Shield die; 50% also reroll 1 own Shield die | (Space Marines) At icon-summing: enemy loses 2 temp Shield (50%, if any) else 1 Shield die |
| 4 | Blessed Power armour | 0/1/0 | Gain 2 temp Shield | (Space Marines) Convert 2 random own dice to Shield |
| 5 | Hold the line | 0/1/1 | Gain 2 temp Shield; as defender also rally one routed unit | (Space Marines) If pool < 8, add 1 die: Shield or Morale (50/50) |
| 6 | Glory and death | 1/0/1 | Gain 2 temp Bolter; as attacker also rally one routed unit | (Space Marines) Remove 1 random enemy Shield/Morale die |
| 7 | Veteran scouts | 1/1/1 | Gain temp tokens = own Morale-die count, split randomly Bolter/Shield | — |
| 8 | Drop Pod assault | 1/1/0 | Roll +1 die (cap 8) | (Space Marines, ≥1 Morale die) Spend 1 Morale die; spawn free Scouts or Space Marines (50/50, dice 0) |
| 9 | Show no fear (T2) | 0/2/1 | This round own units cannot rout: damage smaller than target hp is discarded | (Space Marines) Rally all routed units; if any rallied and ≥1 Morale die, spend 1 Morale die |
| 10 | Break the line (T2) | 1/2/0 | Convert up to min(3, own Morale dice) Morale→Bolter/Shield (round-biased amount) | (Land Raiders; attacker round ≥2, defender any round) Enemy discards 1 random face-up card |
| 11 | Armoured advance (T2) | 2/1/0 | Roll +1 die (cap 8) | (Land Raiders) This round the damage-assignment loop runs twice (reference applies to BOTH sides — see bugs) |
| 12 | Emperor's glory (T3) | 0/2/2 | Roll +2 dice (cap 8) | (Warlord Titans) Rally all routed; convert Bolter/Shield→Morale (round-biased amount) |
| 13 | Emperor's might (T3) | 3/0/0 | Roll +2 dice (cap 8) | (Warlord Titans) Spend N own Bolter dice (random N); gain 2N temp Bolter |

### Orks

| id | Name | B/S/M | General ability | Unit ability (requires) |
|---|---|---|---|---|
| 0 | Slugga Boyz | 1/1/0 | Reroll all own AND all enemy Morale dice | (Ork Boyz, routed unit exists) Rally one routed unit |
| 1 | Shoota Boyz | 2/0/0 | Reroll all own Shield dice | (Ork Boyz) Enemy rerolls N Shield dice, N = # alive unrouted Ork Boyz |
| 2 | Ard Boyz | 0/2/0 | Reroll all own Bolter dice | (Ork Boyz) Enemy rerolls N Bolter dice, N = # alive unrouted Ork Boyz |
| 3 | Gretchin | 0/0/0 | Gain 1 temp Bolter + 1 temp Shield; enemy rerolls 1 random die | — |
| 4 | Mek Boyz | 0/0/1 | Roll +1 die (cap 8) | (Ork Boyz) Steal top undrawn enemy card: add its printed icons as temp tokens, remove it from enemy deck |
| 5 | Biker Nobz | 2/1/0 | Enemy rerolls all Bolter dice | (Nobz) Gain 1 temp Bolter |
| 6 | Sea of green | 1/1/0 | Spawn free Ork Boyz (dice 0). If own unrouted+1 > enemy unrouted: enemy loses 1 Morale die (50%) or routs one unit (lowest tier) | — |
| 7 | Waaagh!!!! | 0/0/3 | Rally one routed unit | (Ork Boyz) Gain temp Bolter = # alive unrouted Ork Boyz |
| 8 | Mega Nobz | 1/2/0 | Enemy rerolls all Shield dice | (Nobz) Gain 1 temp Shield |
| 9 | Rokkit Wagon (T2) | 3/0/0 | — | (Battlewagons) Gain 3 temp Bolter |
| 10 | Weirdboyz (T2) | 1/1/1 | Reroll ALL own and ALL enemy dice | (Ork Boyz) Continuous: whenever opponent's abilities generate temp tokens, copy them |
| 11 | Party Wagon (T2) | 1/2/0 | Spawn free Ork Boyz (dice 0) | (Battlewagons, more unrouted units than enemy) Gain 2 temp Bolter + 2 temp Shield |
| 12 | Snapper Gargant (T3) | 4/1/0 | — | (Gargants; attacker round ≥2, defender any round) Enemy discards 1 random face-up card |
| 13 | Smasher Gargant (T3) | 2/3/0 | — | (Gargants) Target enemy's highest-tier unrouted unit; 50% (if tier > 0) destroy it; else remove min(tier, enemy dice) random enemy dice |

### Deck construction & hand

- Deck = **5 distinct card ids × 2 copies = 10 cards**. Upgrades displace
  start cards one-for-one.
- Upgrade counts by stage: **early** (rounds 1–3): tier-0 1 (30%)/2 (50%)/3 (20%);
  **mid** (4–6): exactly 2 tier-0, tier-2 1 (50%)/2 (50%);
  **late** (7–8): 1 tier-0, 2 tier-2, tier-3 1 (80%)/2 (20%).
- Shuffle; **draw 5 as hand**, 5 undrawn (only reachable via Mek Boyz steal).
- Reference plays cards in a fixed weighted order (no per-round choice under
  random play); in our env card choice is the agent's decision.
- 3 rounds → 3 of 5 hand cards get played. Duplicate copies allowed.

## 4. Combat resolution sequence (per battle)

**Setup:** attacker army size 1–5 (probs [.1,.2,.4,.2,.1]); tier draw per
stage — early [0.7,0.25,0.05,0] (t2 cap 1, no t3), mid [0.4,0.4,0.2,0] (t2 cap
2), late [0.1,0.35,0.45,0.1] (t0 cap 2, t2 cap 2, t3 cap 1); supply decrement.
Defender size = attacker's dice-bearing count + {−1: 30%, 0: 60%, +1: 10%},
clamped 1–5. **Reinforcements** both sides: min(choice([0,1,2,3], w=[.3,.4,.2,.1]),
army size) copies of the faction's tier-0 unit with dice=0. Roll pool =
min(8, Σ unit dice). Build decks and hands.

**Per round (max 3):**
1. Attacker's card resolves (general then unit ability), then defender's card
   resolves on the post-attacker state. Resolution is sequential attacker-first
   (reference has no information leak since play order is fixed at setup; in
   our env both players commit their card before resolution to preserve hidden
   information, then resolution applies attacker-first).
2. Face-up card discard effects (Break the Line, Snapper Gargant) remove a
   random face-up enemy card.
3. **Icon summation** per side: dice pool + this round's temp tokens + printed
   icons of ALL face-up cards (cards keep scoring every round after being
   played, unless discarded). Temp tokens then expire.
4. **Damage, simultaneous**: `unblocked = enemy Bolter − own Shield`. While
   unblocked > 0: a target among own alive units is chosen (reference: a
   stochastic tier heuristic; our env: the owning agent chooses). Target is
   **routed** (killed if unblocked ≥ hp, or if it was already routed);
   `unblocked -= target.hp` always — hp is a per-hit absorption amount, there
   is no persistent wound tracking. `cannot_rout`: damage < target hp is
   discarded. Ambush: rout upgrades to kill unless 1 Morale die spent.
5. Elimination check → wipe = loss (simultaneous wipe = tie).
6. After round 3: `side_morale = Σ morale of unrouted units + round-3 Morale
   icons`. Attacker wins iff strictly greater — **defender wins ties**.

Routed units: give no morale, can't satisfy unit-ability requirements, are
only damage-targeted when no unrouted unit remains, and can be rallied.
Rally prefers highest-tier; forced routs/destroys prefer lowest-tier.

## 5. Calibration targets (reference documentation)

- **SM vs Orks: within 4 percentage points of 50–50** under random play
  (3M simulations per matchup; 1M per stage).
- Overall average attacker win rate across matchups: **51.7%**.
- "Well balanced" band: attacker win 43–57%. "Early Orks vs SM" sits close to
  the upper threshold.
- Reference assumptions: ground battles only, bastions ignored, all decisions
  random except lowest-tier-rout preference and round-biased conversions.

## 6. Known reference-code bugs (we fix these in our port)

1. SM card 8's `name` field says 'Veteran scouts'; it is Drop Pod assault.
2. Weirdboyz continuous-effect removal keyed on card id 8 instead of 10.
3. `additional_damage` (Armoured advance) doubles damage application for BOTH
   sides, not just the SM player's target.
4. `check_any_unit`'s `unit_name_list is not {}` check is always True.

Fixing 2–3 may cause small deviations from the reference balance figures.
