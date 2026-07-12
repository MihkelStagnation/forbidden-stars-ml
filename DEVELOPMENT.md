# Development Log & Next Steps

This is the "read this first" document for the project. It records **what was
built, why, the decisions behind it, what's deliberately unfinished, and exactly
what to do next.** If you're returning to this cold, start here.

Last updated: 2026-07-07.

---

## 1. Goal

Train a neural network to play **Forbidden Stars** (the FFG Warhammer 40K board
game) via self-play. The full game is too large for a first attempt, so the
current target is the **combat sub-game only**: a 2-faction, head-to-head fight
with dice and combat cards.

Why combat first: it's the smallest slice that still has every hard property of
the full game — **2-player zero-sum, hidden information (card hands), and
stochasticity (dice)** — so it's a faithful, tractable first rung on the ladder
toward the full game.

---

## 2. How we got here (decision trail)

The project grew out of a discussion that worked through these conclusions:

1. **Started with a tic-tac-toe AlphaZero** (`../tic-tac-toe-neural/`) to prove
   the self-play → train → play pipeline on a trivial game. That uses MCTS +
   a policy/value net (classic AlphaZero).
2. **Asked whether AlphaZero scales to a big wargame.** Conclusion: the *recipe*
   transfers, but a wargame breaks AlphaZero's assumptions — hidden information,
   dice, huge factored action spaces, long games, asymmetric factions, and the
   need for a fast exact simulator.
3. **Picked Forbidden Stars as the concrete target** and noted it hits almost
   every one of those hard properties.
4. **Decided to drop to 2 factions head-to-head.** This is the single biggest
   simplification: it restores clean **2-player zero-sum** (no multiplayer
   kingmaking), halves the simulator work, and lands us exactly in the regime
   that **DeepNash (Stratego)** and **ReBeL (poker)** were built for.
5. **Found an existing reference repo**
   ([ForbiddenStarsFight](https://github.com/riik-db/ForbiddenStarsFight)) — a
   Monte-Carlo *combat* simulator that plays with **random** decisions to study
   balance. It gave us a head start on rules/data shape, but has no learning.
   Our plan: reuse its data shape and replace random play with a learned policy.
6. **Scaffolded this project** as a combat-only, self-play RL testbed.

Key architectural decisions that fell out of this:

- **Model-free self-play, NOT AlphaZero/MCTS.** Hidden card hands mean you can't
  honestly expand the true game tree, so MCTS is the wrong base. A model-free
  policy-gradient (PPO-style) self-play loop sidesteps that and is the most
  forgiving first choice.
- **Network = entity embeddings → self-attention → recurrent memory → policy +
  value heads.** This mirrors the stack large-game agents (AlphaStar, OpenAI
  Five, DeepNash) converged on, shrunk to the combat mini-game. Units and cards
  are variable-size *sets*, hence embeddings + attention; hidden info across a
  multi-step episode motivates the GRU memory (a belief state).

---

## 3. Repository status

Two sibling projects exist under `dev/`:

| Folder | What it is | State |
|--------|-----------|-------|
| `tic-tac-toe-neural/` | Full AlphaZero (MCTS + CNN) for tic-tac-toe | Complete scaffold, **unrun** |
| `forbidden_star_neural/` | This project: combat-only self-play RL | **Phases 0–2 complete** (2026-07-07): real rules ported + calibrated |

**Phase 0 + Phase 1 were executed on 2026-07-07** (Python 3.12.10, torch
2.12.1+cpu, numpy 2.5.1, venv at `.venv/`). Everything ran with **zero
first-run bugs**: 3/3 pytest green, demo sane, 300-iteration training run
completed (entropy declined gradually 1.09 → 0.46, no collapse), checkpoint at
`checkpoints/model.pt`.

**Phase 1 evaluation — on the OLD simplified rules** (2000 games each,
deterministic agent vs uniform random; checkpoint archived as
`checkpoints/model_v1_simplified_rules.pt`, incompatible with current code):

| Matchup | Win | Draw | Loss |
|---------|-----|------|------|
| random P0 (SM) vs random P1 (Orks) — baseline | 20.0% | 10.3% | 69.7% |
| trained agent as P0 (SM) vs random | 36.4% | 13.0% | 50.6% |
| trained agent as P1 (Orks) vs random | 87.3% | 6.4% | 6.3% |

Conclusions: the agent clearly learns (SM 20.0% → 36.4%, Orks 69.7% → 87.3%),
but that simplified model was heavily Ork-favored.

**Phase 2 (same day)** ported the real combat model from the reference
simulator (see §4 and docs/reference_combat_model.md). Calibration under the
ported rules: SM-attacker vs Orks **49.0%** win overall (early 43.3% / mid
51.7% / late 51.9%, 15k games) — the skew was a rules-fidelity gap, now fixed,
and the model matches the reference's balance figures.

---

## 4. File-by-file reference (`forbidden_star_neural`)

```
docs/
  reference_combat_model.md  ★ The ported rules spec, extracted 2026-07-07 from
                             the ForbiddenStarsFight simulator. Read this for
                             exact dice/roster/card/resolution semantics.

fsneural/
  game_data.py    Real SM + Ork rosters (4 tiers each), all 14 combat cards per
                  faction, stage-based deck building (5 distinct ids x 2 = 10,
                  draw 5), army sampling with supply caps + reinforcements.
  card_abilities.py  All 28 card abilities (general + unit slots): temp tokens,
                  rerolls, conversions, extra dice, rally/rout/destroy, spawns,
                  face-up discard, Mek Boyz steal, Weirdboyz copy, damage flags.
  utils.py        Exact combat dice (3/6 Bolter, 2/6 Shield, 1/6 Morale),
                  pool helpers (cap 8, reroll, convert, remove, biased counts).
  combat_env.py   ★ THE CORE. Gym-style 2-player combat env implementing the
                  reference resolution: persistent dice pool rolled once,
                  face-up cards accumulate printed icons, temp tokens, morale
                  = unrouted units + final-round icons, defender wins ties.
  encoding.py     obs dict (numpy) -> batched torch tensors (obs_to_tensors, stack_obs).
  model.py        CombatNet: unit/card embeddings -> TransformerEncoder ->
                  GRUCell memory -> masked policy head + tanh value head.
  agent.py        PolicyAgent: wraps the model, samples actions, threads per-player
                  GRU memory (each side keeps its own belief state).
  selfplay.py     collect_episode(), build_batch() (Monte-Carlo returns),
                  ppo_update() (clipped surrogate + value loss + entropy bonus).

scripts/
  demo_env.py     Random-vs-random battle. No NN. Proves the env runs end to end.
  train.py        Self-play training loop -> checkpoints/model.pt.
  evaluate.py     Trained agent (P0) vs random (P1); reports win/draw/loss rates.
  calibrate.py    Random-vs-random win rates per stage — the fidelity check
                  against the reference simulator's balance figures.

tests/
  test_combat_env.py   Pure-python env invariants (termination, mask/phase
                       partition, dice cap, deck composition, defender-wins-
                       ties, kill/rout semantics). No torch needed.
```

### The combat model implemented (ported from the reference, 2026-07-07)

See `docs/reference_combat_model.md` for the complete spec. Headlines:

- Battle stage (early/mid/late) sampled per reset; armies of 1-5 units plus
  dice-less reinforcements sampled per reference priors; decks are 10 cards
  (5 distinct x 2) with stage-based upgrades, hand of 5.
- Each side's dice pool is rolled **once** (cap 8) and persists all battle;
  card effects mutate it (rerolls, conversions, spends, removals).
- Up to **3 rounds**; both players commit one card hidden, then cards resolve
  attacker-first (general ability + unit ability if its unit type is unrouted).
- Icons = dice + this round's temp tokens + printed icons of **all face-up
  cards** (played cards keep scoring; discard effects counter this).
- `damage = max(0, enemy Bolter - own Shield)`, assigned by the owning agent
  one target at a time: a target absorbs its full hp — killed if damage >= hp,
  routed by a smaller final hit. Routed = no morale, rally-able, killed if hit
  again.
- Winner: wipe, else morale (unrouted units + final-round Morale icons);
  **defender wins ties** — draws only on simultaneous wipes.

### Action space

Single discrete index, masked by phase:
- `0 .. MAX_HAND-1` → play hand card *i* (PLAY phase)
- `MAX_HAND .. MAX_HAND+MAX_UNITS-1` → assign pending damage to own unit *j*
  (ASSIGN phase; each hit consumes the target's full hp)

---

## 5. Deliberate simplifications / technical debt

These are **known and intentional**. Each is a future task, not a bug:

| Area | Current state | Why it's fine for now |
|------|--------------|----------------------|
| Rules source | Ported from the reference simulator's fan-rebalanced ("SoW") values, not base-game printed components | Keeps the reference balance figures valid as a calibration target |
| Ability choices | Optional/parameterised card effects (token colours, conversion amounts, 50/50s) resolve randomly, exactly like the reference | Agent decision points are card choice + damage assignment; more heads later |
| Enemy face-up cards | Encoded only as printed-icon sums in scalars, not as an entity set | The agent sees the icon contribution but not which abilities are face-up |
| Combat scope | Ground only; no bastions, no void battles | Same scope as the reference simulator |
| Returns | Monte-Carlo (outcome per player), no bootstrapping | Episodes are short (~3 rounds), so this is reasonable |
| Recurrence | Stored GRU input states, **no backprop-through-time** | Keeps the update batchable and simple |
| Policy head | Single masked distribution | Autoregressive heads only needed for compound actions (full game) |
| Factions | SM + Orks (reference also has Eldar/Chaos cards — portable later) | Enough to train and validate the pipeline |
| Reference bugs | The 4 known reference-code bugs are FIXED here (see spec §6) | May cause small deviations from reference figures |

---

## 6. Environment (done 2026-07-07)

Python 3.12.10 is installed; the venv exists at `.venv/` with all requirements.
To use it:

```powershell
cd forbidden_star_neural
.venv\Scripts\Activate.ps1
```

---

## 7. Next steps (prioritized)

### Phase 0 — Make it run & trust it ✅ (done 2026-07-07)
1. ✅ Python installed; venv created; requirements installed.
2. ✅ `python -m pytest` — 3/3 green, no rule bugs found.
3. ✅ `python -m scripts.demo_env` — sane random battle, terminates.
4. ✅ Smoke train — no shape/dtype bugs; loss finite, entropy healthy.

### Phase 1 — Confirm it actually learns ✅ (done 2026-07-07)
5. ✅ `python -m scripts.train --iterations 300` → `checkpoints/model.pt`.
6. ✅ `python -m scripts.evaluate --games 2000` — see results table in §3.
   The agent clearly beats the random baseline **for its side** (SM
   20.0% → 36.4%; as Orks it wins 87.3%). Note the original ">50%" criterion
   ignored the faction asymmetry — the right yardstick is the per-side random
   baseline.
7. ✅ Random-vs-random recorded (SM 20.0% / draw 10.3% / Orks 69.7%).
   ⚠️ Still to do: compare this against ForbiddenStarsFight's balance figures —
   the strong Ork skew needs an external fidelity check.

### Phase 2 — Make the combat real ✅ mostly done (2026-07-07)
8. ✅ Ported all 28 real **card abilities** into `card_abilities.py` (dispatch
   by faction + card id; general and unit ability slots, exact reference
   semantics, reference bugs fixed).
9. ✅ Exact dice (3/6, 2/6, 1/6; pool rolled once, cap 8) and reference damage
   rules (full-hp absorption, rout/kill, defender wins morale ties).
10. ✅ Full 4-tier rosters, 14-card decks, early/mid/late stage upgrades, army
    sampling with supply caps and reinforcements.
    **Calibration (15k random-vs-random games, `scripts.calibrate`):**
    SM attacker vs Orks — early 43.3% / mid 51.7% / late 51.9% / TOTAL 49.0%
    win. Matches the reference's "within 4 points of 50-50", including the
    early-stage Ork lean. The old simplified model's 20/70 skew is gone.
11. ✅ **League training** (2026-07-07, `scripts/train.py`): per episode the
    opponent is sampled — 50% mirror self-play / 35% frozen snapshot from
    `checkpoints/league/` (one loaded per iter, saved every `--snapshot-every`)
    / 15% the scripted heuristic. Only the learner's transitions train.
    Also added **faction-role randomization**: `reset(factions=...)` assigns
    either faction to attacker/defender per episode, with is-SM faction flags
    added to unit/card/scalar features (UNIT 9, CARD 26, SCALAR 40), so one
    net learns all four faction-role combos. `--no-league` restores pure
    mirror self-play. Swapped-direction calibration: random Orks-attacker vs
    SM wins 51.9% overall, **57.1% early** — matching the reference's "early
    Orks vs SM near the 57% threshold" almost exactly.
    Pre-league pointer-head checkpoint archived as
    `checkpoints/model_ptr_mirror_selfplay.pt` (old feature dims).

    **League results** (1000 iters, 2000 games/cell, deterministic agent):

    | Agent seat | vs random | random baseline | vs heuristic |
    |------------|-----------|-----------------|--------------|
    | SM attacker | **64.2%** | 49.0% | **53.6%** |
    | Orks defender | **64.3%** | 47.2% | **51.9%** |
    | Orks attacker | **68.7%** | 51.9% | **54.4%** |
    | SM defender | **62.8%** | 44.5% | 43.5% (random gets 25.0% in this seat) |

    +15–18 points over the random baseline in ALL FOUR faction-role combos
    (the pre-league net only knew two), beats the heuristic head-to-head in
    3 of 4 seats, and in the hardest seat (SM defending vs an Ork attacker)
    still gains +18.5 over random. No-forgetting check: beats its own iter-50
    snapshot 51.8% / 56.9%. Current best checkpoint: `checkpoints/model.pt`.

### Phase 2b — Learn under the real rules (IN PROGRESS — the current frontier)

Findings from the diagnosis trail (2026-07-07):

1. **300-iter run → exactly random baseline** (SM 48.3% vs 49.0%; Orks 46.4%
   vs 47.2%). Entropy barely declined (1.21 → 1.13) but the value head DID
   learn (~half the outcome variance) — observations informative, gradients
   flow, policy flat.
2. **1500-iter run (`model_long.pt`) → still exactly baseline** (48.0% /
   45.9%) even though entropy finally fell (→ 0.66). Sharpening without
   improving.
3. **Heuristic probe: the game HAS large learnable edge.** A simple scripted
   bot (`fsneural/heuristic.py`: attacker leans Bolter cards, absorb damage
   with the cheapest sufficient soaker) beats random by **+12 points on both
   sides** (61.3% / 60.4%). So the failure was the method, not the game.
4. **Root cause identified: the monolithic policy head.** Action logits came
   from one mean-pooled vector, so "slot i" had no stable meaning once decks
   were shuffled (in Phase 1 the hand was FIXED — slot identity was stable,
   which is why the same architecture learned there). Fixed by switching
   `model.py` to **pointer heads**: each hand slot's logit is computed from
   that card's own encoded token (+ GRU memory), likewise per-unit assign
   logits. This is the AlphaStar-style entity-pointer building block the
   README always said compound actions would need.

**Outcome: the pointer head fixed it.** 1000-iter run (healthy dynamics:
entropy 1.20 → 0.64 falling from the start, value loss down to ~0.2).
Evaluation (2000 games each, deterministic agent, `checkpoints/model.pt`):

| Matchup | Win | Draw | Loss | Reference points |
|---------|-----|------|------|------------------|
| agent P0 (SM att.) vs random | **61.2%** | 2.8% | 36.0% | baseline 49.0%, heuristic 61.3% |
| agent P1 (Orks def.) vs random | **65.8%** | 4.3% | 29.8% | baseline 47.2%, heuristic 60.4% |
| agent P0 vs heuristic | **51.1%** | 1.9% | 46.9% | |
| agent P1 vs heuristic | **53.6%** | 5.6% | 40.8% | |

The agent matches the scripted heuristic as attacker, exceeds it as defender,
and beats it head-to-head on both sides. Phase 2b CLOSED (2026-07-07).
Remaining quality work moves to Phase 3 (GAE/BPTT, opponent league —
important before trusting the policy is not a brittle self-play artifact).

### Phase 3 — Strengthen the method (IN PROGRESS 2026-07-07)
12. ✅ **GAE** implemented in `selfplay.py` (per-player trajectories, reward
    at terminal only, lambda-return value targets; `--gamma`/`--gae-lambda`
    CLI args, defaults 0.99/0.95; lam=1, gamma=1 recovers old MC behaviour).
    **Result: a wash at combat scale.** GAE-trained champion vs MC champion
    head-to-head: 49.2% overall (1000 games x 4 seats); vs random 64.1% avg
    (MC: 65.0%); vs heuristic 49.5% avg (MC: 50.9%). Expected in hindsight —
    each player makes only ~4-6 decisions per episode, so MC returns were
    already low-variance enough. GAE stays as the default (equal cost now,
    necessary at full-game horizons). Champion remains `checkpoints/model.pt`.
    ⬜ True BPTT over episodes — deferred for the same reason; revisit when
    episodes get long (full-game ladder), where memory/variance actually bind.
13. ✅ **Exploitability probe: PASSED** (2026-07-07). A fresh adversary
    trained 600 iters purely against the frozen champion
    (`python -m scripts.train --exploit checkpoints/model.pt`, random seats/
    factions) wins only **47.5% overall** head-to-head (per seat: SM att
    40.0% / Orks att 54.2% / Orks def 53.4% / SM def 42.6%, 1000 games each).
    No brittle equilibrium found at this compute scale → **DeepNash-style
    Nash dynamics deferred** — the league PPO policy is holding up. Caveat:
    the probe shares the champion's architecture and training method; a
    larger/longer/different adversary could in principle find more. Re-run
    the probe whenever the champion changes materially.

### Phase 4 — Climb toward the full game (planned 2026-07-07)

Governing principle: one mechanic per rung; `reset()/step()` stays stable;
every rung passes the same gate — rules tests → random-rollout sanity →
trained agent beats random → beats a per-rung scripted heuristic →
exploitability spot-check. (No reference simulator exists beyond combat, so
heuristic bots replace calibration as the yardstick from here.)

14. ✅ **Rungs A+B — campaign env** (built 2026-07-07, `fsneural/game_env.py`;
    training in progress). A war of 3 battles (early→mid→late), same
    `reset()/step()` interface, composing CombatEnv as a subroutine with
    per-battle attacker alternation (P0, P1, P0) and seat remapping.
    - **Build phase** before each battle: both players get 5 income, then
      alternate PUBLIC purchases — units (cost tier+1; stage/supply/cap-8
      gated) or deck-upgrade cards (t0/t2/t3 = 1/2/3 pts, each id once,
      displaces a start card) — pass is binding, empty-roster players must
      rebuy. Rung A realized as: each battle draws a fresh 5-card hand from
      the player's CURRENT purchase-shaped 10-card deck (`build_deck_from`).
    - **Persistence**: killed units leave the roster permanently; routed
      units rally; battle-spawned units are battle-only. Battle win = 1 pt;
      most points wins the campaign (+1/-1; equal = draw). NO per-battle
      reward shaping — credit flows through the value function (GAE's job).
    - **Model**: CombatNet parameterized (scalar_feats, n_build_actions);
      build actions (buy tier t / buy card id / pass = 14) get a plain head
      off memory — their indices have STABLE meanings, unlike shuffled hand
      slots. Campaign dims: ACTION_DIM 31, SCALAR_FEATS 68 (combat 40 +
      campaign block 28: build flag, battle one-hot, scores, points, owned-
      upgrade bits both sides, roster sizes).
    - **Baselines (2000 games each)**: random-vs-random P0=SM 36.4% / P0=Orks
      61.0% (early Ork strength compounds when casualties persist);
      campaign heuristic (greedy army-first build + combat heuristic) beats
      random 83-94% from every seat — the campaign hugely amplifies decision
      leverage vs single combat (~61%). Gate for the trained agent: clearly
      above per-seat random baselines, then approach the heuristic.
    - Train/evaluate via `--env campaign` (checkpoints/campaign.pt,
      league_campaign/); campaign heuristic in `fsneural/heuristic.py`;
      tests in `tests/test_game_env.py`.
    - **Results (600-iter league run, 1000 games/cell):** vs random
      **97.7-99.5%** from every seat (baselines 36-61%); vs the heuristic
      **84.0-94.1%** from every seat. The campaign's compounding decision
      leverage let the learned policy decisively surpass the scripted bot —
      a sharp contrast with single combat, where they were near-equal. GAE +
      league handled the ~40-70-step credit assignment without reward
      shaping.
    - **Exploitability probe: PASSED, with a balance finding** (600-iter
      adversary, `campaign_adversary.pt`, 1000 games/seat): adversary overall
      **49.5%** — no policy hole. But the seat split is extreme: whichever
      strong net holds Orks wins ~80% (adversary-as-Orks 79.9/81.6%,
      adversary-as-SM 17.1/19.4%). **Skilled campaign play amplifies the Ork
      edge** from 61/36 (random) to ~80/20: the compounding economy +
      persistent casualties reward the early-stage Ork combat advantage, and
      SM's late-game identity (Titans, t3 cards) doesn't get enough runway in
      a 3-battle war. Open design item for rung C+: balance knobs to consider
      — battle count / stage mix (more late-stage play), asymmetric income or
      costs, or accepting faction asymmetry and evaluating seat-paired (each
      matchup played once per side, like the reference). Training itself is
      unaffected (faction-role randomization means the net masters both
      seats).
16. **Rung C — small board** (built 2026-07-07, `fsneural/board_env.py`;
    first training in progress). 5-planet symmetric map (homes worth 2,
    flanks 1, center 2; all-pairs adjacency minus the flank-to-flank edge),
    N_ROUNDS=8 of build → move → resolve:
    - Income = 3 + controlled planet values (control = sole occupancy);
      builds deploy at home (blocked while the enemy holds it); purchase
      rules as in the campaign, tier access paced early/mid/late by round.
    - **Autoregressive movement**: alternating single-unit moves taken as two
      sequenced masked decisions — SELECT a unit (reusing the unit-pointer
      action slots 5-16) then pick an adjacent DESTINATION (5 flat planet
      actions + move-pass). The GRU carries context between the two steps; a
      'selected' flag + planet one-hot extend unit features (UNIT_FEATS 15).
      No new head machinery needed — the pointer architecture composes.
    - Battles resolve per contested planet via the combat env (attacker =
      last mover-in; decks from owned upgrades); losers' survivors retreat
      toward home (destroyed if surrounded); killed units are permanent.
    - Win: solely occupy the enemy home at round end (immediate), else most
      controlled value after 8 rounds; equal = draw. Terminal-only reward
      across 100-300 decisions.
    - Board dims: ACTION_DIM 37, SCALAR_FEATS 92, UNIT_FEATS 15 (model fully
      parameterized). `--env board` in train/evaluate; board heuristic
      (march-on-enemy-home, defend-when-invaded) in heuristic.py; tests in
      tests/test_board_env.py.
    - **Baselines (500 games)**: random-vs-random P0=SM 37.6% / P0=Orks
      57.8%; the board heuristic beats random **98-99.8%** from every seat —
      on a board, purposeful movement dominates, so "beats random" is nearly
      meaningless and THE HEURISTIC IS THE YARDSTICK for this rung.
    - **Results (400-iter league run, `checkpoints/board.pt`):** beats the
      heuristic **94.0-96.3%** from every seat (300 games/cell) and random
      100%. First-run success — the autoregressive movement + terminal-only
      reward across 100-300 decisions learned without any extra machinery
      (no BPTT needed at this map size). Game traces show a coherent learned
      **blitz**: build wide with cheap units in round 1, mass on the center,
      win the fight, storm the enemy home in round 2 for the instant win.
      **v1 probe verdict: design flaw found.** The adversary managed only
      44.9% overall but **87.3% as first-moving Orks** — the permanent
      P0-acts-first structure let a first-mover counter-blitz win the mirror
      race. Fixed by **rotating initiative per round** (as real FS does;
      exposed in scalars, SCALAR_FEATS 92→93). v1 champion archived as
      `board_v1_p0first.pt`.
    - **v2 champion** (retrained under rotation, `checkpoints/board.pt`):
      beats the heuristic 95.7 / 92.7 / 98.0 / 86.3% across the four seats.
      **v2 probe verdict: FAILED (2026-07-09).** 400-iter adversary
      (`board_adversary_v2.pt`) vs champion, 300 games/seat: as P0/SM
      **66.3%**, P0/Orks 61.7%, P1/Orks 34.0%, P1/SM 17.3% (overall 44.8%).
      Raw numbers look passable until baselined against the champion's
      self-mirror: P0/SM wins only **24.7%** and P0/Orks **87.0%** — i.e.
      skilled board play is heavily **Ork-favored (75-87% mirror)**, echoing
      the campaign's 80/20 Ork lean. So the adversary flipping P0/SM from a
      24.7% baseline to 66.3% is a **~41-point hole in the champion's
      Ork play**, not a seat effect. Lesson: judge probe seats against the
      champion's own mirror baseline, not against 50%. Remediation chosen:
      PSRO-lite — model-vs-model eval promoted into scripts/evaluate.py
      (`--opponent model:<path>`).
    - **PSRO-lite cycle + v3 probe: PASSED — rung C CLOSED (2026-07-09).**
      Seeded 3 copies of `board_adversary_v2.pt` into the league pool
      (~10% of episodes; old snapshots preserved as `v2snap_*`), resumed
      the champion 400 iters (`board.pt`; pre-league champion archived as
      `board_v2_preleague.pt`). Gauntlet: (1) heuristic per-seat
      **93.7/99.3/99.3/95.7** — worst seat UP from 86.3; (2) new mirror:
      P0/SM 14.3% / P0/Orks 84.7% — Orks now ~85% from either seat, seat
      effect gone, faction lean remains; (3) old adversary's exploit seat
      collapsed 66.3% → **6.0%**; (4) fresh 400-iter v3 adversary
      (`board_adversary_v3.pt`) per seat vs mirror baseline: P0/SM 26.7
      (+12.4), P1/Orks 88.0 (+2.7), P0/Orks 98.7 (+14.0), P1/SM 0.0
      (−14.7). Max excess +14 pts (vs v2's +41.6) — expected best-response
      margin at this compute; the exploiter specialized into Orks and
      abandoned SM entirely. **One exploiter-in-the-league cycle closed
      the hole and improved general strength — league training remains
      the default; DeepNash stays deferred until a probe fails
      post-remediation.** Balance: skilled board play is Ork-favored
      ~85/15 (knobs still noted for later).
    Rulebook fidelity decisions deferred to rung D+.
17. **Rung D — order stacks** (built 2026-07-12, `fsneural/order_env.py`;
    first training in progress). The strategic heart of FS: rung C's
    build/move phases replaced by the real planning/operations structure on
    the same 5-planet map.
    - **Planning**: initiative-first alternating placement of 4 face-down
      tokens each (pool: 2× Advance/Deploy/Strategize/Dominate) on ANY
      planet; tokens stack LIFO, covering the enemy's delays them. The
      opponent sees where/whose, never which — the Stratego-like core.
    - **Operations**: alternating, a player MUST reveal+execute one of their
      own top-of-stack tokens (skipped if none). Advance at X = pull unmoved
      units from adjacent planets into X, battle if contested (advancer =
      attacker; revealing an Advance and moving nothing is the legal bluff).
      Deploy at X = buy units arriving exhausted at X, requires sole
      occupancy (else fizzles). Strategize = buy one upgrade card.
      Dominate at X = gain PLANET_VALUE×2 if controlled — planet income
      flows ONLY through Dominate now (base income stays flat 3), so economy
      competes with tempo for order slots.
    - Simplifications: Advance pulls from all adjacent planets (not exactly
      one system); no ships/transports; token pool refreshes fully each
      round (as FS refresh does).
    - Dims: ACTION_DIM 41 (4 new order-type actions; planet + build + pass
      actions reused), SCALAR_FEATS 240 (order block 200: phase flags,
      pools, pending/executing order, per-planet stack encoding 4 deep
      top-down with OWN-ONLY type visibility, token counts). Enemy token
      types are never observable — tested (`tests/test_order_env.py`, 10
      tests: LIFO turn order, pool caps, deploy fizzle, dominate control
      gating, hidden-info encoding, termination).
    - **TBPTT implemented** (`selfplay.py: build_sequences/ppo_update_bptt`,
      `train.py --bptt CHUNK`): trajectories keep temporal order, split into
      ≤CHUNK-step chunks, chunk-initial memory from rollout (stored state,
      no burn-in), gradient flows through the GRU within a chunk. Padded
      tails masked out of every loss term. Old stored-state path untouched
      (`--bptt 0` default).
    - **Baselines (400 games/cell, 2026-07-12)**: random-vs-random P0=SM
      43.0 / P0=Orks 46.0 (draws ~12%) — the order game at random play is
      FAR more balanced than rung C's 38/58, and episodes run ~230
      decisions. Order heuristic (defend-home/deploy/press/dominate
      placement, economy-first execution, don't-feed-outnumbered advances)
      beats random 77.5/93.2/84.2/91.5 per seat — big learnable edge, yet
      NOT the 98-99% of rung C: hidden orders blunt scripted play. Gate:
      trained agent beats random clearly, then approaches/beats the
      heuristic from every seat, then the exploitability probe vs the
      mirror baseline. This is the DeepNash decision point — bluffing
      equilibria are where plain self-play PPO historically cracks.
18. **Rung E — objectives & victory** (medium): faction objective tokens on
    planets, win by collecting N over up to 8 rounds.
19. **Rung F — fidelity backlog** (ongoing): void combat, bastions, real
    materiel economy, event cards; port Eldar/Chaos from the reference
    (possible at any rung — data already mapped in the cloned repo/spec).

---

## 8. Validation checklist (how to know each layer works)

- [x] `pytest` green → env rules are self-consistent. (3/3, 2026-07-07)
- [x] `demo_env` produces sensible-looking battles that always terminate.
- [x] `train` runs without shape/dtype errors and the loss is finite.
- [x] Entropy (`H`) decreases gradually, not instantly to ~0 (1.09 → 0.46 over
      300 iters).
- [x] `evaluate` win rate vs random is well above the per-side random baseline
      (SM 36.4% vs 20.0% baseline; Orks 87.3% vs 69.7% baseline).
- [x] Random-vs-random win rate is in the right ballpark vs ForbiddenStarsFight.
      (Ported rules: SM attacker 49.0% overall vs the reference's "within 4
      points of 50-50"; early-stage Ork lean reproduced. `scripts.calibrate`.)

---

## 9. Open questions / risks

- **Reward sign correctness in self-play** is the most likely subtle bug. Each
  transition is tagged with its acting `player`; `build_batch` assigns the final
  outcome *from that player's perspective*. Verify this carefully — a sign error
  here makes the agent train against itself and never improve.
- **The recurrent simplification** (stored hidden states, no BPTT) may limit how
  much the agent can exploit hidden-info memory. Revisit in Phase 3.
- **Combat-model fidelity**: this is our interpretation of the rules. Before
  trusting any "balance" conclusions, validate against the reference repo and,
  ideally, the actual rulebook.
- **Compute**: combat is small and CPU-trainable. The full game (Phase 4) will
  need a GPU and far more self-play.
```
