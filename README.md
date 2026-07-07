# forbidden_star_neural

A reinforcement-learning testbed that trains a neural network to play the
**combat** sub-game of *Forbidden Stars* — a 2-faction, head-to-head fight with
dice and combat cards — via **self-play**.

> **New here? Read [`DEVELOPMENT.md`](DEVELOPMENT.md) first** — it logs every
> design decision, the current status, known simplifications, and a prioritized
> next-steps plan.

It is deliberately scoped to *combat only* (no board, movement, order stacks, or
objectives yet), because combat is the smallest slice that still has every hard
property of the full game: **2-player zero-sum, hidden information (card hands),
and stochasticity (dice).** That makes it the right first rung on the ladder and
a clean match for self-play RL.

> Inspiration / rules reference: the combat model is informed by the
> [ForbiddenStarsFight](https://github.com/riik-db/ForbiddenStarsFight)
> Monte-Carlo simulator. That project plays combat with *random* decisions to
> study balance; here we replace the random card/target choices with a **learned
> policy** and train it to win.

## The architecture (why these pieces)

This mirrors the stack that large-game agents (AlphaStar, OpenAI Five, DeepNash)
converged on, shrunk to the combat mini-game:

```
 observation (units + hand + scalars)
        │
   per-entity embeddings        ← units and cards are *sets*, variable size
        │
   self-attention encoder       ← captures unit/card interactions
        │
   recurrent memory (GRU)       ← belief state across a hidden-info episode
        │
   ┌────┴─────┐
 policy head   value head        ← masked action logits  +  win prediction
```

- **Paradigm:** model-free self-play policy gradient (a simplified, PPO-style
  clipped update). This is the forgiving first choice for a messy, hidden-info
  game; swap toward DeepNash/ReBeL-style Nash dynamics later if the policy
  proves exploitable.
- **Why not AlphaZero/MCTS here?** Hidden card hands mean you can't expand the
  true game tree honestly. Model-free self-play sidesteps that.

## Layout

| Path | Purpose |
|------|---------|
| `docs/reference_combat_model.md` | The ported rules spec (from the reference simulator) |
| `fsneural/game_data.py` | Real rosters, 14-card decks, stage-based deck/army sampling |
| `fsneural/card_abilities.py` | All 28 card abilities (general + unit slots) |
| `fsneural/combat_env.py` | Gym-style `reset()/step()` 2-player combat env |
| `fsneural/encoding.py` | Turns env observations into batched torch tensors |
| `fsneural/model.py` | Policy/value network (embeddings → attention → GRU → heads) |
| `fsneural/agent.py` | Wraps the model: action sampling, hidden-state threading |
| `fsneural/selfplay.py` | Self-play rollout + PPO-style training update |
| `fsneural/utils.py` | Dice + small helpers |
| `scripts/demo_env.py` | Random-vs-random battle — sanity-checks the env |
| `scripts/train.py` | Self-play training entry point |
| `scripts/evaluate.py` | Trained agent vs random baseline |
| `scripts/calibrate.py` | Random-vs-random balance check vs the reference simulator |
| `tests/test_combat_env.py` | Basic env invariants |

## Setup

```bash
cd forbidden_star_neural
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## Use

```bash
python -m scripts.demo_env        # watch a random battle, confirm the env runs
python -m scripts.train           # self-play training, saves checkpoints/model.pt
python -m scripts.evaluate        # win rate vs random — should beat ~50%
```

A trained agent should beat the random-play baseline; the
ForbiddenStarsFight random-play win rates are a useful external calibration
target.

## Status / honest caveats

- The combat rules are ported from the reference simulator (ground combat
  only; the reference's fan-rebalanced values) and **calibrated against its
  balance figures**: random-vs-random SM vs Orks lands at 49% (see
  `DEVELOPMENT.md` §3 and `docs/reference_combat_model.md`).
- `selfplay.py` uses Monte-Carlo returns with a value baseline and a clipped
  objective — a deliberately simple starting point. Proper GAE and recurrent
  advantage handling are marked `TODO`.
- The policy head is a single masked distribution over the combined action set.
  Genuine **autoregressive action heads** become necessary when you scale to the
  full game (compound multi-unit orders) — noted as the extension point.

## Roadmap

1. ✅ Combat env + self-play loop.
2. ✅ Real combat rules (all card abilities, exact dice, deck/army sampling);
   win rates validated against the reference simulator.
3. Opponent league; GAE/BPTT (see `DEVELOPMENT.md` Phase 3).
4. Move up the ladder: full board, order stacks, movement, objectives.
5. Swap in DeepNash-style training if exploitability matters.
