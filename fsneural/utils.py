"""Small shared helpers: dice and icon pools.

Forbidden Stars combat dice are d6 with 3 Bolter faces, 2 Shield faces and
1 Morale face (exactly the reference simulator's model — see
docs/reference_combat_model.md §1). Pools are icon->count dicts. A side's
dice pool is rolled ONCE at battle start, capped at DICE_CAP, and persists
across rounds; card effects mutate it (rerolls, conversions, extra dice).
"""

import numpy as np

ICONS = ("Bolter", "Shield", "Morale")
DICE_CAP = 8
_DIE_PROBS = np.array([3 / 6, 2 / 6, 1 / 6])


def roll_dice(n, rng):
    """Roll `n` combat dice. Returns a dict of icon -> count."""
    counts = {icon: 0 for icon in ICONS}
    if n <= 0:
        return counts
    draws = rng.choice(len(ICONS), size=int(n), p=_DIE_PROBS)
    for d in draws:
        counts[ICONS[d]] += 1
    return counts


def add_icons(a, b):
    """Sum two icon dicts."""
    return {icon: a.get(icon, 0) + b.get(icon, 0) for icon in ICONS}


def pool_size(pool):
    return sum(pool.values())


def roll_extra(pool, n, rng):
    """Roll up to `n` extra dice into `pool`, respecting DICE_CAP (overflow lost)."""
    n = min(n, DICE_CAP - pool_size(pool))
    if n > 0:
        for icon, c in roll_dice(n, rng).items():
            pool[icon] += c


def reroll(pool, icon, n, rng):
    """Re-roll up to `n` dice currently showing `icon` (results replace them)."""
    n = min(n, pool[icon])
    if n <= 0:
        return
    pool[icon] -= n
    for ic, c in roll_dice(n, rng).items():
        pool[ic] += c


def reroll_all(pool, rng):
    """Re-roll every die in the pool."""
    n = pool_size(pool)
    for icon in ICONS:
        pool[icon] = 0
    for ic, c in roll_dice(n, rng).items():
        pool[ic] += c


def remove_random_dice(pool, n, rng, icons=ICONS):
    """Remove up to `n` dice picked uniformly among those showing `icons`."""
    for _ in range(n):
        avail = [ic for ic in icons for _ in range(pool[ic])]
        if not avail:
            return
        pool[avail[int(rng.integers(len(avail)))]] -= 1


def convert_random_dice(pool, n, to_icon, rng):
    """Turn up to `n` uniformly-random dice in the pool into `to_icon`."""
    for _ in range(n):
        avail = [ic for ic in ICONS for _ in range(pool[ic])]
        if not avail:
            return
        picked = avail[int(rng.integers(len(avail)))]
        pool[picked] -= 1
        pool[to_icon] += 1


def biased_count(max_n, bias, rng):
    """Sample k in 0..max_n with weights bias**k (the reference's biased_spend).

    bias 1.0 = uniform; bias < 1 favours smaller k.
    """
    if max_n <= 0:
        return 0
    weights = np.array([bias ** k for k in range(max_n + 1)])
    return int(rng.choice(max_n + 1, p=weights / weights.sum()))
