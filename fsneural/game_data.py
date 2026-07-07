"""Static combat data ported from the ForbiddenStarsFight simulator.

See docs/reference_combat_model.md for the full spec and provenance. Values
are the reference's fan-rebalanced ("SoW") set, kept identical so the
reference balance figures remain a valid calibration target.

Unit dict keys:    faction, name, tier, unit_count, hp, morale, dice
Card dict keys:    id, name, Bolter, Shield, Morale, card_type, group
Card ability implementations live in card_abilities.py, keyed by (faction, id).
"""

import copy

import numpy as np

FACTIONS = ("SM", "Orks")
STAGES = ("early", "mid", "late")

# --- Unit rosters -----------------------------------------------------------
# `dice` = dice contributed to the side's shared pool (rolled once, cap 8).
# `hp` = damage a single hit on the unit absorbs (no wound tracking: a hit
# routs the unit, or kills it outright if the incoming damage >= hp).
# `unit_count` = supply cap used when sampling armies.
UNIT_ROSTERS = {
    "SM": [
        {"faction": "SM", "name": "Scouts",         "tier": 0, "unit_count": 6, "dice": 1, "hp": 2, "morale": 2},
        {"faction": "SM", "name": "Space Marines",  "tier": 1, "unit_count": 6, "dice": 2, "hp": 3, "morale": 3},
        {"faction": "SM", "name": "Land Raiders",   "tier": 2, "unit_count": 6, "dice": 3, "hp": 4, "morale": 3},
        {"faction": "SM", "name": "Warlord Titans", "tier": 3, "unit_count": 3, "dice": 3, "hp": 5, "morale": 4},
    ],
    "Orks": [
        {"faction": "Orks", "name": "Ork Boyz",     "tier": 0, "unit_count": 9, "dice": 2, "hp": 2, "morale": 1},
        {"faction": "Orks", "name": "Nobz",         "tier": 1, "unit_count": 6, "dice": 2, "hp": 4, "morale": 2},
        {"faction": "Orks", "name": "Battlewagons", "tier": 2, "unit_count": 3, "dice": 3, "hp": 5, "morale": 2},
        {"faction": "Orks", "name": "Gargants",     "tier": 3, "unit_count": 3, "dice": 3, "hp": 6, "morale": 3},
    ],
}

MAX_TIER = 3
MAX_HP = max(u["hp"] for r in UNIT_ROSTERS.values() for u in r)
N_CARD_IDS = 14

# --- Combat cards -----------------------------------------------------------
# groups: ids 0-4 'start', 5-8 tier-0 upgrades, 9-11 tier-2, 12-13 tier-3.
def _card(cid, name, b, s, m, ctype):
    group = ("start" if cid <= 4 else "t0" if cid <= 8 else "t2" if cid <= 11 else "t3")
    return {"id": cid, "name": name, "Bolter": b, "Shield": s, "Morale": m,
            "card_type": ctype, "group": group}

CARDS = {
    "SM": [
        _card(0,  "Reconnaissance",        0, 1, 0, "both"),
        _card(1,  "Faith in the Emperor",  0, 0, 1, "def"),
        _card(2,  "Ambush",                1, 0, 0, "att"),
        _card(3,  "Fury of the Ultramar",  1, 0, 0, "att"),
        _card(4,  "Blessed Power armour",  0, 1, 0, "def"),
        _card(5,  "Hold the line",         0, 1, 1, "def"),
        _card(6,  "Glory and death",       1, 0, 1, "att"),
        _card(7,  "Veteran scouts",        1, 1, 1, "both"),
        _card(8,  "Drop Pod assault",      1, 1, 0, "both"),
        _card(9,  "Show no fear",          0, 2, 1, "def"),
        _card(10, "Break the line",        1, 2, 0, "both"),
        _card(11, "Armoured advance",      2, 1, 0, "att"),
        _card(12, "Emperor's glory",       0, 2, 2, "def"),
        _card(13, "Emperor's might",       3, 0, 0, "att"),
    ],
    "Orks": [
        _card(0,  "Slugga Boyz",           1, 1, 0, "att"),
        _card(1,  "Shoota Boyz",           2, 0, 0, "att"),
        _card(2,  "Ard Boyz",              0, 2, 0, "def"),
        _card(3,  "Gretchin",              0, 0, 0, "both"),
        _card(4,  "Mek Boyz",              0, 0, 1, "def"),
        _card(5,  "Biker Nobz",            2, 1, 0, "att"),
        _card(6,  "Sea of green",          1, 1, 0, "both"),
        _card(7,  "Waaagh!!!!",            0, 0, 3, "both"),
        _card(8,  "Mega Nobz",             1, 2, 0, "def"),
        _card(9,  "Rokkit Wagon",          3, 0, 0, "att"),
        _card(10, "Weirdboyz",             1, 1, 1, "both"),
        _card(11, "Party Wagon",           1, 2, 0, "def"),
        _card(12, "Snapper Gargant",       4, 1, 0, "att"),
        _card(13, "Smasher Gargant",       2, 3, 0, "def"),
    ],
}

_GROUP_IDS = {"start": (0, 1, 2, 3, 4), "t0": (5, 6, 7, 8), "t2": (9, 10, 11), "t3": (12, 13)}


# --- Units ------------------------------------------------------------------
def make_unit(faction, name, dice=None):
    """Fresh unit instance with live combat state."""
    base = copy.deepcopy(next(u for u in UNIT_ROSTERS[faction] if u["name"] == name))
    if dice is not None:
        base["dice"] = dice
    base["routed"] = False
    base["killed"] = False
    return base


# Army-sampling priors (reference sim_run.py):
_ATTACKER_SIZE_PROBS = [0.1, 0.2, 0.4, 0.2, 0.1]           # sizes 1..5
_TIER_PRIORS = {
    "early": [0.7, 0.25, 0.05, 0.0],
    "mid":   [0.4, 0.4, 0.2, 0.0],
    "late":  [0.1, 0.35, 0.45, 0.1],
}
_TIER_CAPS = {   # max copies of a tier per army (None = no cap beyond supply)
    "early": {2: 1, 3: 0},
    "mid":   {2: 2, 3: 0},
    "late":  {0: 2, 2: 2, 3: 1},
}
_REINF_PROBS = [0.3, 0.4, 0.2, 0.1]                         # 0..3 reinforcements


def _sample_units(faction, n, stage, rng, supply):
    """Draw `n` units by stage tier priors, honouring caps and shared supply."""
    roster = {u["tier"]: u for u in UNIT_ROSTERS[faction]}
    caps = _TIER_CAPS[stage]
    counts = {t: 0 for t in roster}
    units = []
    for _ in range(n):
        probs = np.array([
            _TIER_PRIORS[stage][t]
            if supply[roster[t]["name"]] > 0 and counts[t] < caps.get(t, 99)
            else 0.0
            for t in sorted(roster)
        ])
        if probs.sum() == 0:
            break
        tier = int(rng.choice(len(probs), p=probs / probs.sum()))
        name = roster[tier]["name"]
        units.append(make_unit(faction, name))
        counts[tier] += 1
        supply[name] -= 1
    return units


def sample_armies(faction_a, faction_d, stage, rng):
    """Sample (attacker, defender) armies per the reference setup, including
    dice-less tier-0 reinforcements for both sides."""
    supply_a = {u["name"]: u["unit_count"] for u in UNIT_ROSTERS[faction_a]}
    supply_d = {u["name"]: u["unit_count"] for u in UNIT_ROSTERS[faction_d]}

    n_a = 1 + int(rng.choice(5, p=_ATTACKER_SIZE_PROBS))
    attacker = _sample_units(faction_a, n_a, stage, rng, supply_a)

    delta = int(rng.choice([-1, 0, 1], p=[0.3, 0.6, 0.1]))
    n_d = min(5, max(1, len(attacker) + delta))
    defender = _sample_units(faction_d, n_d, stage, rng, supply_d)

    for units, faction in ((attacker, faction_a), (defender, faction_d)):
        tier0 = UNIT_ROSTERS[faction][0]["name"]
        n_r = min(int(rng.choice(4, p=_REINF_PROBS)), len(units))
        units.extend(make_unit(faction, tier0, dice=0) for _ in range(n_r))
    return attacker, defender


# --- Decks ------------------------------------------------------------------
def build_deck(faction, stage, rng):
    """Build a 10-card deck (5 distinct ids x 2) per the reference's stage-based
    upgrade rules; returns (hand, undrawn) of 5 fresh card copies each."""
    if stage == "early":
        n_up = {"t0": int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))}
    elif stage == "mid":
        n_up = {"t0": 2, "t2": int(rng.choice([1, 2], p=[0.5, 0.5]))}
    else:
        n_up = {"t0": 1, "t2": 2, "t3": int(rng.choice([1, 2], p=[0.8, 0.2]))}

    ids = []
    for group, n in n_up.items():
        ids.extend(rng.choice(_GROUP_IDS[group], size=n, replace=False).tolist())
    ids.extend(rng.choice(_GROUP_IDS["start"], size=5 - len(ids), replace=False).tolist())

    by_id = {c["id"]: c for c in CARDS[faction]}
    deck = [copy.deepcopy(by_id[i]) for i in ids for _ in (0, 1)]
    rng.shuffle(deck)
    return deck[:5], deck[5:]


def build_deck_from(faction, upgrade_ids, rng):
    """Build a deck from an explicit set of owned upgrade ids (campaign play):
    the owned upgrades plus randomly-chosen start cards to fill 5 distinct ids,
    each x2; shuffle; return (hand of 5, undrawn 5)."""
    ids = list(upgrade_ids)
    n_start = 5 - len(ids)
    if n_start > 0:
        ids += rng.choice(_GROUP_IDS["start"], size=n_start, replace=False).tolist()
    by_id = {c["id"]: c for c in CARDS[faction]}
    deck = [copy.deepcopy(by_id[i]) for i in ids for _ in (0, 1)]
    rng.shuffle(deck)
    return deck[:5], deck[5:]
