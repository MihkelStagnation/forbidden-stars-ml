"""Card ability implementations, ported from the reference simulator's
card_abilities.py (see docs/reference_combat_model.md §3-4, including the
reference bugs we deliberately fix).

Every card has a GENERAL ability (always fires when played) and optionally a
UNIT ability that fires only if a required unit type is alive and unrouted.
Abilities mutate the env directly through its small primitive API:
    env.dice[p] / env.temp[p]            icon pools
    env.flags[p] / env.continuous[p]     per-round / persistent effect flags
    env.gain_temp(p, ...)                temp tokens (triggers Weirdboyz copy)
    env.rally(p, all_units=False)        unroute, highest tier first
    env.rout_lowest(p)                   rout lowest-tier unrouted unit
    env.spawn(p, name)                   add a dice-less unit
    env.discard_faceup(p)                remove a random face-up card
    env.unrouted(p) / env.alive(p)       unit queries

Choices the reference resolves randomly (token colours, conversion amounts,
optional 50/50 effects) stay random here; the RL agent's decision points are
card choice and damage assignment only. Round-round flags set here:
    'cannot_rout'    (SM 9)  damage smaller than the target's hp is discarded
    'ambush'         (SM 2)  enemy rout upgrades to kill unless a Morale die is spent
    'double_damage'  (SM 11) opponent's pending damage is doubled
    'fury_strip'     (SM 3)  at icon-summing, strip enemy temp Shield or a Shield die
Persistent: 'copy_temp' (Orks 10) copy opponents' ability-generated temp tokens.
"""

from .utils import (
    DICE_CAP, pool_size, roll_extra, reroll, reroll_all,
    remove_random_dice, convert_random_dice, biased_count,
)


def _coin(env):
    return env.rng.random() < 0.5


# --------------------------------------------------------------- SM: general
def _sm_g0(env, p, r):  # Reconnaissance
    if _coin(env):
        env.gain_temp(p, bolter=2)
    else:
        env.gain_temp(p, shield=2)

def _sm_g1(env, p, r):  # Faith in the Emperor
    roll_extra(env.dice[p], 1, env.rng)

def _sm_g2(env, p, r):  # Ambush
    env.gain_temp(p, bolter=2)

def _sm_g3(env, p, r):  # Fury of the Ultramar
    reroll(env.dice[1 - p], "Shield", 1, env.rng)
    if _coin(env):
        reroll(env.dice[p], "Shield", 1, env.rng)

def _sm_g4(env, p, r):  # Blessed Power armour
    env.gain_temp(p, shield=2)

def _sm_g5(env, p, r):  # Hold the line
    env.gain_temp(p, shield=2)
    if p != 0:  # defender
        env.rally(p)

def _sm_g6(env, p, r):  # Glory and death
    env.gain_temp(p, bolter=2)
    if p == 0:  # attacker
        env.rally(p)

def _sm_g7(env, p, r):  # Veteran scouts
    n = env.dice[p]["Morale"]
    b = int(env.rng.binomial(n, 0.5)) if n else 0
    env.gain_temp(p, bolter=b, shield=n - b)

def _sm_g8(env, p, r):  # Drop Pod assault
    roll_extra(env.dice[p], 1, env.rng)

def _sm_g9(env, p, r):  # Show no fear
    env.flags[p].add("cannot_rout")

def _sm_g10(env, p, r):  # Break the line
    bias = {1: 1.0, 2: 0.45, 3: 0.35}[r]
    n = biased_count(min(3, env.dice[p]["Morale"]), bias, env.rng)
    for _ in range(n):
        env.dice[p]["Morale"] -= 1
        env.dice[p]["Bolter" if _coin(env) else "Shield"] += 1

def _sm_g11(env, p, r):  # Armoured advance
    roll_extra(env.dice[p], 1, env.rng)

def _sm_g12(env, p, r):  # Emperor's glory
    roll_extra(env.dice[p], 2, env.rng)

def _sm_g13(env, p, r):  # Emperor's might
    roll_extra(env.dice[p], 2, env.rng)


# ----------------------------------------------------------------- SM: unit
def _sm_u1(env, p, r):  # Faith in the Emperor (Scouts | Space Marines)
    if pool_size(env.dice[p]) >= DICE_CAP or (env.routed_units(p) and _coin(env)):
        env.rally(p)
    else:
        env.dice[p]["Morale"] += 1

def _sm_u2(env, p, r):  # Ambush (Scouts)
    env.flags[p].add("ambush")

def _sm_u3(env, p, r):  # Fury of the Ultramar (Space Marines)
    env.flags[p].add("fury_strip")

def _sm_u4(env, p, r):  # Blessed Power armour (Space Marines)
    convert_random_dice(env.dice[p], 2, "Shield", env.rng)

def _sm_u5(env, p, r):  # Hold the line (Space Marines)
    if pool_size(env.dice[p]) < DICE_CAP:
        env.dice[p]["Shield" if _coin(env) else "Morale"] += 1

def _sm_u6(env, p, r):  # Glory and death (Space Marines)
    remove_random_dice(env.dice[1 - p], 1, env.rng, icons=("Shield", "Morale"))

def _sm_u8(env, p, r):  # Drop Pod assault (Space Marines, >=1 Morale die)
    if env.dice[p]["Morale"] >= 1:
        env.dice[p]["Morale"] -= 1
        env.spawn(p, "Scouts" if _coin(env) else "Space Marines")

def _sm_u9(env, p, r):  # Show no fear (Space Marines)
    had_routed = bool(env.routed_units(p))
    env.rally(p, all_units=True)
    if had_routed and env.dice[p]["Morale"] >= 1:
        env.dice[p]["Morale"] -= 1

def _sm_u10(env, p, r):  # Break the line (Land Raiders)
    if p != 0 or r >= 2:  # attacker only from round 2; defender any round
        env.discard_faceup(1 - p)

def _sm_u11(env, p, r):  # Armoured advance (Land Raiders)
    env.flags[p].add("double_damage")

def _sm_u12(env, p, r):  # Emperor's glory (Warlord Titans)
    env.rally(p, all_units=True)
    bias = {1: 0.35, 2: 0.45, 3: 1.0}[r]
    for icon in ("Bolter", "Shield"):
        n = biased_count(min(DICE_CAP, env.dice[p][icon]), bias, env.rng)
        env.dice[p][icon] -= n
        env.dice[p]["Morale"] += n

def _sm_u13(env, p, r):  # Emperor's might (Warlord Titans)
    n = int(env.rng.integers(0, min(DICE_CAP, env.dice[p]["Bolter"]) + 1))
    env.dice[p]["Bolter"] -= n
    env.gain_temp(p, bolter=2 * n)


# ------------------------------------------------------------- Orks: general
def _ork_g0(env, p, r):  # Slugga Boyz
    reroll(env.dice[p], "Morale", 99, env.rng)
    reroll(env.dice[1 - p], "Morale", 99, env.rng)

def _ork_g1(env, p, r):  # Shoota Boyz
    reroll(env.dice[p], "Shield", 99, env.rng)

def _ork_g2(env, p, r):  # Ard Boyz
    reroll(env.dice[p], "Bolter", 99, env.rng)

def _ork_g3(env, p, r):  # Gretchin
    env.gain_temp(p, bolter=1, shield=1)
    opp = env.dice[1 - p]
    avail = [ic for ic in opp for _ in range(opp[ic])]
    if avail:
        reroll(opp, avail[int(env.rng.integers(len(avail)))], 1, env.rng)

def _ork_g4(env, p, r):  # Mek Boyz
    roll_extra(env.dice[p], 1, env.rng)

def _ork_g5(env, p, r):  # Biker Nobz
    reroll(env.dice[1 - p], "Bolter", 99, env.rng)

def _ork_g6(env, p, r):  # Sea of green
    env.spawn(p, "Ork Boyz")
    opp = 1 - p
    if len(env.unrouted(p)) > len(env.unrouted(opp)):
        has_morale = env.dice[opp]["Morale"] > 0
        if has_morale and (not env.unrouted(opp) or _coin(env)):
            env.dice[opp]["Morale"] -= 1
        else:
            env.rout_lowest(opp)

def _ork_g7(env, p, r):  # Waaagh!!!!
    env.rally(p)

def _ork_g8(env, p, r):  # Mega Nobz
    reroll(env.dice[1 - p], "Shield", 99, env.rng)

def _ork_g10(env, p, r):  # Weirdboyz
    reroll_all(env.dice[p], env.rng)
    reroll_all(env.dice[1 - p], env.rng)

def _ork_g11(env, p, r):  # Party Wagon
    env.spawn(p, "Ork Boyz")


# ---------------------------------------------------------------- Orks: unit
def _n_boyz(env, p):
    return sum(1 for u in env.unrouted(p) if u["name"] == "Ork Boyz")

def _ork_u0(env, p, r):  # Slugga Boyz (Ork Boyz, has routed unit)
    if env.routed_units(p):
        env.rally(p)

def _ork_u1(env, p, r):  # Shoota Boyz (Ork Boyz)
    reroll(env.dice[1 - p], "Shield", _n_boyz(env, p), env.rng)

def _ork_u2(env, p, r):  # Ard Boyz (Ork Boyz)
    reroll(env.dice[1 - p], "Bolter", _n_boyz(env, p), env.rng)

def _ork_u4(env, p, r):  # Mek Boyz (Ork Boyz) — steal top undrawn enemy card
    opp = 1 - p
    if env.undrawn[opp]:
        card = env.undrawn[opp].pop()
        env.gain_temp(p, bolter=card["Bolter"], shield=card["Shield"],
                      morale=card["Morale"])

def _ork_u5(env, p, r):  # Biker Nobz (Nobz)
    env.gain_temp(p, bolter=1)

def _ork_u7(env, p, r):  # Waaagh!!!! (Ork Boyz)
    env.gain_temp(p, bolter=_n_boyz(env, p))

def _ork_u8(env, p, r):  # Mega Nobz (Nobz)
    env.gain_temp(p, shield=1)

def _ork_u9(env, p, r):  # Rokkit Wagon (Battlewagons)
    env.gain_temp(p, bolter=3)

def _ork_u10(env, p, r):  # Weirdboyz (Ork Boyz) — continuous copy
    env.continuous[p].add("copy_temp")

def _ork_u11(env, p, r):  # Party Wagon (Battlewagons, more unrouted than enemy)
    if len(env.unrouted(p)) > len(env.unrouted(1 - p)):
        env.gain_temp(p, bolter=2, shield=2)

def _ork_u12(env, p, r):  # Snapper Gargant (Gargants)
    if p != 0 or r >= 2:
        env.discard_faceup(1 - p)

def _ork_u13(env, p, r):  # Smasher Gargant (Gargants)
    opp = 1 - p
    pool = env.unrouted(opp) or env.routed_units(opp)
    if not pool:
        return
    target = max(pool, key=lambda u: u["tier"])
    if target["tier"] > 0 and _coin(env):
        target["routed"] = True
        target["killed"] = True
    else:
        remove_random_dice(env.dice[opp],
                           min(target["tier"], pool_size(env.dice[opp])), env.rng)


# ------------------------------------------------------------------ dispatch
GENERAL = {
    "SM": [_sm_g0, _sm_g1, _sm_g2, _sm_g3, _sm_g4, _sm_g5, _sm_g6, _sm_g7,
           _sm_g8, _sm_g9, _sm_g10, _sm_g11, _sm_g12, _sm_g13],
    "Orks": [_ork_g0, _ork_g1, _ork_g2, _ork_g3, _ork_g4, _ork_g5, _ork_g6,
             _ork_g7, _ork_g8, None, _ork_g10, _ork_g11, None, None],
}

# card id -> (required unit names, fn). Requirement: any alive UNROUTED unit
# whose name is in the tuple.
UNIT_ABILITIES = {
    "SM": {
        1: (("Scouts", "Space Marines"), _sm_u1),
        2: (("Scouts",), _sm_u2),
        3: (("Space Marines",), _sm_u3),
        4: (("Space Marines",), _sm_u4),
        5: (("Space Marines",), _sm_u5),
        6: (("Space Marines",), _sm_u6),
        8: (("Space Marines",), _sm_u8),
        9: (("Space Marines",), _sm_u9),
        10: (("Land Raiders",), _sm_u10),
        11: (("Land Raiders",), _sm_u11),
        12: (("Warlord Titans",), _sm_u12),
        13: (("Warlord Titans",), _sm_u13),
    },
    "Orks": {
        0: (("Ork Boyz",), _ork_u0),
        1: (("Ork Boyz",), _ork_u1),
        2: (("Ork Boyz",), _ork_u2),
        4: (("Ork Boyz",), _ork_u4),
        5: (("Nobz",), _ork_u5),
        7: (("Ork Boyz",), _ork_u7),
        8: (("Nobz",), _ork_u8),
        9: (("Battlewagons",), _ork_u9),
        10: (("Ork Boyz",), _ork_u10),
        11: (("Battlewagons",), _ork_u11),
        12: (("Gargants",), _ork_u12),
        13: (("Gargants",), _ork_u13),
    },
}


def apply_card(env, p, card, round_id):
    """Resolve a just-played card: general ability, then unit ability if its
    required unit type is alive and unrouted."""
    faction = env.factions[p]
    general = GENERAL[faction][card["id"]]
    if general is not None:
        general(env, p, round_id)
    entry = UNIT_ABILITIES[faction].get(card["id"])
    if entry is not None:
        required, fn = entry
        if any(u["name"] in required for u in env.unrouted(p)):
            fn(env, p, round_id)
