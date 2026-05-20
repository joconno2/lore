"""NetHack domain knowledge from the wiki.

Structured corpse effects, monster properties, item prices, and
situational rules. Used by kb.py to build the entity table and
rule table consumed by KBConditioner.

Regenerated with Claude Opus (May 2026), replacing incomplete
Qwen extraction.
"""
from __future__ import annotations

import numpy as np
from nle import nethack

# ============================================================
# Food safety: corpse effects
# ============================================================

# Corpses that grant beneficial intrinsics (always eat)
BENEFICIAL_CORPSES = {
    # Resistances
    "floating eye",        # telepathy
    "stalker",             # invisibility + see invisible
    "yellow light",        # blindness resistance (and explodes, but corpse is fine)
    "fire ant",            # fire resistance
    "fire giant",          # fire resistance
    "red dragon",          # fire resistance
    "hell hound pup",      # fire resistance
    "hell hound",          # fire resistance
    "pyrolisk",            # fire resistance
    "frost giant",         # cold resistance
    "blue dragon",         # cold resistance
    "winter wolf cub",     # cold resistance
    "winter wolf",         # cold resistance
    "brown mold",          # cold resistance (if eaten, not melee)
    "storm giant",         # shock resistance
    "blue jelly",          # shock resistance
    "electric eel",        # shock resistance
    "black dragon",        # disintegration resistance
    "green dragon",        # poison resistance
    "yellow dragon",       # acid resistance
    "orange dragon",       # sleep resistance
    "white dragon",        # cold resistance
    "gray unicorn",        # poison resistance
    # Other beneficial
    "wraith",              # gain level
    "newt",                # mana restore chance
    "tengu",               # teleport control (or teleportitis)
    "disenchanter",        # disenchant resistance
    "quantum mechanic",    # speed (or teleportitis)
    "nurse",               # heal HP (but only if no armor on)
    "lizard",              # cures stoning, never rots, always safe
    "lichen",              # never rots, vegetarian safe
    "giant",               # strength gain
    "hill giant",          # strength gain
    "stone giant",         # strength gain
    "lord surtur",         # fire resistance
}

# Corpses that are poisonous (need poison resistance)
POISONOUS_CORPSES = {
    "killer bee", "giant spider", "scorpion", "pit viper",
    "cobra", "water moccasin", "asp", "python",
    "rabid rat", "rabid jackal",
    "werewolf", "werejackal", "wererat",
    "green slime",         # sliming (instant death without cure)
    "quasit",              # poison
    "rotted corpse",       # sickness
}

# Acidic corpses (safe if acid resistant, otherwise hurt)
ACIDIC_CORPSES = {
    "yellow light", "acid blob", "blue jelly",
    "gelatinous cube",
}

# Corpses that can petrify you (never eat without stoneproof)
PETRIFYING_CORPSES = {
    "cockatrice", "chickatrice",
}

# Corpses with lycanthropy risk
LYCANTHROPIC_CORPSES = {
    "jackal",    # werewolf if unlucky
    "rat",       # wererat
    "wolf",      # werewolf
}


def should_eat_corpse(monster_name: str, has_poison_res: bool = False,
                      has_stone_res: bool = False) -> bool:
    """Whether it's safe to eat this corpse."""
    name = monster_name.lower()
    if name in PETRIFYING_CORPSES and not has_stone_res:
        return False
    if name == "green slime":
        return False
    if name in POISONOUS_CORPSES and not has_poison_res:
        return False
    if name in BENEFICIAL_CORPSES:
        return True
    return True


# ============================================================
# Prayer timing
# ============================================================

def should_pray(hp: int, max_hp: int, hunger: int, turn: int,
                last_pray_turn: int, alignment: int) -> bool:
    """Whether prayer is likely to succeed and useful."""
    if alignment < 0:
        return False
    if turn - last_pray_turn < 300:
        return False
    if hp <= max_hp // 7:
        return True
    if hunger >= 4:
        return True
    return False


# ============================================================
# Elbereth
# ============================================================

# Monsters that ignore Elbereth
ELBERETH_IMMUNE = {
    # Unique named demons
    "Demogorgon", "Asmodeus", "Baalzebub", "Orcus", "Juiblex",
    "Yeenoghu", "Dispater", "Geryon",
    # Riders
    "Death", "Pestilence", "Famine",
    # Quest and special
    "Wizard of Yendor", "Medusa",
    # Minotaur
    "minotaur",
    # Human @ class (guards, shopkeepers, priests, etc.)
    "shopkeeper", "guard", "aligned priest", "high priest",
    # Angels and archons
    "Archon",
}

# Blind monsters can't see Elbereth
# Monsters with hands can smudge engraved Elbereth

def is_safe_to_engrave_elbereth(monster_name: str) -> bool:
    """Whether Elbereth will scare this monster."""
    return monster_name not in ELBERETH_IMMUNE


# ============================================================
# Monsters with special melee contact effects
# ============================================================

MELEE_RISK_MONSTERS = {
    # Paralysis
    "floating eye",        # gaze paralyzes on melee
    "gelatinous cube",     # engulf paralyzes
    # Petrification
    "cockatrice",          # touch = stone
    "chickatrice",         # touch = stone
    # Sliming
    "green slime",         # touch = sliming
    # Level drain
    "vampire",             # level drain bite
    "vampire lord",
    "vampire king",        # if present in version
    "Vlad the Impaler",
    # Intelligence drain
    "mind flayer",         # brain eating
    "master mind flayer",
    # Passive damage
    "blue jelly",          # passive cold
    "brown mold",          # passive cold (explodes)
    "yellow mold",         # passive acid
    "acid blob",           # passive acid
    # Grabbing/drowning
    "electric eel",        # drowning attack
    "giant eel",           # drowning attack
    "kraken",              # drowning attack
    # Rust
    "rust monster",        # destroys metal armor/weapons
    # Illness
    "Pestilence",          # terminal illness
}


# ============================================================
# Danger assessment
# ============================================================

# Monsters that should always be avoided or approached with caution
EXTREMELY_DANGEROUS = {
    # Instant/near-instant death
    "cockatrice",          # touch petrification
    "chickatrice",
    "Medusa",              # gaze petrification
    "green slime",         # sliming
    "Demogorgon",          # disease + summoning + sting
    # Brain drain
    "mind flayer",
    "master mind flayer",
    # Powerful magic
    "arch-lich",           # double trouble, curses, summons
    "master lich",
    "Wizard of Yendor",    # steals amulet, summons, harasses
    # Riders
    "Death",               # touch of death
    "Pestilence",          # terminal illness
    "Famine",              # starvation
    # Dangerous large monsters
    "purple worm",         # swallow, digestion
    "minotaur",            # very strong melee, ignores Elbereth
    "titan",               # strong melee + magic
    "balrog",
    # Paralysis (deadly if other monsters nearby)
    "floating eye",        # gaze = paralysis on melee
}


def assess_danger(monster_name: str, player_level: int) -> str:
    """Quick danger assessment."""
    name = monster_name.lower()
    if name in {n.lower() for n in EXTREMELY_DANGEROUS}:
        return "deadly"
    if name in {"minotaur", "titan", "balrog"} and player_level < 15:
        return "dangerous"
    return "safe"


# ============================================================
# Item identification by price
# ============================================================

SCROLL_PRICES = {
    20: ["identify", "light"],
    50: ["enchant weapon", "enchant armor", "remove curse"],
    60: ["teleportation", "gold detection"],
    80: ["magic mapping", "fire"],
    100: ["charging", "genocide", "punishment", "stinking cloud"],
    200: ["earth"],
    300: ["blank"],
}

POTION_PRICES = {
    0: ["uncursed water"],
    50: ["booze", "fruit juice", "see invisible", "sickness"],
    100: ["confusion", "extra healing", "hallucination", "healing",
          "restore ability", "sleeping"],
    150: ["blindness", "gain ability", "invisibility", "monster detection",
          "object detection"],
    200: ["enlightenment", "full healing", "levitation", "polymorph", "speed"],
    250: ["acid", "oil"],
    300: ["gain energy", "gain level", "holy water", "unholy water"],
}


# ============================================================
# Dungeon navigation rules
# ============================================================

def should_descend(dlvl: int, xl: int, hp: int, max_hp: int,
                   has_key_items: bool = False) -> bool:
    """Heuristic for whether to go deeper."""
    if hp < max_hp * 0.5:
        return False
    if dlvl > xl * 2:
        return False
    return True


# ============================================================
# Action mask helpers
# ============================================================

def build_safety_mask(blstats: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Modify action mask based on safety rules."""
    mask = action_mask.copy()
    return mask
