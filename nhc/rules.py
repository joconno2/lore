"""Auto-generated NetHack symbolic rules.

Hard-coded domain knowledge from the NetHack wiki. Used as action
masks/filters during training (E7/E8). No LLM at runtime.

Usage:
    from nhc.rules import should_eat_corpse, should_pray, is_safe_to_engrave_elbereth
"""
from __future__ import annotations

import numpy as np
from nle import nethack

# ============================================================
# Food safety: which corpses are safe to eat
# ============================================================

# Monsters whose corpses give intrinsics (always eat if possible)
BENEFICIAL_CORPSES = {
    "floating eye",        # telepathy
    "wraith",              # level drain immunity + XP
    "newt",                # mana restore chance
    "tengu",               # teleport control
    "stalker",             # invisibility + see invisible
    "disenchanter",        # disenchant resistance
    "quantum mechanic",    # speed / teleportitis
}

# Monsters whose corpses are always dangerous
POISONOUS_CORPSES = {
    "killer bee", "giant spider", "scorpion", "pit viper",
    "cobra", "water moccasin", "asp", "python",
    "rabid rat", "rabid jackal", "werewolf", "werejackal", "wererat",
    "green slime",  # sliming (instant death without cure)
}

# Acidic corpses (safe to eat, give acid resistance eventually)
ACIDIC_CORPSES = {
    "yellow light", "acid blob", "blue jelly",
}

# Corpses that can petrify you
PETRIFYING_CORPSES = {
    "cockatrice", "chickatrice",  # instant death if not stoneproof
}

# Corpses with bad effects
BAD_CORPSES = {
    "jackal",          # lycanthropy risk (werewolf)
    "bat",             # stunning
    "yellow light",    # blindness (but gives acid res)
    "floating eye",    # safe ONLY if you have telepathy already or want it
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
    return True  # most corpses are safe


# ============================================================
# Prayer timing
# ============================================================

def should_pray(hp: int, max_hp: int, hunger: int, turn: int,
                last_pray_turn: int, alignment: int) -> bool:
    """Whether prayer is likely to succeed and useful.

    Prayer succeeds if:
    - Alignment is non-negative
    - At least 300 turns since last prayer (500 for non-lawful)
    - On appropriate level (not in Gehennom without Amulet)
    """
    if alignment < 0:
        return False
    if turn - last_pray_turn < 300:
        return False
    # Pray when HP is critically low
    if hp <= max_hp // 7:
        return True
    # Pray when starving
    if hunger >= 4:  # FAINTING or worse
        return True
    return False


# ============================================================
# Elbereth
# ============================================================

# Monsters that respect Elbereth (most do)
ELBERETH_IMMUNE = {
    "minotaur",  # never respects Elbereth
    # Unique/quest monsters and riders also ignore it
    "death", "pestilence", "famine",
    "wizard of yendor",
}

# Blind monsters can't see Elbereth
# Monsters with hands can smudge it

def is_safe_to_engrave_elbereth(monster_name: str) -> bool:
    """Whether Elbereth will scare this monster."""
    return monster_name.lower() not in ELBERETH_IMMUNE


# ============================================================
# Item identification by price
# ============================================================

# Base prices for common scroll identification
SCROLL_PRICES = {
    # price -> possible scrolls (most useful listed first)
    20: ["identify", "light"],
    50: ["enchant weapon", "enchant armor", "remove curse"],
    60: ["teleportation", "gold detection"],
    80: ["magic mapping", "fire"],
    100: ["charging", "genocide", "punishment", "stinking cloud"],
    200: ["earth"],
    300: ["blank"],
}

POTION_PRICES = {
    # price -> possible potions
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
# Danger assessment
# ============================================================

EXTREMELY_DANGEROUS = {
    "cockatrice",    # touch = stone
    "mind flayer",   # intelligence drain
    "master mind flayer",
    "arch-lich",     # summons + curses
    "green slime",   # sliming
    "yellow light",  # blindness explosion
    "floating eye",  # paralysis on melee (let them come to you)
}


def assess_danger(monster_name: str, player_level: int) -> str:
    """Quick danger assessment: 'safe', 'caution', 'dangerous', 'deadly'."""
    name = monster_name.lower()
    if name in EXTREMELY_DANGEROUS:
        return "deadly"
    if name in {"minotaur", "titan", "balrog"}:
        if player_level < 15:
            return "dangerous"
    return "safe"


# ============================================================
# Dungeon navigation rules
# ============================================================

def should_descend(dlvl: int, xl: int, hp: int, max_hp: int,
                   has_key_items: bool = False) -> bool:
    """Heuristic for whether to go deeper.

    Conservative: don't descend if underleveled or low HP.
    """
    if hp < max_hp * 0.5:
        return False
    # Rough guideline: don't go deeper than 2x your XL
    if dlvl > xl * 2:
        return False
    return True


# ============================================================
# Action mask helpers (integrate with NLE action space)
# ============================================================

def build_safety_mask(blstats: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Modify action mask based on safety rules.

    Prevents known-bad actions like eating when not hungry (wastes nutrition),
    praying when it won't work, etc.

    Args:
        blstats: NLE blstats vector (27 dims)
        action_mask: current legal action mask (121 bools)

    Returns:
        Modified action mask
    """
    # blstats indices (from NLE source):
    # 0-1: x,y  2: strength  10: HP  11: max_HP  12: depth
    # 18: hunger  21: experience_level  22: alignment
    mask = action_mask.copy()
    # For now, pass through. Rules will be wired in as we validate each one.
    return mask
