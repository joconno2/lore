"""Structured Knowledge Base for NetHack.

Extracts entity properties and situational rules from NLE's built-in
monster data and the hand-coded wiki rules in nhc/rules.py. The KB is
a static Python data structure: no LLM at runtime.

Two components consumed by KBConditioner:

1. Entity table: glyph_id -> property vector (danger, corpse_effect, ...)
   Covers all 381 monsters + pet/body/ridden variants via glyph offset.

2. Rule table: list of (condition_fn, description) pairs for situational
   rules derived from wiki knowledge. Each rule maps blstats + nearby
   glyphs to an action recommendation.

Usage:
    from nhc.kb import build_entity_table, build_rule_table, GLYPH_CLASS
    entity_props = build_entity_table()   # (MAX_GLYPH, N_ENTITY_FEATURES)
    rules = build_rule_table()            # list of Rule namedtuples
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from nle import nethack

from nhc.rules import (
    BENEFICIAL_CORPSES, POISONOUS_CORPSES, PETRIFYING_CORPSES,
    EXTREMELY_DANGEROUS, ELBERETH_IMMUNE,
)

# ============================================================
# Constants
# ============================================================

NUM_MONSTERS = nethack.GLYPH_PET_OFF - nethack.GLYPH_MON_OFF  # 381
MAX_GLYPH = nethack.MAX_GLYPH  # 5976

# Glyph class IDs (for entity_type_weights in meta-controller)
GLYPH_CLASS_MONSTER = 0
GLYPH_CLASS_PET = 1
GLYPH_CLASS_INVIS = 2
GLYPH_CLASS_DETECT = 3
GLYPH_CLASS_BODY = 4
GLYPH_CLASS_RIDDEN = 5
GLYPH_CLASS_OBJECT = 6
GLYPH_CLASS_DUNGEON = 7
GLYPH_CLASS_EFFECT = 8  # explode, zap, swallow, warning
GLYPH_CLASS_STATUE = 9
NUM_GLYPH_CLASSES = 10

# Entity property indices
EP_DANGER = 0        # 0=safe, 0.5=caution, 0.8=dangerous, 1.0=deadly
EP_CORPSE_GOOD = 1   # 1.0 if corpse gives beneficial intrinsic
EP_CORPSE_BAD = 2    # 1.0 if corpse is poisonous/petrifying
EP_MELEE_RISK = 3    # 1.0 if melee contact is dangerous (floating eye, cockatrice)
EP_ELBERETH_IMMUNE = 4  # 1.0 if ignores Elbereth
EP_LEVEL = 5         # normalized monster level (mlevel / 50)
EP_AC = 6            # normalized AC (ac / 20, lower=better armor)
EP_SPEED = 7         # normalized speed (mmove / 30)
EP_MR = 8            # magic resistance (mr / 100)
EP_DIFFICULTY = 9    # normalized difficulty rating
EP_IS_MONSTER = 10   # 1.0 if this glyph is a monster/pet/ridden
EP_IS_BODY = 11      # 1.0 if this is a corpse
EP_IS_OBJECT = 12    # 1.0 if this is an object
EP_IS_DUNGEON = 13   # 1.0 if this is dungeon terrain
N_ENTITY_FEATURES = 14

# Blstats indices (from NLE)
BL_X = 0
BL_Y = 1
BL_STR = 2
BL_HP = 10
BL_MAXHP = 11
BL_DEPTH = 12
BL_HUNGER = 18
BL_XL = 21
BL_ALIGNMENT = 22

# Monsters with special melee contact effects
MELEE_RISK_MONSTERS = {
    "floating eye",       # paralysis
    "cockatrice",         # petrification
    "chickatrice",        # petrification
    "green slime",        # sliming
    "yellow light",       # blindness explosion
    "black light",        # blindness explosion
    "mind flayer",        # intelligence drain
    "master mind flayer",  # intelligence drain
}

# Build name-to-id lookup once
_NAME_TO_ID: dict[str, int] = {}
for _i in range(NUM_MONSTERS):
    _m = nethack.permonst(_i)
    _NAME_TO_ID[_m.mname] = _i


def _name_to_monster_id(name: str) -> int | None:
    """Look up monster ID by name. Returns None if not found."""
    return _NAME_TO_ID.get(name)


# ============================================================
# Entity table
# ============================================================

def build_entity_table() -> np.ndarray:
    """Build entity property table for all glyphs.

    Returns:
        np.float32 array of shape (MAX_GLYPH, N_ENTITY_FEATURES).
        Most entries are zero (dungeon tiles, effects). Monster glyphs
        have populated properties from NLE's permonst data + wiki rules.
    """
    table = np.zeros((MAX_GLYPH, N_ENTITY_FEATURES), dtype=np.float32)

    # Populate monster properties for all 381 monsters
    for mon_id in range(NUM_MONSTERS):
        m = nethack.permonst(mon_id)
        name = m.mname

        props = np.zeros(N_ENTITY_FEATURES, dtype=np.float32)

        # Danger assessment
        if name in {n.lower() if isinstance(n, str) else n for n in EXTREMELY_DANGEROUS}:
            props[EP_DANGER] = 1.0
        elif m.difficulty > 20:
            props[EP_DANGER] = 0.8
        elif m.difficulty > 10:
            props[EP_DANGER] = 0.5
        else:
            props[EP_DANGER] = max(0.0, m.difficulty / 30.0)

        # Corpse properties
        if name in BENEFICIAL_CORPSES:
            props[EP_CORPSE_GOOD] = 1.0
        if name in POISONOUS_CORPSES or name in PETRIFYING_CORPSES:
            props[EP_CORPSE_BAD] = 1.0

        # Melee contact risk
        if name in MELEE_RISK_MONSTERS:
            props[EP_MELEE_RISK] = 1.0

        # Elbereth immunity
        if name in ELBERETH_IMMUNE:
            props[EP_ELBERETH_IMMUNE] = 1.0

        # Numeric stats from NLE (normalized)
        props[EP_LEVEL] = min(1.0, m.mlevel / 50.0)
        props[EP_AC] = min(1.0, max(0.0, (10 - m.ac) / 20.0))  # lower AC = better
        props[EP_SPEED] = min(1.0, m.mmove / 30.0)
        props[EP_MR] = m.mr / 100.0
        props[EP_DIFFICULTY] = min(1.0, m.difficulty / 40.0)
        props[EP_IS_MONSTER] = 1.0

        # Apply to all glyph variants for this monster
        glyph_mon = nethack.GLYPH_MON_OFF + mon_id
        glyph_pet = nethack.GLYPH_PET_OFF + mon_id
        glyph_ridden = nethack.GLYPH_RIDDEN_OFF + mon_id

        for g in (glyph_mon, glyph_pet, glyph_ridden):
            if g < MAX_GLYPH:
                table[g] = props

        # Body (corpse) glyph
        glyph_body = nethack.GLYPH_BODY_OFF + mon_id
        if glyph_body < MAX_GLYPH:
            body_props = props.copy()
            body_props[EP_IS_MONSTER] = 0.0
            body_props[EP_IS_BODY] = 1.0
            table[glyph_body] = body_props

    # Mark object glyphs
    obj_start = nethack.GLYPH_OBJ_OFF
    obj_end = nethack.GLYPH_CMAP_OFF
    for g in range(obj_start, min(obj_end, MAX_GLYPH)):
        table[g, EP_IS_OBJECT] = 1.0

    # Mark dungeon glyphs
    dun_start = nethack.GLYPH_CMAP_OFF
    for g in range(dun_start, MAX_GLYPH):
        table[g, EP_IS_DUNGEON] = 1.0

    return table


def build_glyph_class_table() -> np.ndarray:
    """Map each glyph to its class ID (0-9).

    Returns:
        np.int64 array of shape (MAX_GLYPH,).
    """
    classes = np.full(MAX_GLYPH, GLYPH_CLASS_DUNGEON, dtype=np.int64)

    # Monsters
    for g in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
        classes[g] = GLYPH_CLASS_MONSTER
    # Pets
    for g in range(nethack.GLYPH_PET_OFF, nethack.GLYPH_INVIS_OFF):
        classes[g] = GLYPH_CLASS_PET
    # Invisible
    classes[nethack.GLYPH_INVIS_OFF] = GLYPH_CLASS_INVIS
    # Detected
    for g in range(nethack.GLYPH_DETECT_OFF, nethack.GLYPH_BODY_OFF):
        classes[g] = GLYPH_CLASS_DETECT
    # Bodies
    for g in range(nethack.GLYPH_BODY_OFF, nethack.GLYPH_RIDDEN_OFF):
        classes[g] = GLYPH_CLASS_BODY
    # Ridden
    for g in range(nethack.GLYPH_RIDDEN_OFF, nethack.GLYPH_OBJ_OFF):
        classes[g] = GLYPH_CLASS_RIDDEN
    # Objects
    for g in range(nethack.GLYPH_OBJ_OFF, nethack.GLYPH_CMAP_OFF):
        classes[g] = GLYPH_CLASS_OBJECT
    # Statues (if offset exists)
    if hasattr(nethack, 'GLYPH_STATUE_OFF'):
        for g in range(nethack.GLYPH_STATUE_OFF, MAX_GLYPH):
            classes[g] = GLYPH_CLASS_STATUE

    return classes


# ============================================================
# Rule table
# ============================================================

@dataclass
class Rule:
    """A situational rule from wiki knowledge.

    condition: callable(blstats, visible_monster_ids) -> bool
    action_name: human-readable action recommendation
    category: rule category for grouping
    """
    rule_id: int
    condition: Callable
    action_name: str
    category: str
    description: str


def build_rule_table() -> list[Rule]:
    """Build situational rules from wiki knowledge.

    Each rule is a (condition, action) pair. The condition is a
    callable that takes (blstats, visible_monster_ids) and returns
    True if the rule applies.

    Returns list of Rule objects.
    """
    rules: list[Rule] = []
    rid = 0

    # --- Prayer rules ---
    def _pray_critical_hp(bl, _mons):
        return bl[BL_HP] <= bl[BL_MAXHP] // 7 and bl[BL_ALIGNMENT] >= 0

    rules.append(Rule(rid, _pray_critical_hp, "pray",
                       "prayer", "Pray when HP critically low and aligned"))
    rid += 1

    def _pray_starving(bl, _mons):
        return bl[BL_HUNGER] >= 4 and bl[BL_ALIGNMENT] >= 0

    rules.append(Rule(rid, _pray_starving, "pray",
                       "prayer", "Pray when fainting from hunger"))
    rid += 1

    # --- Food safety rules ---
    def _dont_eat_petrify(bl, mons):
        body_ids = {m - nethack.GLYPH_BODY_OFF for m in mons
                    if nethack.GLYPH_BODY_OFF <= m < nethack.GLYPH_RIDDEN_OFF}
        return any(_id_is_petrifying(i) for i in body_ids)

    rules.append(Rule(rid, _dont_eat_petrify, "avoid_eat",
                       "food", "Don't eat petrifying corpses"))
    rid += 1

    def _eat_beneficial(bl, mons):
        body_ids = {m - nethack.GLYPH_BODY_OFF for m in mons
                    if nethack.GLYPH_BODY_OFF <= m < nethack.GLYPH_RIDDEN_OFF}
        return any(_id_is_beneficial(i) for i in body_ids)

    rules.append(Rule(rid, _eat_beneficial, "eat_corpse",
                       "food", "Eat corpses that grant beneficial intrinsics"))
    rid += 1

    # --- Combat avoidance rules ---
    for mon_name in MELEE_RISK_MONSTERS:
        mon_id = _name_to_monster_id(mon_name)
        if mon_id is None:
            continue

        def _make_avoid_melee(mid, mname):
            def _check(bl, mons):
                return mid in mons or (mid + nethack.GLYPH_MON_OFF) in mons
            return _check

        rules.append(Rule(rid, _make_avoid_melee(mon_id, mon_name),
                           "avoid_melee",
                           "combat",
                           "Avoid melee with %s" % mon_name))
        rid += 1

    # --- Elbereth rules ---
    def _use_elbereth_when_surrounded(bl, mons):
        monster_count = sum(1 for m in mons
                           if nethack.GLYPH_MON_OFF <= m < nethack.GLYPH_PET_OFF)
        return monster_count >= 3 and bl[BL_HP] < bl[BL_MAXHP] * 0.4

    rules.append(Rule(rid, _use_elbereth_when_surrounded, "engrave_elbereth",
                       "defense", "Engrave Elbereth when surrounded and low HP"))
    rid += 1

    # --- Navigation rules ---
    def _dont_descend_low_hp(bl, _mons):
        return bl[BL_HP] < bl[BL_MAXHP] * 0.5

    rules.append(Rule(rid, _dont_descend_low_hp, "avoid_descend",
                       "navigation", "Don't descend when HP below 50%"))
    rid += 1

    def _dont_descend_underleveled(bl, _mons):
        return bl[BL_DEPTH] > bl[BL_XL] * 2

    rules.append(Rule(rid, _dont_descend_underleveled, "avoid_descend",
                       "navigation", "Don't descend when dungeon depth > 2x XL"))
    rid += 1

    # --- Danger assessment rules ---
    def _flee_deadly(bl, mons):
        for m in mons:
            if nethack.GLYPH_MON_OFF <= m < nethack.GLYPH_PET_OFF:
                mid = m - nethack.GLYPH_MON_OFF
                name = nethack.permonst(mid).mname
                if name in EXTREMELY_DANGEROUS and bl[BL_XL] < 10:
                    return True
        return False

    rules.append(Rule(rid, _flee_deadly, "flee",
                       "combat", "Flee from extremely dangerous monsters when low level"))
    rid += 1

    # --- Item management rules ---
    def _low_hp_heal(bl, _mons):
        return bl[BL_HP] < bl[BL_MAXHP] * 0.3

    rules.append(Rule(rid, _low_hp_heal, "use_healing",
                       "items", "Use healing item when HP below 30%"))
    rid += 1

    # --- Exploration rules ---
    def _explore_when_healthy(bl, _mons):
        return bl[BL_HP] > bl[BL_MAXHP] * 0.7 and bl[BL_HUNGER] < 3

    rules.append(Rule(rid, _explore_when_healthy, "explore",
                       "navigation", "Explore when healthy and not hungry"))
    rid += 1

    # --- Level-appropriate combat ---
    def _fight_when_strong(bl, mons):
        monster_count = sum(1 for m in mons
                           if nethack.GLYPH_MON_OFF <= m < nethack.GLYPH_PET_OFF)
        return monster_count > 0 and bl[BL_HP] > bl[BL_MAXHP] * 0.6

    rules.append(Rule(rid, _fight_when_strong, "fight",
                       "combat", "Engage monsters when HP above 60%"))
    rid += 1

    # Pad to round number for meta-controller alignment
    while len(rules) < 80:
        rules.append(Rule(rid, lambda bl, m: False, "noop",
                           "padding", "Unused rule slot"))
        rid += 1

    return rules[:80]


# ============================================================
# Helpers
# ============================================================

def _id_is_petrifying(mon_id: int) -> bool:
    if mon_id < 0 or mon_id >= NUM_MONSTERS:
        return False
    return nethack.permonst(mon_id).mname in PETRIFYING_CORPSES


def _id_is_beneficial(mon_id: int) -> bool:
    if mon_id < 0 or mon_id >= NUM_MONSTERS:
        return False
    return nethack.permonst(mon_id).mname in BENEFICIAL_CORPSES


def get_kb_stats() -> dict:
    """Return summary statistics about the KB for logging."""
    entity_table = build_entity_table()
    rules = build_rule_table()
    active_rules = sum(1 for r in rules if r.category != "padding")

    monster_entries = int((entity_table[:, EP_IS_MONSTER] > 0).sum())
    body_entries = int((entity_table[:, EP_IS_BODY] > 0).sum())
    dangerous = int((entity_table[:, EP_DANGER] >= 0.8).sum())
    melee_risk = int((entity_table[:, EP_MELEE_RISK] > 0).sum())
    beneficial_corpse = int((entity_table[:, EP_CORPSE_GOOD] > 0).sum())

    return {
        "total_glyphs": MAX_GLYPH,
        "monster_entries": monster_entries,
        "body_entries": body_entries,
        "dangerous_monsters": dangerous,
        "melee_risk_monsters": melee_risk,
        "beneficial_corpses": beneficial_corpse,
        "total_rules": len(rules),
        "active_rules": active_rules,
        "entity_features": N_ENTITY_FEATURES,
    }
