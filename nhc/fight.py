"""Combat decision system based on AutoAscend's fight_heur.py.

Provides fight/flee/elbereth decisions using a priority-based system
that considers monster danger, HP ratio, adjacent threat count,
and available weapons.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

# Monster categories (from AutoAscend combat/monster_utils.py)
NEVER_MELEE = {"floating eye", "gas spore"}
ONLY_RANGED_SLOW = {"floating eye", "blue jelly", "brown mold", "gas spore", "acid blob"}
EXPLODING = {"yellow light", "gas spore", "flaming sphere", "freezing sphere", "shocking sphere"}
WEAK = {"lichen", "newt", "shrieker", "grid bug"}
WEIRD = {"leprechaun", "nymph"}
INSTAKILL = {
    "cockatrice", "chickatrice", "Medusa",
    "green slime", "Death", "Pestilence", "Famine",
    "purple worm",
}
FAST_MONSTERS = {"bat", "giant bat", "dog", "large dog", "cat", "large cat",
                 "kitten", "pony", "horse", "bee", "fox", "coyote"}

# Peaceful monster IDs and names
PEACEFUL_IDS = {268, 269, 271, 272, 273}  # shopkeeper, guard, Oracle, priests
PEACEFUL_NAMES = {"shopkeeper", "aligned priest", "high priest",
                  "guard", "Oracle", "watchman", "watch captain"}


@dataclass
class FightDecision:
    action: str  # "melee", "flee", "elbereth", "wait", "approach", "avoid"
    target_dy: int = 0
    target_dx: int = 0
    priority: float = 0.0
    reason: str = ""


def assess_monster(name: str, mon_id: int = -1) -> dict:
    """Assess a monster's threat level and recommended action."""
    info = {
        "danger": 5,
        "never_melee": name in NEVER_MELEE,
        "instakill": name in INSTAKILL,
        "weak": name in WEAK,
        "weird": name in WEIRD,
        "fast": name in FAST_MONSTERS,
        "exploding": name in EXPLODING,
        "peaceful": name in PEACEFUL_NAMES or mon_id in PEACEFUL_IDS,
    }

    if info["instakill"]:
        info["danger"] = 10
    elif info["never_melee"]:
        info["danger"] = 8
    elif info["exploding"]:
        info["danger"] = 7
    elif info["weird"]:
        info["danger"] = 6
    elif info["weak"]:
        info["danger"] = 1

    return info


def should_elbereth(hp: int, max_hp: int, adjacent_monsters: list,
                    on_elbereth: bool, elbereth_cooldown: int) -> Optional[FightDecision]:
    """Decide if we should write Elbereth.

    Based on AutoAscend fight_heur.py:201-226.
    """
    if on_elbereth:
        # Wait on Elbereth if HP low
        hp_ratio = hp / max(max_hp, 1)
        if hp_ratio < 0.8:
            return FightDecision("wait", priority=30 - 40 * hp_ratio, reason="resting on Elbereth")
        return None  # healed up, move off

    if elbereth_cooldown > 0:
        return None

    if not adjacent_monsters:
        return None

    # Calculate weighted threat
    adj_weight = 0.0
    for mon in adjacent_monsters:
        info = assess_monster(mon.name, getattr(mon, 'mon_id', -1))
        if info["peaceful"]:
            continue
        hp_mult = min(20.0 / max(hp, 1), 2.0)
        if info["weak"]:
            adj_weight += 0.2 * hp_mult
        elif info["danger"] >= 7:
            adj_weight += 3.0 * hp_mult
        else:
            adj_weight += 1.0 * hp_mult
        if info["fast"]:
            adj_weight *= 1.5

    hp_ratio = (hp / max(max_hp, 1)) ** 0.5
    priority = -5 + 20 * adj_weight * (1 - hp_ratio)

    if priority > 0:
        return FightDecision("elbereth", priority=priority,
                             reason=f"threat={adj_weight:.1f} hp={hp}/{max_hp}")
    return None


def pick_melee_target(py: int, px: int, adjacent_monsters: list,
                      refused_attacks: set, refused_positions: set) -> Optional[FightDecision]:
    """Pick best adjacent monster to melee.

    Filters peacefuls, pets, never-melee. Prioritizes by danger level.
    Based on AutoAscend fight_heur.py:15-47, 237-249.
    """
    targets = []
    for mon in adjacent_monsters:
        if mon.is_pet:
            continue
        info = assess_monster(mon.name, getattr(mon, 'mon_id', -1))
        if info["peaceful"]:
            continue
        if mon.name.lower() in refused_attacks:
            continue
        if (mon.row, mon.col) in refused_positions:
            continue
        if info["never_melee"]:
            continue
        if info["instakill"]:
            continue  # handled by flee/elbereth

        priority = info["danger"]
        if info["weak"]:
            priority = 1
        if info["weird"]:
            continue  # avoid melee with nymphs/leprechauns

        targets.append((mon, priority))

    if not targets:
        return None

    # Sort by priority descending
    targets.sort(key=lambda x: -x[1])
    best_mon = targets[0][0]
    dy = best_mon.row - py
    dx = best_mon.col - px

    return FightDecision("melee", target_dy=dy, target_dx=dx,
                         priority=targets[0][1],
                         reason=f"melee {best_mon.name}")


def should_flee(hp: int, max_hp: int, adjacent_monsters: list) -> bool:
    """Check if we should flee based on HP."""
    if hp <= 5 or hp < max_hp * 0.15:
        return True
    # Check for instakill threats
    for mon in adjacent_monsters:
        if not mon.is_pet:
            info = assess_monster(mon.name, getattr(mon, 'mon_id', -1))
            if info["instakill"]:
                return True
    return False
