"""Equipment management: weapon DPS, armor AC, auto-equip.

Based on AutoAscend's character.py and inventory management.
Computes best weapon by DPS, best armor by AC per slot,
and provides equip actions.
"""
from __future__ import annotations
from typing import Optional

# Weapon damage data (name -> (to_hit_bonus, avg_damage))
# From NetHack 3.6.7 objects.c
WEAPON_DATA = {
    # Swords
    "long sword":       (0, 5.0),   # 1d8
    "two-handed sword": (0, 7.5),   # 1d12 (two-handed)
    "broadsword":       (0, 5.0),   # 2d4
    "short sword":      (0, 4.0),   # 1d6
    "scimitar":         (0, 5.0),   # 1d8
    "katana":           (1, 6.0),   # 1d10
    "silver saber":     (0, 5.0),   # 1d8
    # Axes
    "axe":              (0, 4.0),   # 1d6
    "battle-axe":       (0, 5.5),   # 1d8+1d4 (two-handed)
    # Maces/hammers
    "mace":             (0, 4.5),   # 1d6+1
    "war hammer":       (0, 4.5),   # 1d4+1 small / 1d4 large
    "morning star":     (0, 5.5),   # 2d4
    "flail":            (0, 4.5),   # 1d6+1
    # Polearms
    "spear":            (0, 4.0),   # 1d6
    "trident":          (0, 4.5),   # 1d6+1
    # Light weapons
    "dagger":           (0, 3.0),   # 1d4
    "knife":            (0, 2.0),   # 1d3
    "aklys":            (0, 4.0),   # 1d6
    # Artifacts
    "Excalibur":        (5, 10.5),  # 1d8 + 1d10 + to-hit 5
    "Mjollnir":         (5, 12.5),  # 1d4+1 + 1d24 shock
}

# Armor AC values (name -> AC bonus)
ARMOR_DATA = {
    # Body armor
    "plate mail":           7,
    "crystal plate mail":   7,
    "splint mail":          6,
    "banded mail":          6,
    "chain mail":           5,
    "scale mail":           4,
    "ring mail":            3,
    "studded leather armor": 3,
    "leather armor":        2,
    "leather jacket":       1,
    # Shields
    "large shield":         2,
    "small shield":         1,
    "shield of reflection": 2,
    # Helmets
    "helm of brilliance":   1,
    "helm of opposite alignment": 1,
    "helm of telepathy":    1,
    "dwarvish iron helm":   2,
    "orcish helm":          1,
    "elven leather helm":   1,
    "fedora":               0,
    "dunce cap":            0,
    # Cloaks
    "cloak of magic resistance": 1,
    "cloak of protection":  3,
    "cloak of displacement": 1,
    "cloak of invisibility": 1,
    "oilskin cloak":        1,
    "elven cloak":          1,
    "orcish cloak":         0,
    # Gloves
    "gauntlets of power":   1,
    "gauntlets of fumbling": 1,
    "gauntlets of dexterity": 1,
    "leather gloves":       1,
    # Boots
    "speed boots":          1,
    "water walking boots":  1,
    "jumping boots":        1,
    "elven boots":          1,
    "iron shoes":           2,
    "low boots":            1,
    "high boots":           2,
}

# Armor slot keywords
SLOT_KEYWORDS = {
    "body": ["mail", "armor", "jacket"],
    "shield": ["shield"],
    "helm": ["helm", "helmet", "hat", "cap", "fedora"],
    "cloak": ["cloak", "robe"],
    "gloves": ["gloves", "gauntlets"],
    "boots": ["boots", "shoes"],
}


class EquipmentManager:
    """Evaluates and manages equipment."""

    def find_best_weapon(self, inventory: dict) -> Optional[str]:
        """Find the best weapon to wield from inventory. Returns letter or None."""
        if not inventory:
            return None

        current_letter = None
        current_dps = 0.0

        best_letter = None
        best_dps = 0.0

        for letter, item_str in inventory.items():
            lower = item_str.lower()
            if "cursed" in lower:
                continue

            # Check if currently wielded
            is_wielded = "(weapon in hand)" in lower or "(wielded)" in lower
            if "(being worn)" in lower:
                continue

            # Find weapon match
            for wname, (to_hit, dmg) in WEAPON_DATA.items():
                if wname in lower:
                    # Enchantment bonus
                    enchant = 0
                    for prefix in ["+", "-"]:
                        idx = lower.find(prefix)
                        if idx != -1 and idx + 1 < len(lower):
                            try:
                                enchant = int(lower[idx:idx+2])
                            except ValueError:
                                pass

                    dps = dmg + enchant + to_hit * 0.5  # rough DPS estimate

                    if is_wielded:
                        current_letter = letter
                        current_dps = dps
                    elif dps > best_dps:
                        best_dps = dps
                        best_letter = letter
                    break

        # Only switch if significantly better
        if best_letter and best_dps > current_dps + 1.0:
            return best_letter
        return None

    def find_best_armor(self, inventory: dict) -> Optional[str]:
        """Find unworn armor that improves AC. Returns letter or None."""
        if not inventory:
            return None

        # Track worn slots
        worn_slots = set()
        for letter, item_str in inventory.items():
            lower = item_str.lower()
            if "(being worn)" in lower:
                for slot, keywords in SLOT_KEYWORDS.items():
                    if any(kw in lower for kw in keywords):
                        worn_slots.add(slot)
                        break

        # Find unworn armor for empty slots
        for letter, item_str in inventory.items():
            lower = item_str.lower()
            if "(being worn)" in lower or "cursed" in lower:
                continue
            for slot, keywords in SLOT_KEYWORDS.items():
                if any(kw in lower for kw in keywords) and slot not in worn_slots:
                    return letter

        return None
