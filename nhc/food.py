"""Food management system for NetHack expert agent.

Tracks corpse positions, freshness, and nutrition budget. Provides
actions for eating corpses on the ground, from inventory, and
navigating to known fresh corpses.

Based on AutoAscend's food system (global_logic.py:618-631, agent.py:1252-1379).

Corpse rules (NetHack 3.6.7):
- Corpses rot after 50 turns (except lizard, lichen)
- Not all monsters leave corpses (undead, gas spores, etc.)
- Some corpses are dangerous (poison, acid, petrification, etc.)
- Eating provides nutrition (50-800 depending on monster weight)
- Some corpses grant intrinsic resistances
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Monsters that don't leave corpses
NO_CORPSE = {
    "grid bug", "gas spore", "yellow light", "black light",
    "flaming sphere", "freezing sphere", "shocking sphere",
}

# Undead never leave corpses
UNDEAD_FRAGMENTS = ["zombie", "mummy", "skeleton", "wraith", "vampire", "ghost", "shade", "lich"]

# Dangerous corpses (from AutoAscend agent.py:1252-1317)
UNSAFE_CORPSES = {
    # Petrification
    "cockatrice", "chickatrice", "Medusa",
    # Polymorph
    "chameleon", "doppelganger", "sandestin",
    # Lycanthropy
    "wererat", "werejackal", "werewolf",
    # Stun/hallucination
    "bat", "giant bat", "yellow mold", "violet fungus",
    # Aggravate monster (pets)
    "little dog", "dog", "large dog",
    "kitten", "housecat", "large cat",
    # Paralysis
    "small mimic", "large mimic", "giant mimic",
    # Movement paralysis
    "floating eye",
    # Strips intrinsics
    "disenchanter",
    # Green slime
    "green slime",
    # Instakill
    "Death", "Pestilence", "Famine",
    # Acid damage
    "acid blob",
}

# Poisonous corpses (safe if you have poison resistance)
POISONOUS_CORPSES = {
    "killer bee", "scorpion", "pit viper", "cobra",
    "water moccasin", "asp", "python", "giant spider",
    "quasit", "rabid rat", "garter snake",
}

# Corpses that grant intrinsics
INTRINSIC_CORPSES = {
    "poison": ["killer bee", "scorpion", "pit viper", "cobra", "garter snake",
               "water moccasin", "asp", "python", "giant spider", "quasit"],
    "fire": ["red dragon", "red naga", "fire ant", "fire giant", "hell hound"],
    "cold": ["white dragon", "blue jelly", "winter wolf", "frost giant"],
    "shock": ["blue dragon", "electric eel", "storm giant"],
    "sleep": ["orange dragon", "elf"],
    "telepathy": ["floating eye", "mind flayer", "master mind flayer"],
}

# Maximum corpse age before it rots (turns)
MAX_CORPSE_AGE = 50
# Lizard and lichen never rot
NEVER_ROT = {"lizard", "lichen"}


@dataclass
class CorpseInfo:
    name: str
    row: int
    col: int
    turn_killed: int
    safe: bool = True
    intrinsic: Optional[str] = None


class FoodManager:
    """Tracks food state and provides eating decisions."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset for new episode."""
        self.corpses: list[CorpseInfo] = []
        self.nutrition_eaten: int = 0
        self.last_eat_turn: int = -100

    def on_level_change(self):
        """Clear corpse tracking on level change."""
        self.corpses = []

    def on_kill(self, monster_name: str, kill_row: int, kill_col: int,
                turn: int, resistances: set) -> Optional[CorpseInfo]:
        """Record a kill. Returns CorpseInfo if corpse is edible, None otherwise."""
        # Check if this monster leaves a corpse
        if monster_name in NO_CORPSE:
            return None
        if any(u in monster_name for u in UNDEAD_FRAGMENTS):
            return None

        # Check safety
        safe = monster_name not in UNSAFE_CORPSES
        if monster_name in POISONOUS_CORPSES and "poison resistance" not in resistances:
            safe = False

        # Check for intrinsic
        intrinsic = None
        for resist, monsters in INTRINSIC_CORPSES.items():
            if monster_name in monsters:
                intrinsic = resist
                break

        corpse = CorpseInfo(
            name=monster_name,
            row=kill_row, col=kill_col,
            turn_killed=turn,
            safe=safe,
            intrinsic=intrinsic,
        )
        self.corpses.append(corpse)
        return corpse if safe else None

    def get_fresh_corpses(self, current_turn: int) -> list[CorpseInfo]:
        """Get all fresh, safe corpses."""
        fresh = []
        for c in self.corpses:
            age = current_turn - c.turn_killed
            if c.name in NEVER_ROT or age <= MAX_CORPSE_AGE:
                if c.safe:
                    fresh.append(c)
        return fresh

    def expire_corpses(self, current_turn: int):
        """Remove rotted corpses."""
        self.corpses = [
            c for c in self.corpses
            if c.name in NEVER_ROT or (current_turn - c.turn_killed) <= MAX_CORPSE_AGE
        ]

    def nearest_corpse(self, py: int, px: int, current_turn: int,
                       bfs_dis: np.ndarray, max_distance: int = 8) -> Optional[CorpseInfo]:
        """Find nearest reachable fresh safe corpse."""
        best = None
        best_d = max_distance + 1
        for c in self.get_fresh_corpses(current_turn):
            d = bfs_dis[c.row, c.col]
            if 0 < d < best_d:
                best_d = d
                best = c
        return best

    def should_eat_inventory(self, hunger_state: str) -> bool:
        """Should we eat from inventory?"""
        return hunger_state in ("hungry", "weak", "fainting", "fainted")

    def should_seek_corpse(self, hunger_state: str) -> bool:
        """Should we walk to a corpse to eat?"""
        # Eat proactively, not just when hungry
        return hunger_state != "satiated"

    def is_corpse_safe(self, name: str, resistances: set) -> bool:
        """Check if a corpse is safe to eat."""
        if name in UNSAFE_CORPSES:
            return False
        if name in POISONOUS_CORPSES and "poison resistance" not in resistances:
            return False
        return True

    def remove_corpse_at(self, row: int, col: int):
        """Remove a corpse from tracking (eaten or disappeared)."""
        self.corpses = [c for c in self.corpses if not (c.row == row and c.col == col)]
