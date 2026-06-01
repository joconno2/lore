"""Strategic decision layer: milestones, descent, and level objectives.

Based on AutoAscend's global_logic.py milestone system.
Provides high-level goals that drive the agent's behavior:
- When to descend
- What to do on each level
- Target XL/DL thresholds
"""
from __future__ import annotations
from enum import IntEnum, auto
from typing import Optional


class Milestone(IntEnum):
    """Game progression milestones."""
    FARM_DL1 = auto()          # Stay on DL1, farm to XL threshold
    FIND_EXCALIBUR = auto()    # Descend looking for fountains, dip at XL 7
    EXPLORE_AND_DESCEND = auto()  # Clear each level, descend when ready
    DEEP_PUSH = auto()         # Go as deep as possible, fight everything


class StrategyManager:
    """Manages game-level strategy and milestone progression."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.milestone = Milestone.FARM_DL1
        self.excalibur_acquired = False
        self.levels_explored = set()  # set of dlevel numbers fully explored

    def update(self, dlevel: int, xlevel: int, hp: int, max_hp: int,
               has_excalibur: bool, level_explored: bool,
               total_searches: int, has_food: bool):
        """Update milestone based on current state."""
        self.excalibur_acquired = has_excalibur

        if self.milestone == Milestone.FARM_DL1:
            # Stay on DL1 until XL 3 (compromise: AutoAscend uses 8, but we can't survive that long)
            # Also leave if we've searched too much (nothing left to do)
            if xlevel >= 3 or total_searches > 300:
                self.milestone = Milestone.FIND_EXCALIBUR

        elif self.milestone == Milestone.FIND_EXCALIBUR:
            if has_excalibur:
                self.milestone = Milestone.EXPLORE_AND_DESCEND
            # If we've been searching for Excalibur too long, just push
            if xlevel >= 10 or dlevel >= 8:
                self.milestone = Milestone.DEEP_PUSH

        elif self.milestone == Milestone.EXPLORE_AND_DESCEND:
            if dlevel >= 10:
                self.milestone = Milestone.DEEP_PUSH

    def should_descend(self, dlevel: int, xlevel: int, hp: int, max_hp: int,
                       level_explored: bool, total_searches: int,
                       visible_monsters: int) -> bool:
        """Decide whether to descend stairs."""
        # Always need decent HP
        if hp < max_hp * 0.6:
            return False

        if self.milestone == Milestone.FARM_DL1:
            # Farm: only descend when XL gate met
            if dlevel == 1:
                return xlevel >= 3 and level_explored
            return xlevel >= dlevel + 1

        elif self.milestone == Milestone.FIND_EXCALIBUR:
            # Descend freely to find fountains, but need XL >= DL
            if xlevel < dlevel:
                return False
            return level_explored or total_searches > 100

        elif self.milestone == Milestone.EXPLORE_AND_DESCEND:
            # Clear each level before descending
            if xlevel < dlevel:
                return False
            if visible_monsters > 0 and not level_explored:
                return False  # kill remaining monsters first
            return level_explored or total_searches > 150

        elif self.milestone == Milestone.DEEP_PUSH:
            # Just go deep
            return xlevel >= dlevel and (level_explored or total_searches > 80)

        return False

    def should_force_descend(self, dlevel: int, xlevel: int, hp: int, max_hp: int,
                             total_searches: int) -> bool:
        """Should we force-descend even below normal gates?"""
        if hp < max_hp * 0.5:
            return False

        if self.milestone == Milestone.FARM_DL1 and dlevel == 1:
            # Don't force-descend on DL1 during farming
            return total_searches > 400

        # On other levels, force after exhausting searches
        search_limit = {
            Milestone.FARM_DL1: 300,
            Milestone.FIND_EXCALIBUR: 150,
            Milestone.EXPLORE_AND_DESCEND: 200,
            Milestone.DEEP_PUSH: 80,
        }.get(self.milestone, 150)

        return total_searches > search_limit

    @property
    def name(self) -> str:
        return self.milestone.name
