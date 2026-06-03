"""Sokoban puzzle solver using pre-computed solutions.

Matches the current level layout to known Sokoban maps, then executes
the pre-computed boulder push sequence. Based on AutoAscend's soko_solver.

Sokoban in NetHack 3.6.7:
- 4 levels, entered from DL5-9 main dungeon
- dungeon_number == 4 in NLE
- Each level has boulders that must be pushed onto holes/traps
- Prize on top level (bag of holding or amulet of reflection)
"""
import numpy as np
from .soko_maps import maps

IGNORE = 0
EMPTY = 1
WALL = 2
BOULDER = 3
TARGET = 4

# Parse all known maps at import time
_PARSED_MAPS = {}

def _parse_map(text):
    """Convert ASCII map text to numpy array + start position."""
    mapping = {'<': EMPTY, '>': EMPTY, '.': EMPTY, '?': EMPTY, '+': EMPTY,
               '0': BOULDER, '-': WALL, '|': WALL, ' ': IGNORE, '^': TARGET}
    rows = []
    for line in text.splitlines():
        if not line:
            continue
        rows.append([mapping.get(c, IGNORE) for c in line])
    # Pad to uniform width
    max_w = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_w:
            r.append(IGNORE)
    return np.array(rows, dtype=np.int8)

for _text, _solution in maps.items():
    _parsed = _parse_map(_text)
    _PARSED_MAPS[id(_parsed)] = (_parsed, _solution)

# Store as list for iteration
SOKOBAN_MAPS = [(arr, sol) for arr, sol in _PARSED_MAPS.values()]
# Also keep raw text -> solution for matching
SOKOBAN_RAW = [((_parse_map(text), solution)) for text, solution in maps.items()]


def match_sokoban_level(wall_mask):
    """Match a wall mask from the game to a known Sokoban map.

    Args:
        wall_mask: boolean numpy array (MAP_H, MAP_W) where True = wall

    Returns:
        (solution, offset_y, offset_x) or None if no match.
        Solution is list of ((boulder_y, boulder_x), (push_dy, push_dx))
        in map-local coordinates. Add offset to get game coordinates.
    """
    for soko_map, solution in SOKOBAN_RAW:
        soko_walls = (soko_map == WALL)
        sh, sw = soko_walls.shape
        gh, gw = wall_mask.shape

        # Find walls in both
        soko_ys, soko_xs = soko_walls.nonzero()
        if len(soko_ys) == 0:
            continue

        game_ys, game_xs = wall_mask.nonzero()
        if len(game_ys) == 0:
            continue

        # Try to align by matching the top-left wall position
        soko_min_y, soko_min_x = soko_ys.min(), soko_xs.min()
        game_min_y, game_min_x = game_ys.min(), game_xs.min()

        off_y = game_min_y - soko_min_y
        off_x = game_min_x - soko_min_x

        # Check if the wall pattern matches at this offset
        match = True
        for sy, sx in zip(soko_ys, soko_xs):
            gy, gx = sy + off_y, sx + off_x
            if not (0 <= gy < gh and 0 <= gx < gw):
                match = False
                break
            if not wall_mask[gy, gx]:
                match = False
                break

        if match:
            return solution, off_y, off_x

    return None
