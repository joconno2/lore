"""Navigation and pathfinding for the NetHack expert system.

BFS pathfinding on the 21x79 NLE dungeon grid. Maintains per-level
tile maps, finds paths to stairs/altars/exploration frontiers, and
converts path steps to NLE compass actions.

Glyph classification uses NLE 3.6.7 offsets (NUMMONS=381, MAX_GLYPH=5976).
Works without NLE installed by defining constants directly.
"""
from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

# ============================================================
# NLE glyph offset constants (3.6.7)
# ============================================================

try:
    from nle import nethack as _nh
    GLYPH_MON_OFF = _nh.GLYPH_MON_OFF
    GLYPH_PET_OFF = _nh.GLYPH_PET_OFF
    GLYPH_INVIS_OFF = _nh.GLYPH_INVIS_OFF
    GLYPH_DETECT_OFF = _nh.GLYPH_DETECT_OFF
    GLYPH_BODY_OFF = _nh.GLYPH_BODY_OFF
    GLYPH_RIDDEN_OFF = _nh.GLYPH_RIDDEN_OFF
    GLYPH_OBJ_OFF = _nh.GLYPH_OBJ_OFF
    GLYPH_CMAP_OFF = _nh.GLYPH_CMAP_OFF
    MAX_GLYPH = _nh.MAX_GLYPH
except ImportError:
    # Standard NLE 3.6.7 values
    GLYPH_MON_OFF = 0
    GLYPH_PET_OFF = 381
    GLYPH_INVIS_OFF = 762
    GLYPH_DETECT_OFF = 763
    GLYPH_BODY_OFF = 1144
    GLYPH_RIDDEN_OFF = 1525
    GLYPH_OBJ_OFF = 1906
    GLYPH_CMAP_OFF = 5782
    MAX_GLYPH = 5976

NUMMONS = GLYPH_PET_OFF - GLYPH_MON_OFF  # 381

# ============================================================
# CMAP symbol indices (offsets from GLYPH_CMAP_OFF)
# These match nethack's display.h / drawing.c
# ============================================================

SS_STONE = 0
SS_VWALL = 1
SS_HWALL = 2
SS_TLCORN = 3
SS_TRCORN = 4
SS_BLCORN = 5
SS_BRCORN = 6
SS_CRWALL = 7
SS_TUWALL = 8
SS_TDWALL = 9
SS_TLWALL = 10
SS_TRWALL = 11
SS_NDOOR = 12       # no door (doorway)
SS_VODOOR = 13      # vertical open door
SS_HODOOR = 14      # horizontal open door
SS_VCDOOR = 15      # vertical closed door
SS_HCDOOR = 16      # horizontal closed door
SS_BARS = 17
SS_TREE = 18
SS_ROOM = 19
SS_DARKROOM = 20
SS_CORR = 21
SS_LITCORR = 22
SS_UPSTAIR = 23
SS_DNSTAIR = 24
SS_UPLADDER = 25
SS_DNLADDER = 26
SS_ALTAR = 27
SS_GRAVE = 28
SS_THRONE = 29
SS_SINK = 30
SS_FOUNTAIN = 31
SS_POOL = 32
SS_ICE = 33
SS_LAVA = 34
SS_VODBRIDGE = 35   # vertical open drawbridge
SS_HODBRIDGE = 36   # horizontal open drawbridge
SS_VCDBRIDGE = 37
SS_HCDBRIDGE = 38
SS_AIR = 39
SS_CLOUD = 40
SS_WATER = 41

# Number of dungeon feature symbols before traps start
_DUNGEON_FEATURES_END = 42

ROWS = 21
COLS = 79

# ============================================================
# Tile classification enum
# ============================================================

class Tile(IntEnum):
    UNEXPLORED = 0
    WALL = 1
    FLOOR = 2
    DOOR = 3
    CORRIDOR = 4
    STAIRS_DOWN = 5
    STAIRS_UP = 6
    ALTAR = 7
    FOUNTAIN = 8
    MONSTER = 9
    ITEM = 10
    POOL = 11
    LAVA = 12
    STONE = 13
    BARS = 14
    TREE = 15
    OTHER = 16


# ============================================================
# Glyph classification
# ============================================================

def _cmap(glyph: int) -> int:
    """Return CMAP index if glyph is a dungeon feature, else -1."""
    idx = glyph - GLYPH_CMAP_OFF
    if 0 <= idx < (MAX_GLYPH - GLYPH_CMAP_OFF):
        return idx
    return -1


def is_monster(glyph: int) -> bool:
    """True for monster, pet, ridden, and detected monster glyphs."""
    if GLYPH_MON_OFF <= glyph < GLYPH_PET_OFF:
        return True
    if GLYPH_PET_OFF <= glyph < GLYPH_INVIS_OFF:
        return True
    if glyph == GLYPH_INVIS_OFF:
        return True
    if GLYPH_DETECT_OFF <= glyph < GLYPH_BODY_OFF:
        return True
    if GLYPH_RIDDEN_OFF <= glyph < GLYPH_OBJ_OFF:
        return True
    return False


def is_object(glyph: int) -> bool:
    """True for object and body/corpse glyphs."""
    if GLYPH_OBJ_OFF <= glyph < GLYPH_CMAP_OFF:
        return True
    if GLYPH_BODY_OFF <= glyph < GLYPH_RIDDEN_OFF:
        return True
    return False


def is_wall(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_VWALL, SS_HWALL, SS_TLCORN, SS_TRCORN,
                 SS_BLCORN, SS_BRCORN, SS_CRWALL,
                 SS_TUWALL, SS_TDWALL, SS_TLWALL, SS_TRWALL)


def is_stone(glyph: int) -> bool:
    return _cmap(glyph) == SS_STONE


def is_floor(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_ROOM, SS_DARKROOM)


def is_corridor(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_CORR, SS_LITCORR)


def is_door(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_NDOOR, SS_VODOOR, SS_HODOOR, SS_VCDOOR, SS_HCDOOR)


def is_open_door(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_NDOOR, SS_VODOOR, SS_HODOOR)


def is_closed_door(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_VCDOOR, SS_HCDOOR)


def is_stairs_down(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_DNSTAIR, SS_DNLADDER)


def is_stairs_up(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_UPSTAIR, SS_UPLADDER)


def is_altar(glyph: int) -> bool:
    return _cmap(glyph) == SS_ALTAR


def is_fountain(glyph: int) -> bool:
    return _cmap(glyph) == SS_FOUNTAIN


def is_pool(glyph: int) -> bool:
    c = _cmap(glyph)
    return c in (SS_POOL, SS_WATER)


def is_lava(glyph: int) -> bool:
    return _cmap(glyph) == SS_LAVA


def is_walkable(glyph: int) -> bool:
    """True if the agent can walk on this glyph without special actions.

    Includes floor, corridor, open doors, stairs, altar, fountain,
    ice, open drawbridges, grave, throne, sink, air.
    Monsters standing on walkable tiles are also walkable (attack/swap).
    Objects on the ground are walkable.
    """
    if is_monster(glyph):
        return True
    if is_object(glyph):
        return True
    c = _cmap(glyph)
    if c < 0:
        return False
    return c in (
        SS_ROOM, SS_DARKROOM, SS_CORR, SS_LITCORR,
        SS_NDOOR, SS_VODOOR, SS_HODOOR,
        SS_UPSTAIR, SS_DNSTAIR, SS_UPLADDER, SS_DNLADDER,
        SS_ALTAR, SS_GRAVE, SS_THRONE, SS_SINK, SS_FOUNTAIN,
        SS_ICE, SS_VODBRIDGE, SS_HODBRIDGE, SS_AIR,
    )


def classify_glyph(glyph: int) -> Tile:
    """Classify a glyph into a Tile enum value."""
    if is_monster(glyph):
        return Tile.MONSTER
    if is_object(glyph):
        return Tile.ITEM
    c = _cmap(glyph)
    if c < 0:
        return Tile.OTHER
    if c == SS_STONE:
        return Tile.STONE
    if c in (SS_VWALL, SS_HWALL, SS_TLCORN, SS_TRCORN,
             SS_BLCORN, SS_BRCORN, SS_CRWALL,
             SS_TUWALL, SS_TDWALL, SS_TLWALL, SS_TRWALL):
        return Tile.WALL
    if c in (SS_ROOM, SS_DARKROOM):
        return Tile.FLOOR
    if c in (SS_CORR, SS_LITCORR):
        return Tile.CORRIDOR
    if c in (SS_NDOOR, SS_VODOOR, SS_HODOOR, SS_VCDOOR, SS_HCDOOR):
        return Tile.DOOR
    if c in (SS_DNSTAIR, SS_DNLADDER):
        return Tile.STAIRS_DOWN
    if c in (SS_UPSTAIR, SS_UPLADDER):
        return Tile.STAIRS_UP
    if c == SS_ALTAR:
        return Tile.ALTAR
    if c == SS_FOUNTAIN:
        return Tile.FOUNTAIN
    if c in (SS_POOL, SS_WATER):
        return Tile.POOL
    if c == SS_LAVA:
        return Tile.LAVA
    if c == SS_BARS:
        return Tile.BARS
    if c == SS_TREE:
        return Tile.TREE
    return Tile.OTHER


# ============================================================
# DungeonMap
# ============================================================

class DungeonMap:
    """Per-level map of explored tiles.

    Tracks tile types, exploration state, and positions of key features
    (stairs, altars). Updated each step from NLE glyph observations.
    """

    def __init__(self):
        self.tiles = np.full((ROWS, COLS), Tile.UNEXPLORED, dtype=np.int8)
        self.explored = np.zeros((ROWS, COLS), dtype=bool)
        self.stairs_down: list[tuple[int, int]] = []
        self.stairs_up: list[tuple[int, int]] = []
        self.altars: list[tuple[int, int]] = []
        self.fountains: list[tuple[int, int]] = []

    def update_from_glyphs(self, glyphs: np.ndarray,
                           player_pos: tuple[int, int]) -> None:
        """Update tile knowledge from NLE glyph observation.

        Args:
            glyphs: (21, 79) int array of NLE glyphs.
            player_pos: (row, col) of the player.
        """
        for r in range(ROWS):
            for c in range(COLS):
                g = int(glyphs[r, c])
                tile = classify_glyph(g)

                # NLE fills unseen tiles with SS_STONE. We can't
                # distinguish "real stone" from "never seen" in a
                # single observation, so only mark stone as explored
                # when it's adjacent to the player (definitely real).
                if tile == Tile.STONE:
                    if not self.explored[r, c]:
                        dr = abs(r - player_pos[0])
                        dc = abs(c - player_pos[1])
                        if dr <= 1 and dc <= 1:
                            self.tiles[r, c] = tile
                            self.explored[r, c] = True
                    continue

                # Wall glyphs are only rendered for tiles the player
                # can actually see, so they're always real.
                if tile != Tile.UNEXPLORED:
                    self.tiles[r, c] = tile
                    self.explored[r, c] = True

                    pos = (r, c)
                    if tile == Tile.STAIRS_DOWN and pos not in self.stairs_down:
                        self.stairs_down.append(pos)
                    elif tile == Tile.STAIRS_UP and pos not in self.stairs_up:
                        self.stairs_up.append(pos)
                    elif tile == Tile.ALTAR and pos not in self.altars:
                        self.altars.append(pos)
                    elif tile == Tile.FOUNTAIN and pos not in self.fountains:
                        self.fountains.append(pos)

        # Player position is always explored
        pr, pc = player_pos
        self.explored[pr, pc] = True

    def is_tile_walkable(self, r: int, c: int) -> bool:
        """Check if a tile is walkable based on stored tile type."""
        t = self.tiles[r, c]
        return t in (
            Tile.FLOOR, Tile.CORRIDOR, Tile.DOOR,
            Tile.STAIRS_DOWN, Tile.STAIRS_UP,
            Tile.ALTAR, Tile.FOUNTAIN,
            Tile.MONSTER, Tile.ITEM, Tile.OTHER,
        )

    def default_walkable(self, r: int, c: int) -> bool:
        """Default walkability check for BFS: explored + walkable tile."""
        if not self.explored[r, c]:
            return False
        return self.is_tile_walkable(r, c)


# ============================================================
# BFS pathfinder
# ============================================================

def find_path(start: tuple[int, int], goal: tuple[int, int],
              walkable_fn: Callable[[int, int], bool]) -> Optional[list[tuple[int, int]]]:
    """BFS shortest path from start to goal on the 21x79 grid.

    Args:
        start: (row, col)
        goal: (row, col)
        walkable_fn: (row, col) -> bool. Start and goal are always
                     treated as walkable.

    Returns:
        List of (row, col) positions from start to goal (inclusive),
        or None if no path exists.
    """
    if start == goal:
        return [start]

    sr, sc = start
    gr, gc = goal
    if not (0 <= gr < ROWS and 0 <= gc < COLS):
        return None

    visited = np.zeros((ROWS, COLS), dtype=bool)
    visited[sr, sc] = True
    parent = {}
    queue = deque()
    queue.append(start)

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if visited[nr, nc]:
                continue
            if (nr, nc) != goal and not walkable_fn(nr, nc):
                continue
            visited[nr, nc] = True
            parent[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                # Reconstruct path
                path = [(nr, nc)]
                pos = (nr, nc)
                while pos != start:
                    pos = parent[pos]
                    path.append(pos)
                path.reverse()
                return path
            queue.append((nr, nc))

    return None


def find_nearest(start: tuple[int, int],
                 predicate_fn: Callable[[int, int], bool],
                 walkable_fn: Callable[[int, int], bool]) -> Optional[tuple[int, int]]:
    """BFS to find the nearest tile matching predicate.

    Args:
        start: (row, col)
        predicate_fn: (row, col) -> bool, return True for target tiles.
        walkable_fn: (row, col) -> bool.

    Returns:
        (row, col) of nearest matching tile, or None.
    """
    sr, sc = start
    if predicate_fn(sr, sc):
        return start

    visited = np.zeros((ROWS, COLS), dtype=bool)
    visited[sr, sc] = True
    queue = deque()
    queue.append(start)

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if visited[nr, nc]:
                continue
            visited[nr, nc] = True
            if predicate_fn(nr, nc):
                return (nr, nc)
            if walkable_fn(nr, nc):
                queue.append((nr, nc))

    return None


def find_nearest_unexplored(start: tuple[int, int],
                            dungeon_map: DungeonMap) -> Optional[tuple[int, int]]:
    """Find nearest unexplored tile adjacent to an explored walkable tile.

    BFS from start through walkable explored tiles, looking for
    unexplored tiles on the frontier.
    """
    def pred(r, c):
        if dungeon_map.explored[r, c]:
            return False
        # Must be adjacent to at least one explored walkable tile
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ar, ac = r + dr, c + dc
            if 0 <= ar < ROWS and 0 <= ac < COLS:
                if dungeon_map.explored[ar, ac] and dungeon_map.is_tile_walkable(ar, ac):
                    return True
        return False

    return find_nearest(start, pred, dungeon_map.default_walkable)


def find_stairs_down(start: tuple[int, int],
                     dungeon_map: DungeonMap) -> Optional[tuple[int, int]]:
    """Find nearest known downstairs from start."""
    if not dungeon_map.stairs_down:
        return None

    def pred(r, c):
        return (r, c) in dungeon_map.stairs_down

    return find_nearest(start, pred, dungeon_map.default_walkable)


def find_stairs_up(start: tuple[int, int],
                   dungeon_map: DungeonMap) -> Optional[tuple[int, int]]:
    """Find nearest known upstairs from start."""
    if not dungeon_map.stairs_up:
        return None

    def pred(r, c):
        return (r, c) in dungeon_map.stairs_up

    return find_nearest(start, pred, dungeon_map.default_walkable)


# ============================================================
# Direction conversion
# ============================================================

# NLE CompassDirection enum values (from nle.nethack.actions):
#   N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7
# In the canonical ACTIONS tuple, compass directions are the first 8 entries.
# (dr, dc) -> action index
_DELTA_TO_ACTION: dict[tuple[int, int], int] = {
    (-1,  0): 0,   # N
    ( 0,  1): 1,   # E
    ( 1,  0): 2,   # S
    ( 0, -1): 3,   # W
    (-1,  1): 4,   # NE
    ( 1,  1): 5,   # SE
    ( 1, -1): 6,   # SW
    (-1, -1): 7,   # NW
}

# Reverse: action index -> (dr, dc)
_ACTION_TO_DELTA: dict[int, tuple[int, int]] = {v: k for k, v in _DELTA_TO_ACTION.items()}


def path_to_action(current_pos: tuple[int, int],
                   next_pos: tuple[int, int]) -> int:
    """Convert a single path step to an NLE compass action index.

    Args:
        current_pos: (row, col)
        next_pos: (row, col)

    Returns:
        Action index (0-7) for the compass direction.

    Raises:
        ValueError: if next_pos is not adjacent to current_pos.
    """
    dr = next_pos[0] - current_pos[0]
    dc = next_pos[1] - current_pos[1]
    action = _DELTA_TO_ACTION.get((dr, dc))
    if action is None:
        raise ValueError(
            f"Positions not adjacent: {current_pos} -> {next_pos} "
            f"(delta=({dr}, {dc}))"
        )
    return action


def path_to_actions(path: list[tuple[int, int]]) -> list[int]:
    """Convert a full path to a list of NLE compass action indices.

    Args:
        path: List of (row, col) positions (from find_path).

    Returns:
        List of action indices, length = len(path) - 1.
    """
    actions = []
    for i in range(len(path) - 1):
        actions.append(path_to_action(path[i], path[i + 1]))
    return actions


# ============================================================
# Exploration strategy
# ============================================================

def get_explore_target(player_pos: tuple[int, int],
                       dungeon_map: DungeonMap) -> Optional[tuple[int, int]]:
    """Pick the best tile to explore next.

    Priority order:
    1. Unexplored tiles adjacent to corridors/doors (likely lead somewhere).
    2. Unexplored tiles adjacent to any explored walkable tile.

    Returns (row, col) of target, or None if fully explored.
    """
    pr, pc = player_pos
    best = None
    best_dist = float('inf')
    best_priority = 0

    # BFS from player to rank candidates by distance
    visited = np.zeros((ROWS, COLS), dtype=bool)
    visited[pr, pc] = True
    dist = np.full((ROWS, COLS), -1, dtype=np.int32)
    dist[pr, pc] = 0
    queue = deque()
    queue.append((pr, pc))

    while queue:
        r, c = queue.popleft()
        d = dist[r, c]

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if visited[nr, nc]:
                continue
            visited[nr, nc] = True

            if not dungeon_map.explored[nr, nc]:
                # This is a frontier tile. Score it.
                priority = 0
                for adr, adc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ar, ac = nr + adr, nc + adc
                    if 0 <= ar < ROWS and 0 <= ac < COLS and dungeon_map.explored[ar, ac]:
                        t = dungeon_map.tiles[ar, ac]
                        if t in (Tile.CORRIDOR, Tile.DOOR):
                            priority = max(priority, 2)
                        elif dungeon_map.is_tile_walkable(ar, ac):
                            priority = max(priority, 1)

                if priority > 0:
                    if (priority > best_priority or
                            (priority == best_priority and d + 1 < best_dist)):
                        best = (nr, nc)
                        best_dist = d + 1
                        best_priority = priority
            elif dungeon_map.is_tile_walkable(nr, nc):
                dist[nr, nc] = d + 1
                queue.append((nr, nc))

    return best


# ============================================================
# Multi-level dungeon state
# ============================================================

class DungeonState:
    """Tracks DungeonMap instances for multiple dungeon levels.

    Levels are keyed by (dungeon_number, level_number).
    """

    def __init__(self):
        self.levels: dict[tuple[int, int], DungeonMap] = {}
        self.current_level_key: Optional[tuple[int, int]] = None

    def get_or_create(self, dungeon_number: int,
                      level_number: int) -> DungeonMap:
        key = (dungeon_number, level_number)
        if key not in self.levels:
            self.levels[key] = DungeonMap()
        return self.levels[key]

    def current_map(self) -> Optional[DungeonMap]:
        if self.current_level_key is None:
            return None
        return self.levels.get(self.current_level_key)

    def set_level(self, dungeon_number: int, level_number: int) -> DungeonMap:
        self.current_level_key = (dungeon_number, level_number)
        return self.get_or_create(dungeon_number, level_number)
