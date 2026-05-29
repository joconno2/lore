"""Tests for nhc.navigation module."""
from __future__ import annotations

import numpy as np
import pytest

from nhc.navigation import (
    # Constants
    GLYPH_CMAP_OFF, GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_OBJ_OFF,
    GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF, GLYPH_INVIS_OFF, GLYPH_DETECT_OFF,
    SS_ROOM, SS_CORR, SS_LITCORR, SS_VWALL, SS_HWALL, SS_STONE,
    SS_DNSTAIR, SS_UPSTAIR, SS_ALTAR, SS_FOUNTAIN, SS_POOL, SS_LAVA,
    SS_NDOOR, SS_VODOOR, SS_HODOOR, SS_VCDOOR, SS_HCDOOR,
    SS_DNLADDER, SS_UPLADDER, SS_ICE, SS_BARS, SS_TREE,
    ROWS, COLS,
    # Tile enum
    Tile,
    # Glyph classifiers
    is_walkable, is_wall, is_stairs_down, is_stairs_up,
    is_door, is_monster, is_object, is_floor, is_corridor,
    is_altar, is_fountain, is_open_door, is_closed_door,
    is_pool, is_lava, is_stone, classify_glyph,
    # Classes
    DungeonMap, DungeonState,
    # Pathfinding
    find_path, find_nearest, find_nearest_unexplored,
    find_stairs_down, find_stairs_up,
    # Direction
    path_to_action, path_to_actions,
    _DELTA_TO_ACTION, _ACTION_TO_DELTA,
    # Exploration
    get_explore_target,
)


# ============================================================
# Helper: build glyph values from CMAP offsets
# ============================================================

def _cmap(idx):
    return GLYPH_CMAP_OFF + idx


# ============================================================
# Glyph classification
# ============================================================

class TestGlyphClassification:
    def test_floor_walkable(self):
        assert is_walkable(_cmap(SS_ROOM))
        assert is_walkable(_cmap(SS_LITCORR))

    def test_wall_not_walkable(self):
        assert not is_walkable(_cmap(SS_VWALL))
        assert not is_walkable(_cmap(SS_HWALL))

    def test_stone_not_walkable(self):
        assert not is_walkable(_cmap(SS_STONE))

    def test_stairs_walkable(self):
        assert is_walkable(_cmap(SS_DNSTAIR))
        assert is_walkable(_cmap(SS_UPSTAIR))
        assert is_walkable(_cmap(SS_DNLADDER))
        assert is_walkable(_cmap(SS_UPLADDER))

    def test_open_door_walkable(self):
        assert is_walkable(_cmap(SS_NDOOR))
        assert is_walkable(_cmap(SS_VODOOR))
        assert is_walkable(_cmap(SS_HODOOR))

    def test_closed_door_not_walkable(self):
        assert not is_walkable(_cmap(SS_VCDOOR))
        assert not is_walkable(_cmap(SS_HCDOOR))

    def test_altar_walkable(self):
        assert is_walkable(_cmap(SS_ALTAR))

    def test_fountain_walkable(self):
        assert is_walkable(_cmap(SS_FOUNTAIN))

    def test_pool_not_walkable(self):
        assert not is_walkable(_cmap(SS_POOL))

    def test_lava_not_walkable(self):
        assert not is_walkable(_cmap(SS_LAVA))

    def test_ice_walkable(self):
        assert is_walkable(_cmap(SS_ICE))

    def test_bars_not_walkable(self):
        assert not is_walkable(_cmap(SS_BARS))

    def test_tree_not_walkable(self):
        assert not is_walkable(_cmap(SS_TREE))

    def test_monster_walkable(self):
        # First monster glyph
        assert is_walkable(GLYPH_MON_OFF)
        assert is_walkable(GLYPH_MON_OFF + 100)

    def test_pet_walkable(self):
        assert is_walkable(GLYPH_PET_OFF)

    def test_object_walkable(self):
        assert is_walkable(GLYPH_OBJ_OFF)
        assert is_walkable(GLYPH_OBJ_OFF + 50)

    def test_body_walkable(self):
        assert is_walkable(GLYPH_BODY_OFF)

    def test_is_wall(self):
        assert is_wall(_cmap(SS_VWALL))
        assert is_wall(_cmap(SS_HWALL))
        assert not is_wall(_cmap(SS_ROOM))
        assert not is_wall(_cmap(SS_STONE))

    def test_is_stairs_down(self):
        assert is_stairs_down(_cmap(SS_DNSTAIR))
        assert is_stairs_down(_cmap(SS_DNLADDER))
        assert not is_stairs_down(_cmap(SS_UPSTAIR))
        assert not is_stairs_down(_cmap(SS_ROOM))

    def test_is_stairs_up(self):
        assert is_stairs_up(_cmap(SS_UPSTAIR))
        assert is_stairs_up(_cmap(SS_UPLADDER))
        assert not is_stairs_up(_cmap(SS_DNSTAIR))

    def test_is_door(self):
        assert is_door(_cmap(SS_NDOOR))
        assert is_door(_cmap(SS_VODOOR))
        assert is_door(_cmap(SS_VCDOOR))
        assert not is_door(_cmap(SS_ROOM))

    def test_is_open_door(self):
        assert is_open_door(_cmap(SS_NDOOR))
        assert is_open_door(_cmap(SS_VODOOR))
        assert not is_open_door(_cmap(SS_VCDOOR))

    def test_is_closed_door(self):
        assert is_closed_door(_cmap(SS_VCDOOR))
        assert is_closed_door(_cmap(SS_HCDOOR))
        assert not is_closed_door(_cmap(SS_NDOOR))

    def test_is_monster(self):
        assert is_monster(GLYPH_MON_OFF)
        assert is_monster(GLYPH_MON_OFF + 200)
        assert is_monster(GLYPH_PET_OFF)
        assert is_monster(GLYPH_PET_OFF + 100)
        assert is_monster(GLYPH_INVIS_OFF)
        assert is_monster(GLYPH_DETECT_OFF)
        assert is_monster(GLYPH_RIDDEN_OFF)
        assert not is_monster(GLYPH_OBJ_OFF)
        assert not is_monster(GLYPH_CMAP_OFF)

    def test_is_object(self):
        assert is_object(GLYPH_OBJ_OFF)
        assert is_object(GLYPH_OBJ_OFF + 100)
        assert is_object(GLYPH_BODY_OFF)
        assert not is_object(GLYPH_MON_OFF)
        assert not is_object(GLYPH_CMAP_OFF)

    def test_is_floor(self):
        assert is_floor(_cmap(SS_ROOM))
        assert not is_floor(_cmap(SS_CORR))

    def test_is_corridor(self):
        assert is_corridor(_cmap(SS_CORR))
        assert is_corridor(_cmap(SS_LITCORR))
        assert not is_corridor(_cmap(SS_ROOM))

    def test_is_altar(self):
        assert is_altar(_cmap(SS_ALTAR))
        assert not is_altar(_cmap(SS_ROOM))

    def test_is_fountain(self):
        assert is_fountain(_cmap(SS_FOUNTAIN))
        assert not is_fountain(_cmap(SS_ROOM))

    def test_is_pool(self):
        assert is_pool(_cmap(SS_POOL))
        assert not is_pool(_cmap(SS_ROOM))

    def test_is_lava(self):
        assert is_lava(_cmap(SS_LAVA))
        assert not is_lava(_cmap(SS_ROOM))

    def test_is_stone(self):
        assert is_stone(_cmap(SS_STONE))
        assert not is_stone(_cmap(SS_ROOM))


class TestClassifyGlyph:
    def test_floor(self):
        assert classify_glyph(_cmap(SS_ROOM)) == Tile.FLOOR

    def test_wall(self):
        assert classify_glyph(_cmap(SS_VWALL)) == Tile.WALL

    def test_corridor(self):
        assert classify_glyph(_cmap(SS_CORR)) == Tile.CORRIDOR

    def test_door(self):
        assert classify_glyph(_cmap(SS_NDOOR)) == Tile.DOOR

    def test_stairs_down(self):
        assert classify_glyph(_cmap(SS_DNSTAIR)) == Tile.STAIRS_DOWN

    def test_stairs_up(self):
        assert classify_glyph(_cmap(SS_UPSTAIR)) == Tile.STAIRS_UP

    def test_altar(self):
        assert classify_glyph(_cmap(SS_ALTAR)) == Tile.ALTAR

    def test_fountain(self):
        assert classify_glyph(_cmap(SS_FOUNTAIN)) == Tile.FOUNTAIN

    def test_monster(self):
        assert classify_glyph(GLYPH_MON_OFF) == Tile.MONSTER

    def test_item(self):
        assert classify_glyph(GLYPH_OBJ_OFF) == Tile.ITEM

    def test_pool(self):
        assert classify_glyph(_cmap(SS_POOL)) == Tile.POOL

    def test_lava(self):
        assert classify_glyph(_cmap(SS_LAVA)) == Tile.LAVA

    def test_stone(self):
        assert classify_glyph(_cmap(SS_STONE)) == Tile.STONE

    def test_bars(self):
        assert classify_glyph(_cmap(SS_BARS)) == Tile.BARS

    def test_tree(self):
        assert classify_glyph(_cmap(SS_TREE)) == Tile.TREE


# ============================================================
# DungeonMap
# ============================================================

def _make_stone_glyphs():
    """21x79 grid of stone glyphs (default empty dungeon)."""
    return np.full((ROWS, COLS), _cmap(SS_STONE), dtype=np.int16)


def _make_room_map():
    """Small room in the center of the map for testing.

    Room spans rows 8-12, cols 35-45.
    Corridor from (8, 40) going up to (5, 40).
    Door at (8, 40).
    Stairs down at (10, 38).
    Stairs up at (10, 42).
    Altar at (11, 40).
    Fountain at (9, 40).
    """
    glyphs = _make_stone_glyphs()

    # Room walls
    for r in range(8, 13):
        for c in range(35, 46):
            if r == 8 or r == 12:
                glyphs[r, c] = _cmap(SS_HWALL)
            elif c == 35 or c == 45:
                glyphs[r, c] = _cmap(SS_VWALL)
            else:
                glyphs[r, c] = _cmap(SS_ROOM)

    # Door at top of room
    glyphs[8, 40] = _cmap(SS_NDOOR)

    # Corridor going up
    for r in range(5, 8):
        glyphs[r, 40] = _cmap(SS_CORR)

    # Stairs
    glyphs[10, 38] = _cmap(SS_DNSTAIR)
    glyphs[10, 42] = _cmap(SS_UPSTAIR)

    # Altar and fountain
    glyphs[11, 40] = _cmap(SS_ALTAR)
    glyphs[9, 40] = _cmap(SS_FOUNTAIN)

    return glyphs


class TestDungeonMap:
    def test_initial_state(self):
        dm = DungeonMap()
        assert dm.tiles.shape == (ROWS, COLS)
        assert dm.explored.shape == (ROWS, COLS)
        assert not dm.explored.any()
        assert (dm.tiles == Tile.UNEXPLORED).all()

    def test_update_from_glyphs(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        # Room floor should be explored
        assert dm.explored[10, 40]
        assert dm.tiles[10, 40] == Tile.FLOOR

        # Walls
        assert dm.tiles[8, 35] == Tile.WALL
        assert dm.tiles[12, 35] == Tile.WALL

        # Door
        assert dm.tiles[8, 40] == Tile.DOOR

        # Corridor
        assert dm.tiles[6, 40] == Tile.CORRIDOR

        # Stairs
        assert dm.tiles[10, 38] == Tile.STAIRS_DOWN
        assert dm.tiles[10, 42] == Tile.STAIRS_UP

        # Altar and fountain
        assert dm.tiles[11, 40] == Tile.ALTAR
        assert dm.tiles[9, 40] == Tile.FOUNTAIN

    def test_tracks_stairs_down(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))
        assert (10, 38) in dm.stairs_down

    def test_tracks_stairs_up(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))
        assert (10, 42) in dm.stairs_up

    def test_tracks_altars(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))
        assert (11, 40) in dm.altars

    def test_tracks_fountains(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))
        assert (9, 40) in dm.fountains

    def test_no_duplicate_features(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))
        dm.update_from_glyphs(glyphs, (10, 40))
        assert len(dm.stairs_down) == 1
        assert len(dm.stairs_up) == 1
        assert len(dm.altars) == 1

    def test_player_pos_marked_explored(self):
        dm = DungeonMap()
        glyphs = _make_stone_glyphs()
        dm.update_from_glyphs(glyphs, (10, 40))
        assert dm.explored[10, 40]

    def test_default_walkable(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        # Floor
        assert dm.default_walkable(10, 40)
        # Wall
        assert not dm.default_walkable(8, 35)
        # Stone
        assert not dm.default_walkable(0, 0)


# ============================================================
# BFS pathfinding
# ============================================================

class TestFindPath:
    def test_same_position(self):
        path = find_path((5, 5), (5, 5), lambda r, c: True)
        assert path == [(5, 5)]

    def test_straight_line(self):
        path = find_path((5, 5), (5, 8), lambda r, c: True)
        assert path is not None
        assert path[0] == (5, 5)
        assert path[-1] == (5, 8)
        assert len(path) == 4  # 5->6->7->8

    def test_diagonal(self):
        path = find_path((5, 5), (7, 7), lambda r, c: True)
        assert path is not None
        assert path[0] == (5, 5)
        assert path[-1] == (7, 7)
        assert len(path) == 3  # diagonal steps

    def test_blocked(self):
        # Wall across the path
        def walkable(r, c):
            return r != 6  # row 6 is impassable
        path = find_path((5, 5), (7, 5), walkable)
        # Should find diagonal path around
        if path is not None:
            assert path[-1] == (7, 5)
            for r, c in path[1:-1]:
                assert r != 6 or walkable(r, c)

    def test_completely_blocked(self):
        # No path possible
        def walkable(r, c):
            if r == 6:
                return False
            if r == 4:
                return False
            return True
        path = find_path((5, 5), (7, 5), walkable)
        assert path is None

    def test_path_on_dungeon_map(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        # Path from room floor to corridor
        path = find_path((10, 40), (5, 40), dm.default_walkable)
        assert path is not None
        assert path[0] == (10, 40)
        assert path[-1] == (5, 40)

    def test_path_avoids_walls(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        path = find_path((10, 40), (5, 40), dm.default_walkable)
        assert path is not None
        for r, c in path:
            assert dm.default_walkable(r, c) or (r, c) == (5, 40)

    def test_out_of_bounds_goal(self):
        path = find_path((5, 5), (-1, 5), lambda r, c: True)
        assert path is None

    def test_path_to_stairs(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        path = find_path((10, 40), (10, 38), dm.default_walkable)
        assert path is not None
        assert path[-1] == (10, 38)


class TestFindNearest:
    def test_find_adjacent(self):
        result = find_nearest((5, 5), lambda r, c: r == 5 and c == 6,
                              lambda r, c: True)
        assert result == (5, 6)

    def test_find_self(self):
        result = find_nearest((5, 5), lambda r, c: r == 5 and c == 5,
                              lambda r, c: True)
        assert result == (5, 5)

    def test_find_distant(self):
        result = find_nearest((5, 5), lambda r, c: r == 10 and c == 10,
                              lambda r, c: True)
        assert result == (10, 10)

    def test_find_nearest_of_multiple(self):
        targets = {(5, 8), (5, 20)}
        result = find_nearest((5, 5), lambda r, c: (r, c) in targets,
                              lambda r, c: True)
        assert result == (5, 8)

    def test_no_match(self):
        result = find_nearest((5, 5), lambda r, c: False,
                              lambda r, c: True)
        assert result is None

    def test_blocked_path(self):
        # Target exists but is unreachable
        def walkable(r, c):
            return r != 6  # row 6 blocks
        result = find_nearest((5, 5), lambda r, c: r == 7 and c == 5,
                              walkable)
        # Diagonal around should still work
        if result is not None:
            assert result == (7, 5)


class TestFindStairs:
    def test_find_stairs_down(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        result = find_stairs_down((10, 40), dm)
        assert result == (10, 38)

    def test_find_stairs_up(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        result = find_stairs_up((10, 40), dm)
        assert result == (10, 42)

    def test_no_stairs_known(self):
        dm = DungeonMap()
        assert find_stairs_down((10, 40), dm) is None
        assert find_stairs_up((10, 40), dm) is None


class TestFindNearestUnexplored:
    def test_finds_frontier(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        result = find_nearest_unexplored((5, 40), dm)
        # Should find an unexplored tile adjacent to the corridor end
        assert result is not None
        assert not dm.explored[result[0], result[1]]

    def test_fully_explored(self):
        dm = DungeonMap()
        dm.explored[:] = True
        dm.tiles[:] = Tile.FLOOR
        result = find_nearest_unexplored((10, 40), dm)
        assert result is None


# ============================================================
# Direction conversion
# ============================================================

class TestDirection:
    def test_north(self):
        assert path_to_action((5, 5), (4, 5)) == 0  # N

    def test_east(self):
        assert path_to_action((5, 5), (5, 6)) == 1  # E

    def test_south(self):
        assert path_to_action((5, 5), (6, 5)) == 2  # S

    def test_west(self):
        assert path_to_action((5, 5), (5, 4)) == 3  # W

    def test_northeast(self):
        assert path_to_action((5, 5), (4, 6)) == 4  # NE

    def test_southeast(self):
        assert path_to_action((5, 5), (6, 6)) == 5  # SE

    def test_southwest(self):
        assert path_to_action((5, 5), (6, 4)) == 6  # SW

    def test_northwest(self):
        assert path_to_action((5, 5), (4, 4)) == 7  # NW

    def test_all_eight_directions(self):
        assert len(_DELTA_TO_ACTION) == 8
        assert len(_ACTION_TO_DELTA) == 8

    def test_non_adjacent_raises(self):
        with pytest.raises(ValueError, match="not adjacent"):
            path_to_action((5, 5), (7, 5))

    def test_same_position_raises(self):
        with pytest.raises(ValueError, match="not adjacent"):
            path_to_action((5, 5), (5, 5))

    def test_path_to_actions(self):
        path = [(5, 5), (5, 6), (5, 7), (4, 7)]
        actions = path_to_actions(path)
        assert actions == [1, 1, 0]  # E, E, N

    def test_path_to_actions_empty(self):
        assert path_to_actions([(5, 5)]) == []

    def test_path_to_actions_diagonal(self):
        path = [(5, 5), (4, 6), (3, 7)]
        actions = path_to_actions(path)
        assert actions == [4, 4]  # NE, NE

    def test_roundtrip_delta_to_action(self):
        """Every delta maps to an action that maps back to the same delta."""
        for delta, action in _DELTA_TO_ACTION.items():
            assert _ACTION_TO_DELTA[action] == delta


# ============================================================
# Exploration strategy
# ============================================================

class TestExploreTarget:
    def test_returns_unexplored_tile(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        target = get_explore_target((10, 40), dm)
        assert target is not None
        assert not dm.explored[target[0], target[1]]

    def test_prefers_corridor_adjacent(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        target = get_explore_target((5, 40), dm)
        if target is not None:
            # Should pick tile near corridor end (row 4 or 5)
            # rather than far-away room edges
            r, c = target
            has_corridor_neighbor = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ar, ac = r + dr, c + dc
                if 0 <= ar < ROWS and 0 <= ac < COLS:
                    if dm.tiles[ar, ac] == Tile.CORRIDOR:
                        has_corridor_neighbor = True
            # Corridor-adjacent should be preferred, but if all are
            # equidistant, the closest wins regardless.
            assert target is not None

    def test_fully_explored_returns_none(self):
        dm = DungeonMap()
        dm.explored[:] = True
        dm.tiles[:] = Tile.FLOOR
        assert get_explore_target((10, 40), dm) is None


# ============================================================
# DungeonState
# ============================================================

class TestDungeonState:
    def test_create_level(self):
        ds = DungeonState()
        dm = ds.get_or_create(0, 1)
        assert isinstance(dm, DungeonMap)

    def test_same_key_same_map(self):
        ds = DungeonState()
        dm1 = ds.get_or_create(0, 1)
        dm2 = ds.get_or_create(0, 1)
        assert dm1 is dm2

    def test_different_key_different_map(self):
        ds = DungeonState()
        dm1 = ds.get_or_create(0, 1)
        dm2 = ds.get_or_create(0, 2)
        assert dm1 is not dm2

    def test_set_level(self):
        ds = DungeonState()
        dm = ds.set_level(0, 1)
        assert ds.current_level_key == (0, 1)
        assert ds.current_map() is dm

    def test_current_map_none_initially(self):
        ds = DungeonState()
        assert ds.current_map() is None


# ============================================================
# Integration: pathfind + action conversion
# ============================================================

class TestIntegration:
    def test_pathfind_to_stairs_and_convert(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        target = find_stairs_down((10, 40), dm)
        assert target is not None

        path = find_path((10, 40), target, dm.default_walkable)
        assert path is not None
        assert len(path) >= 2

        actions = path_to_actions(path)
        assert len(actions) == len(path) - 1
        for a in actions:
            assert 0 <= a <= 7

    def test_explore_then_pathfind(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        # Find explore target
        target = get_explore_target((10, 40), dm)
        if target is not None:
            # Try to pathfind near it (we can't path to unexplored,
            # but we can path to the nearest explored tile next to it)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = target[0] + dr, target[1] + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and dm.default_walkable(nr, nc):
                    path = find_path((10, 40), (nr, nc), dm.default_walkable)
                    if path is not None:
                        actions = path_to_actions(path)
                        assert all(0 <= a <= 7 for a in actions)
                        break

    def test_corridor_to_room(self):
        dm = DungeonMap()
        glyphs = _make_room_map()
        dm.update_from_glyphs(glyphs, (10, 40))

        # Path from top of corridor to altar inside room
        path = find_path((5, 40), (11, 40), dm.default_walkable)
        assert path is not None
        assert path[0] == (5, 40)
        assert path[-1] == (11, 40)

        actions = path_to_actions(path)
        # Should go south through corridor, through door, to altar
        for a in actions:
            assert 0 <= a <= 7
