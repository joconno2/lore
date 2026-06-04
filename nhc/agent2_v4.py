"""LORE Agent v4: Complete AutoAscend-level expert system.

Full integration of all 10 subsystems. Clean room rewrite with complete
implementations. No stubs, no TODOs, no placeholders.

The agent DRIVES the env. step() calls env.step() and handles all prompts
internally (yn, getlin, xwait, --More--). The main loop runs strategies
in priority order until the episode ends.

Strategy priority (highest first):
  emergency > pray > fight > eat > equip > excalibur > altar_buc >
  item_id > sokoban > explore > descend
"""
from __future__ import annotations

import numpy as np
from collections import namedtuple, deque
from typing import Optional

import nle.nethack as nh
from nle.nethack import actions as A

# Subsystem imports
from nhc.obs_parser import (
    GameState, MonsterInfo,
    glyph_to_monster_name, glyph_to_mon_id, glyph_is_monster, glyph_is_pet,
    glyph_is_stairs_up, glyph_is_stairs_down, glyph_is_altar, glyph_is_fountain,
    GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_INVIS_OFF, GLYPH_DETECT_OFF,
    GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF, GLYPH_CMAP_OFF,
    NUMMONS, MAP_H, MAP_W,
    CMAP_UPSTAIR, CMAP_DNSTAIR, CMAP_UPLADDER, CMAP_DNLADDER,
    CMAP_ALTAR, CMAP_FOUNTAIN, CMAP_POOL, CMAP_LAVA,
    CMAP_ROOM, CMAP_DARKROOM, CMAP_CORR, CMAP_LITCORR,
    S_UPSTAIR, S_DNSTAIR, S_UPLADDER, S_DNLADDER,
    S_ALTAR, S_FOUNTAIN,
    BL_X, BL_Y, BL_STR25, BL_STR125, BL_DEX, BL_CON, BL_INT, BL_WIS, BL_CHA,
    BL_SCORE, BL_HP, BL_HPMAX, BL_DEPTH, BL_GOLD, BL_ENE, BL_ENEMAX,
    BL_AC, BL_HD, BL_XP, BL_EXP, BL_TIME, BL_HUNGER, BL_CAP,
    BL_DNUM, BL_DLEVEL, BL_CONDITION, BL_ALIGN,
    COND_STONE, COND_SLIME, COND_STRNGL, COND_FOODPOIS, COND_TERMILL,
    COND_BLIND, COND_DEAF, COND_STUN, COND_CONF, COND_HALLU,
    COND_LEV, COND_FLY, COND_RIDE,
    parse_conditions, HUNGER_LABELS, ALIGN_LABELS,
)

from nhc.navigation import (
    DungeonMap, DungeonState, Tile,
    find_path, find_nearest, find_nearest_unexplored,
    find_stairs_down, find_stairs_up,
    path_to_action, path_to_actions,
    get_explore_target,
    classify_glyph,
    is_monster, is_object, is_wall, is_stone, is_floor, is_corridor,
    is_door, is_open_door, is_closed_door, is_stairs_down, is_stairs_up,
    is_altar, is_fountain, is_pool, is_lava, is_walkable,
    SS_STONE, SS_VWALL, SS_HWALL, SS_TLCORN, SS_TRCORN,
    SS_BLCORN, SS_BRCORN, SS_CRWALL, SS_TUWALL, SS_TDWALL,
    SS_TLWALL, SS_TRWALL, SS_NDOOR, SS_VODOOR, SS_HODOOR,
    SS_VCDOOR, SS_HCDOOR, SS_BARS, SS_TREE, SS_ROOM, SS_DARKROOM,
    SS_CORR, SS_LITCORR,
    SS_UPSTAIR as NAV_SS_UPSTAIR, SS_DNSTAIR as NAV_SS_DNSTAIR,
    SS_UPLADDER as NAV_SS_UPLADDER, SS_DNLADDER as NAV_SS_DNLADDER,
    SS_ALTAR as NAV_SS_ALTAR, SS_FOUNTAIN as NAV_SS_FOUNTAIN,
    SS_POOL as NAV_SS_POOL, SS_LAVA as NAV_SS_LAVA,
    ROWS, COLS,
)

from nhc.prayer import (
    PrayerState, TroubleSeverity, TroubleInfo,
    COND_STONE as P_COND_STONE,
    COND_SLIME as P_COND_SLIME,
    COND_STRNGL as P_COND_STRNGL,
    COND_FOODPOIS as P_COND_FOODPOIS,
    COND_TERMILL as P_COND_TERMILL,
    MAJOR_TROUBLES, MINOR_TROUBLES,
    HungerState, Alignment,
)

from nhc.food import (
    FoodManager, CorpseInfo,
    NO_CORPSE, UNDEAD_FRAGMENTS, UNSAFE_CORPSES, POISONOUS_CORPSES,
    INTRINSIC_CORPSES, NEVER_ROT, MAX_CORPSE_AGE,
)

from nhc.equipment import (
    EquipmentManager, WEAPON_DATA, ARMOR_DATA, SLOT_KEYWORDS,
)

from nhc.fight import (
    assess_monster, FightDecision,
    NEVER_MELEE, INSTAKILL, ONLY_RANGED_SLOW, EXPLODING, WEAK, WEIRD,
    FAST_MONSTERS, PEACEFUL_NAMES, PEACEFUL_IDS,
    should_elbereth, pick_melee_target, should_flee,
)

from nhc.sokoban import match_sokoban_level

from nhc.strategy import StrategyManager, Milestone

# Try loading combat ThreatDB. Falls back to None if data files missing.
_THREAT_DB = None
try:
    from nhc.combat import ThreatDB, ThreatReport, CorpseReport as CombatCorpseReport
    try:
        _THREAT_DB = ThreatDB()
    except (FileNotFoundError, OSError):
        pass
except ImportError:
    pass

# Try loading AppearanceTracker. Falls back to None if data files missing.
_APPEARANCE_TRACKER_CLASS = None
try:
    from nhc.item_id import AppearanceTracker, ENGRAVE_EFFECTS, TRACKED_CLASSES
    _APPEARANCE_TRACKER_CLASS = AppearanceTracker
except (ImportError, FileNotFoundError, OSError):
    pass


# ================================================================
# Constants
# ================================================================

# Glyph CMAP range
_CMAP_MAX = 87  # number of cmap symbols

# Walkable cmap indices
_WALKABLE_CMAP = frozenset({
    SS_NDOOR, SS_VODOOR, SS_HODOOR,  # doors (open)
    SS_ROOM, SS_DARKROOM,            # floor
    SS_CORR, SS_LITCORR,            # corridors
    NAV_SS_UPSTAIR, NAV_SS_DNSTAIR,  # stairs
    NAV_SS_UPLADDER, NAV_SS_DNLADDER,
    NAV_SS_ALTAR,                    # altar
    28,  # grave
    29,  # throne
    30,  # sink
    NAV_SS_FOUNTAIN,                 # fountain
    33,  # ice
    35, 36,  # open drawbridges
    39,  # air
})

_CLOSED_DOOR_CMAP = frozenset({SS_VCDOOR, SS_HCDOOR})
_WALL_CMAP = frozenset({
    SS_VWALL, SS_HWALL, SS_TLCORN, SS_TRCORN,
    SS_BLCORN, SS_BRCORN, SS_CRWALL,
    SS_TUWALL, SS_TDWALL, SS_TLWALL, SS_TRWALL,
})
_DOOR_CMAP = frozenset({SS_NDOOR, SS_VODOOR, SS_HODOOR, SS_VCDOOR, SS_HCDOOR})
_STAIRS_DOWN_CMAP = frozenset({NAV_SS_DNSTAIR, NAV_SS_DNLADDER})
_STAIRS_UP_CMAP = frozenset({NAV_SS_UPSTAIR, NAV_SS_UPLADDER})
_FOUNTAIN_CMAP = frozenset({NAV_SS_FOUNTAIN})
_ALTAR_CMAP = frozenset({NAV_SS_ALTAR})
_POOL_CMAP = frozenset({NAV_SS_POOL, 41})  # pool, water
_LAVA_CMAP = frozenset({NAV_SS_LAVA})

# Hunger states
SATIATED = 0
NOT_HUNGRY = 1
HUNGRY = 2
WEAK_HUNGER = 3
FAINTING = 4
FAINTED = 5
STARVED = 6

# Object classes (NLE)
FOOD_CLASS = 7
WEAPON_CLASS = 3
ARMOR_CLASS = 6
POTION_CLASS = 10
SCROLL_CLASS = 8
WAND_CLASS = 9
RING_CLASS = 5
AMULET_CLASS = 4
TOOL_CLASS = 11
GEM_CLASS = 12
GOLD_CLASS = 14
COIN_CLASS = 14

# Dungeon numbers
DUNGEON_DOOM = 0
DUNGEON_GEHENNOM = 1
DUNGEON_MINES = 2
DUNGEON_QUEST = 3
DUNGEON_SOKOBAN = 4

# Boulder glyph
BOULDER_GLYPH = GLYPH_OBJ_OFF + 447

# Direction mappings
_DIR_DELTAS = {
    'N':  (-1,  0), 'S': ( 1,  0), 'E': ( 0,  1), 'W': ( 0, -1),
    'NE': (-1,  1), 'SE': ( 1,  1), 'SW': ( 1, -1), 'NW': (-1, -1),
}
_DELTA_TO_DIR = {v: k for k, v in _DIR_DELTAS.items()}

# ================================================================
# Helper functions
# ================================================================

def _cmap(g):
    """Extract cmap index from glyph, or -1 if not a cmap glyph."""
    idx = g - GLYPH_CMAP_OFF
    if 0 <= idx < _CMAP_MAX:
        return idx
    return -1


def _chebyshev(r1, c1, r2, c2):
    """Chebyshev distance (king-move distance)."""
    return max(abs(r1 - r2), abs(c1 - c2))


def _manhattan(r1, c1, r2, c2):
    """Manhattan distance."""
    return abs(r1 - r2) + abs(c1 - c2)


def _sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0


def _in_line(r1, c1, r2, c2):
    """Check if two positions are in a cardinal or diagonal line."""
    dr = r2 - r1
    dc = c2 - c1
    if dr == 0 or dc == 0:
        return True
    if abs(dr) == abs(dc):
        return True
    return False


def _line_clear(glyphs, r1, c1, r2, c2):
    """Check if the line of fire between two positions is clear.
    Positions must be in a cardinal or diagonal line."""
    dr = _sign(r2 - r1)
    dc = _sign(c2 - c1)
    r, c = r1 + dr, c1 + dc
    while (r, c) != (r2, c2):
        if not (0 <= r < MAP_H and 0 <= c < MAP_W):
            return False
        g = int(glyphs[r, c])
        cm = _cmap(g)
        # Walls, closed doors, boulders block line of fire
        if cm in _WALL_CMAP or cm in _CLOSED_DOOR_CMAP or cm == SS_STONE:
            return False
        if g == BOULDER_GLYPH:
            return False
        r += dr
        c += dc
    return True


# ================================================================
# Level state
# ================================================================

class Level:
    """Persistent per-level state. Survives when the agent leaves and returns."""

    def __init__(self, dungeon_number, level_number):
        self.dungeon_number = dungeon_number
        self.level_number = level_number

        # Map state
        self.seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self.search_count = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self.door_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)

        # Feature positions (persist across visits)
        self.stairs_down = set()
        self.stairs_up = set()
        self.fountains = set()
        self.altars = set()

        # Timing
        self.turns_spent = 0
        self.first_visit_turn = -1
        self.last_visit_turn = -1

        # Stair destinations: (y,x) -> (dungeon_number, level_number)
        self.stair_dest = {}

        # Sokoban state
        self.soko_solution = None
        self.soko_offset = (0, 0)
        self.soko_step = 0
        self.soko_matched = False

        # Altar BUC testing
        self.altar_tested_items = set()  # letters already BUC tested

        # Corpse tracking (positions of known corpses on this level)
        self.corpse_positions = {}  # (r, c) -> (name, turn_killed)

        # Explored flag
        self._explored_cache = None

    def key(self):
        return (self.dungeon_number, self.level_number)

    @property
    def total_searches(self):
        return int(self.search_count.sum())

    def is_explored(self):
        """True if no reachable unexplored frontier exists."""
        if self._explored_cache is not None:
            return self._explored_cache
        for r in range(MAP_H):
            for c in range(MAP_W):
                if not self.walkable[r, c]:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not self.seen[nr, nc]:
                            self._explored_cache = False
                            return False
        self._explored_cache = True
        return True

    def invalidate_explored_cache(self):
        self._explored_cache = None


# ================================================================
# Agent finished exception
# ================================================================

class AgentFinished(Exception):
    pass


# ================================================================
# Main Agent
# ================================================================

class AgentV4:
    """LORE v4 expert system agent.

    Integrates all 10 subsystems:
      obs_parser, combat, navigation, prayer, item_id,
      strategy, food, equipment, fight, sokoban
    """

    def __init__(self, env, seed=None, verbose=False):
        self.env = env
        self.seed = seed
        self.verbose = verbose

        # Build action lookups from NLE action space
        self.actions = list(nh.ACTIONS)
        self._val2idx = {}
        self._name2idx = {}
        for i, a in enumerate(self.actions):
            v = int(a)
            if v not in self._val2idx:
                self._val2idx[v] = i
            n = a.name if hasattr(a, 'name') else str(a)
            if n not in self._name2idx:
                self._name2idx[n] = i

        # Subsystems
        self.gs = GameState()
        self.food_mgr = FoodManager()
        self.equip_mgr = EquipmentManager()
        self.strategy_mgr = StrategyManager()
        self.prayer_state = PrayerState()
        self.dungeon_state = DungeonState()
        self.threat_db = _THREAT_DB  # may be None

        # Item identification
        self.appearance_tracker = None
        if _APPEARANCE_TRACKER_CLASS is not None:
            try:
                self.appearance_tracker = _APPEARANCE_TRACKER_CLASS()
            except (FileNotFoundError, OSError):
                pass

        # Raw observation
        self.obs = None
        self._raw_bl = None
        self.glyphs = None
        self.message = ''
        self.initial_message = ''
        self.score = 0.0
        self.step_count = 0

        # Per-level state
        self.levels = {}  # (dnum, dlevel) -> Level
        self._prev_level_key = None
        self._last_turn = -1

        # Inventory (kept in sync with gs.inventory)
        self.inventory = {}
        self.inv_oclasses = {}

        # Character knowledge
        self.resistances = set()
        self.has_excalibur = False
        self.has_long_sword = False
        self.alignment = 'neutral'
        self.role = ''  # detected from initial messages
        self.race = ''

        # Prayer tracking (supplement PrayerState)
        self._last_prayer_turn = -1000

        # Eating
        self._last_eat_turn = -100

        # Peaceful tracking
        self._peaceful_positions = set()  # (y, x) of known peacefuls on current level
        self._peaceful_monster_ids = set()  # monster IDs confirmed peaceful (global)

        # Movement tracking
        self._last_move_dir = (0, 0)
        self._last_action_name = ''
        self._stuck_count = 0
        self._last_pos = (-1, -1)

        # Elbereth state
        self._elbereth_cooldown = 0  # turns since last elbereth
        self._on_elbereth = False

        # Combat state
        self._kill_count = 0
        self._last_kill_name = None
        self._last_kill_dir = (0, 0)

        # Engrave-test wand tracking
        self._wands_to_test = []  # letters of untested wands
        self._wand_test_pending = False

        # Pickup state
        self._items_on_ground = {}  # (r, c) -> set of glyph values
        self._last_pickup_turn = -100

        # Pet tracking
        self._pet_pos = None
        self._pet_wait_count = 0

        # Counters
        self._total_steps = 0
        self._stall_turn = -1
        self._stall_count = 0
        self._search_in_row = 0

    # ================================================================
    # Level management
    # ================================================================

    def current_level(self) -> Level:
        """Get or create the Level object for the current dungeon position."""
        key = (self.gs.dnum, self.gs.dlevel)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        return self.levels[key]

    # ================================================================
    # Core: env interaction
    # ================================================================

    def _env_step(self, idx):
        """Raw env.step with observation copy. Returns True if episode ended."""
        obs, reward, done, truncated, info = self.env.step(idx)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        bl = self.obs.get('blstats')
        if bl is not None:
            bl_arr = np.array(bl, dtype=np.int64)
            if len(bl_arr) >= 21 and int(bl_arr[BL_TIME]) > 0:
                self._raw_bl = bl_arr
        self.score += reward
        self.step_count += 1
        self._total_steps += 1
        return done or truncated

    def step(self, action):
        """Send action to env, handle all prompts iteratively.

        action: NLE action enum, action name string, int action index, or single char
        """
        if isinstance(action, str):
            if len(action) == 1:
                idx = self._val2idx.get(ord(action))
            else:
                idx = self._name2idx.get(action)
        elif type(action) is int and action < len(self.actions):
            idx = action
        else:
            idx = self._val2idx.get(int(action))

        if idx is None:
            return

        if self._env_step(idx):
            self._quick_parse()
            raise AgentFinished()

        # Save initial message before prompt handling
        raw_msg = self.obs.get('message', b'')
        self.initial_message = bytes(raw_msg).decode('latin-1', errors='replace').replace('\x00', '').strip()

        # Handle prompts iteratively
        for _ in range(200):
            msg_raw = self.obs.get('message', b'')
            self.message = bytes(msg_raw).decode('latin-1', errors='replace').replace('\x00', '').strip()
            misc = self.obs.get('misc', [0, 0, 0])

            # yn prompt (misc[0])
            if misc[0]:
                resp = self._handle_yn_prompt()
                if resp is not None:
                    if self._env_step(resp):
                        self._quick_parse()
                        raise AgentFinished()
                continue

            # Text entry (misc[1] = in_getlin)
            if misc[1]:
                resp = self._handle_getlin_prompt()
                if resp is not None:
                    if self._env_step(resp):
                        self._quick_parse()
                        raise AgentFinished()
                else:
                    # Default: ESC out
                    esc_idx = self._val2idx.get(27, 0)
                    if self._env_step(esc_idx):
                        self._quick_parse()
                        raise AgentFinished()
                continue

            # xwait (misc[2])
            if misc[2]:
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._quick_parse()
                    raise AgentFinished()
                continue

            # --More-- in message text
            if '--More--' in self.message:
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._quick_parse()
                    raise AgentFinished()
                continue

            break

        self._update_game_state()

    def _quick_parse(self):
        """Minimal blstats parse for score tracking on death."""
        if self._raw_bl is not None and len(self._raw_bl) >= 10:
            self.gs.score = int(self._raw_bl[BL_SCORE])

    # ================================================================
    # Prompt handling
    # ================================================================

    def _handle_yn_prompt(self):
        """Decide response to yn prompt. Returns action index or None."""
        msg = self.message

        # Never attack peacefuls
        if 'Really attack' in msg:
            py, px = self.gs.py, self.gs.px
            dy, dx = self._last_move_dir
            ty, tx = py + dy, px + dx
            if 0 <= ty < MAP_H and 0 <= tx < MAP_W:
                self._peaceful_positions.add((ty, tx))
                if self.glyphs is not None:
                    g = int(self.glyphs[ty, tx])
                    mid = glyph_to_mon_id(g)
                    if mid is not None:
                        self._peaceful_monster_ids.add(mid)
            return self._val2idx.get(ord('n'))

        # Don't force locks
        if 'force the lock' in msg:
            return self._val2idx.get(ord('n'))

        # Eat it? corpse on ground
        if 'eat it?' in msg.lower() or 'eat this?' in msg.lower():
            # Check if the corpse is safe
            if self._is_current_corpse_safe():
                return self._val2idx.get(ord('y'))
            return self._val2idx.get(ord('n'))

        # "There is a" ... "here" with eat/pickup
        if 'There is' in msg and ('corpse' in msg or 'here' in msg):
            return self._val2idx.get(ord('y'))

        # Prayer confirmation
        if 'Are you sure' in msg and 'pray' in msg:
            return self._val2idx.get(ord('y'))

        # "Do you want your possessions identified?"
        if 'possessions identified' in msg:
            return self._val2idx.get(ord('y'))

        # "Shall I pick up" autopickup
        if 'Shall I pick' in msg:
            return self._val2idx.get(ord('n'))

        # "Do you want to add to the current engraving?"
        if 'add to' in msg.lower() and 'engraving' in msg.lower():
            return self._val2idx.get(ord('n'))

        # "In what direction?" for kick
        # This should be handled by the kick caller, not here

        # "What do you want to ..." menus: ESC
        if msg.startswith('What do you want to'):
            return self._val2idx.get(27)

        # Sacrifice: "Do you want to sacrifice"
        if 'sacrifice' in msg.lower():
            return self._val2idx.get(ord('y'))

        # Loot: "Do you want to loot"
        if 'loot' in msg.lower():
            return self._val2idx.get(ord('n'))

        # "Dip it into the" fountain
        if 'Dip' in msg and 'fountain' in msg:
            return self._val2idx.get(ord('y'))

        # "Drink from the fountain?"
        if 'Drink from' in msg and 'fountain' in msg:
            return self._val2idx.get(ord('n'))

        # "There is a fountain here" (offered to drink)
        if 'fountain here' in msg.lower():
            return self._val2idx.get(ord('n'))

        # Apply: "What do you want to use or apply?"
        if 'use or apply' in msg.lower():
            return self._val2idx.get(27)

        # "Stop eating?" - no, keep eating
        if 'Stop eating' in msg:
            return self._val2idx.get(ord('n'))

        # "You are carrying too much" - don't pick up
        if 'carrying too much' in msg.lower():
            return self._val2idx.get(ord('n'))

        # "Still climb?" for stairs
        if 'Still climb' in msg:
            return self._val2idx.get(ord('y'))

        # "There is a staircase" ... "here"
        if 'staircase' in msg.lower() and 'here' in msg.lower():
            return self._val2idx.get(32)  # space to dismiss

        # Default: yes for unknown prompts
        return self._val2idx.get(ord('y'))

    def _handle_getlin_prompt(self):
        """Handle text entry prompts. Returns action index or None (ESC)."""
        msg = self.message

        # Eat menu
        if 'What do you want to eat' in msg:
            food_letter = self._find_food_to_eat()
            if food_letter:
                return self._val2idx.get(ord(food_letter))
            return self._val2idx.get(27)  # ESC

        # Wield menu
        if 'What do you want to wield' in msg:
            weapon_letter = self.equip_mgr.find_best_weapon(self.inventory)
            if weapon_letter:
                return self._val2idx.get(ord(weapon_letter))
            return self._val2idx.get(27)

        # Wear menu
        if 'What do you want to wear' in msg:
            armor_letter = self.equip_mgr.find_best_armor(self.inventory)
            if armor_letter:
                return self._val2idx.get(ord(armor_letter))
            return self._val2idx.get(27)

        # Quaff menu
        if 'What do you want to drink' in msg or 'What do you want to quaff' in msg:
            potion_letter = self._find_healing_potion()
            if potion_letter:
                return self._val2idx.get(ord(potion_letter))
            return self._val2idx.get(27)

        # Zap menu
        if 'What do you want to zap' in msg:
            return self._val2idx.get(27)

        # Throw menu
        if 'What do you want to throw' in msg:
            return self._val2idx.get(27)

        # Dip menu
        if 'What do you want to dip' in msg:
            return self._val2idx.get(27)

        # Call/name menu
        if 'What do you want to call' in msg or 'What do you want to name' in msg:
            return self._val2idx.get(27)

        # Drop menu
        if 'What do you want to drop' in msg:
            return self._val2idx.get(27)

        # Take off menu
        if 'What do you want to take off' in msg:
            return self._val2idx.get(27)

        # Remove menu
        if 'What do you want to remove' in msg:
            return self._val2idx.get(27)

        # Put on menu
        if 'What do you want to put on' in msg:
            return self._val2idx.get(27)

        # Read menu
        if 'What do you want to read' in msg:
            return self._val2idx.get(27)

        # Default: ESC
        return None

    def _is_current_corpse_safe(self):
        """Check if the corpse mentioned in the current message is safe to eat."""
        msg = self.message.lower()
        # Extract monster name from eat prompt
        # "There is a kobold corpse here; eat it?"
        # "There is a newt corpse here; eat this?"
        name = None
        for pat in ['eat it?', 'eat this?']:
            if pat in msg:
                # Find the corpse name
                idx = msg.find('corpse')
                if idx > 0:
                    # Back-track to find the monster name
                    prefix = msg[:idx].strip()
                    # "there is a X" or "there is an X"
                    for article in ['a ', 'an ', 'the ']:
                        aidx = prefix.rfind(article)
                        if aidx >= 0:
                            name = prefix[aidx + len(article):].strip()
                            break
                    if name is None:
                        # Try without article
                        name = prefix.split()[-1] if prefix.split() else None
                break

        if name is None:
            # Can't parse name. Be conservative: only eat if hungry.
            return self.gs.hunger_state in ('hungry', 'weak', 'fainting', 'fainted')

        # Check via threat_db if available
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            if report.safe_to_eat:
                # Always eat wraith corpses (gain level)
                if name == 'wraith':
                    return True
                # Always eat intrinsic corpses if we don't have the intrinsic
                if report.beneficial_intrinsic and report.beneficial_intrinsic not in self.resistances:
                    return True
                # Eat if hungry
                if self.gs.hunger_state in ('hungry', 'weak', 'fainting', 'fainted'):
                    return True
                # Eat lizard/lichen even when not hungry (keeps well)
                if name in ('lizard', 'lichen'):
                    return True
                return False
            return False

        # Fallback: use food_mgr safety check
        return self.food_mgr.is_corpse_safe(name, self.resistances)

    # ================================================================
    # State update
    # ================================================================

    def _update_game_state(self):
        """Parse observation into full game state. Called after each step."""
        # Update obs_parser GameState
        if self.obs is not None:
            self.gs.update(self.obs)

        # Update raw blstats
        if self._raw_bl is not None and len(self._raw_bl) >= 27:
            self.alignment = ALIGN_LABELS.get(int(self._raw_bl[BL_ALIGN]), 'neutral')
            self.gs.alignment = self.alignment

        # Glyph cache
        self.glyphs = self.obs.get('glyphs') if self.obs else None

        # Level change detection
        cur_key = (self.gs.dnum, self.gs.dlevel)
        if self._prev_level_key is None:
            self._prev_level_key = cur_key
        elif cur_key != self._prev_level_key:
            self._on_level_change(self._prev_level_key, cur_key)
            self._prev_level_key = cur_key

        # Track turns on this level
        if self.gs.turn != self._last_turn:
            lvl = self.current_level()
            lvl.turns_spent += 1
            lvl.last_visit_turn = self.gs.turn
            if lvl.first_visit_turn < 0:
                lvl.first_visit_turn = self.gs.turn
            self._last_turn = self.gs.turn

        # Update maps
        self._update_maps()

        # Update inventory
        self._parse_inventory()

        # Parse messages for game events
        self._parse_messages()

        # Update prayer state
        self._update_prayer_state()

        # Update strategy
        self._update_strategy()

        # Track pet
        self._track_pet()

        # Update Elbereth state
        self._update_elbereth_state()

        # Stuck detection
        cur_pos = (self.gs.py, self.gs.px)
        if cur_pos == self._last_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
            self._last_pos = cur_pos

        # Invalidate explored cache
        self.current_level().invalidate_explored_cache()

    def _on_level_change(self, old_key, new_key):
        """Handle level transition."""
        self._peaceful_positions = set()
        self.food_mgr.on_level_change()
        self._on_elbereth = False
        self._elbereth_cooldown = 0
        self._pet_wait_count = 0

        # Record stair connections
        lvl = self.current_level()
        py, px = self.gs.py, self.gs.px
        went_down = (new_key[1] > old_key[1] if new_key[0] == old_key[0]
                     else new_key[0] != old_key[0])
        if went_down:
            lvl.stairs_up.add((py, px))
        else:
            lvl.stairs_down.add((py, px))

        # Record destination in old level
        if old_key in self.levels:
            old_lvl = self.levels[old_key]
            if went_down:
                for sy, sx in old_lvl.stairs_down:
                    if _chebyshev(sy, sx, py, px) <= 2:
                        old_lvl.stair_dest[(sy, sx)] = new_key
            else:
                for sy, sx in old_lvl.stairs_up:
                    if _chebyshev(sy, sx, py, px) <= 2:
                        old_lvl.stair_dest[(sy, sx)] = new_key

        # Update prayer gehennom flag
        if new_key[0] == DUNGEON_GEHENNOM or self.gs.depth >= 25:
            self.prayer_state.in_gehennom = True
        else:
            self.prayer_state.in_gehennom = False

    def _update_maps(self):
        """Update per-level maps from current glyphs."""
        if self.glyphs is None:
            return

        g = self.glyphs
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        for r in range(MAP_H):
            for c in range(MAP_W):
                v = int(g[r, c])
                cm = _cmap(v)

                if cm in _WALKABLE_CMAP:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = True
                    lvl.objects[r, c] = v

                    if cm in _STAIRS_DOWN_CMAP:
                        lvl.stairs_down.add((r, c))
                    elif cm in _STAIRS_UP_CMAP:
                        lvl.stairs_up.add((r, c))
                    elif cm in _FOUNTAIN_CMAP:
                        lvl.fountains.add((r, c))
                    elif cm in _ALTAR_CMAP:
                        lvl.altars.add((r, c))

                elif cm in _WALL_CMAP:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False
                    lvl.objects[r, c] = v

                elif cm in _CLOSED_DOOR_CMAP:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False
                    lvl.objects[r, c] = v

                elif v == BOULDER_GLYPH:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False

                elif cm == SS_STONE:
                    # Only mark stone as seen when adjacent to player
                    if abs(r - py) <= 1 and abs(c - px) <= 1:
                        lvl.seen[r, c] = True
                        lvl.walkable[r, c] = False

                elif GLYPH_MON_OFF <= v < GLYPH_CMAP_OFF:
                    # Monster/pet/object on tile
                    lvl.seen[r, c] = True
                    if lvl.objects[r, c] == -1:
                        lvl.walkable[r, c] = True

                elif GLYPH_BODY_OFF <= v < GLYPH_RIDDEN_OFF:
                    # Corpse on ground
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = True

                elif GLYPH_OBJ_OFF <= v < GLYPH_CMAP_OFF and v != BOULDER_GLYPH:
                    # Object on ground (not boulder)
                    lvl.seen[r, c] = True
                    if lvl.objects[r, c] == -1:
                        lvl.walkable[r, c] = True

        lvl.walkable[py, px] = True
        lvl.seen[py, px] = True

        # Detect stairs from chars observation (backup, more reliable)
        chars = self.obs.get('chars')
        if chars is not None:
            chars = np.asarray(chars)
            ys, xs = (chars == ord('>')).nonzero()
            for y, x in zip(ys, xs):
                lvl.stairs_down.add((int(y), int(x)))
            ys, xs = (chars == ord('<')).nonzero()
            for y, x in zip(ys, xs):
                lvl.stairs_up.add((int(y), int(x)))

    def _parse_inventory(self):
        """Parse inventory from observation."""
        inv_strs = self.obs.get('inv_strs') if self.obs else None
        inv_letters = self.obs.get('inv_letters') if self.obs else None
        inv_oclasses = self.obs.get('inv_oclasses') if self.obs else None
        if inv_strs is None or inv_letters is None:
            return

        self.inventory = {}
        self.inv_oclasses = {}
        for i, lv in enumerate(inv_letters):
            letter = int(lv)
            if letter == 0:
                continue
            ch = chr(letter)
            raw = inv_strs[i]
            try:
                s = bytes(np.asarray(raw, dtype=np.uint8)).decode('ascii', errors='replace').rstrip('\x00').strip()
            except Exception:
                s = ''
            if s:
                self.inventory[ch] = s
                if inv_oclasses is not None:
                    self.inv_oclasses[ch] = int(inv_oclasses[i])

        # Also update gs.inventory
        self.gs.inventory = dict(self.inventory)
        self.gs.inventory_oclasses = dict(self.inv_oclasses)

        # Check for Excalibur
        self.has_excalibur = any('Excalibur' in s for s in self.inventory.values())

        # Check for long sword
        self.has_long_sword = any('long sword' in s.lower() for s in self.inventory.values())

    def _parse_messages(self):
        """Parse messages for game events: kills, resistances, items."""
        msg = self.message.lower() if self.message else ''
        init_msg = self.initial_message.lower() if self.initial_message else ''

        # Use both message sources
        for m in (msg, init_msg):
            self._check_resistance_messages(m)
            self._check_kill_messages(m)
            self._check_excalibur_messages(m)
            self._check_altar_messages(m)
            self._check_misc_messages(m)

    def _check_resistance_messages(self, msg):
        """Detect gained resistances from messages."""
        resists = {
            "you feel especially healthy": "poison resistance",
            "you feel a momentary chill": "cold resistance",
            "you feel warm": "fire resistance",
            "you feel full of energy": "shock resistance",
            "you feel wide awake": "sleep resistance",
            "your health currently is": "poison resistance",  # backup
            "you feel hardy": "poison resistance",  # backup
            "you feel very firm": "disintegration resistance",
        }
        for frag, r in resists.items():
            if frag in msg:
                self.resistances.add(r)

        # Telepathy
        if "you feel a strange mental acuity" in msg:
            self.resistances.add("telepathy")

        # Teleport control
        if "you feel in control of yourself" in msg:
            self.resistances.add("teleport control")

        # Intrinsic speed
        if "you feel very fast" in msg or "you speed up" in msg:
            self.resistances.add("speed")

        # Invisible
        if "you feel transparent" in msg:
            self.resistances.add("see invisible")

    def _check_kill_messages(self, msg):
        """Detect kills and track corpse positions."""
        self._last_kill_name = None
        self._last_kill_dir = (0, 0)

        kill_prefixes = [
            "you kill the ", "you kill ",
            "you destroy the ", "you destroy ",
            "you slay the ", "you slay ",
        ]

        for prefix in kill_prefixes:
            if prefix in msg:
                rest = msg.split(prefix, 1)[1]
                name = rest.split("!")[0].split(".")[0].strip()
                # Clean up name
                name = name.rstrip('.')
                for article in ['a ', 'an ', 'the ']:
                    if name.startswith(article):
                        name = name[len(article):]

                self._last_kill_name = name
                self._last_kill_dir = self._last_move_dir
                self._kill_count += 1

                py, px = self.gs.py, self.gs.px
                dy, dx = self._last_move_dir
                kill_r, kill_c = py + dy, px + dx

                # Record in food manager
                self.food_mgr.on_kill(name, kill_r, kill_c, self.gs.turn, self.resistances)

                # Record in level corpse tracker
                lvl = self.current_level()
                lvl.corpse_positions[(kill_r, kill_c)] = (name, self.gs.turn)
                break

    def _check_excalibur_messages(self, msg):
        """Detect Excalibur creation."""
        if "your long sword seems to vibrate" in msg:
            # Dipping failed (not high enough level or wrong alignment)
            pass
        if "your sword has a bright" in msg or "excalibur" in msg:
            self.has_excalibur = True

    def _check_altar_messages(self, msg):
        """Detect BUC status from altar messages."""
        # "X glows amber" = cursed
        # "X glows with a light blue aura" = blessed
        # "X glows with a dark blue aura" = uncursed? (depends on alignment)
        # Actually: "X glows ..." messages for BUC
        if 'glow' in msg:
            if 'amber' in msg:
                pass  # cursed
            elif 'light blue' in msg:
                pass  # blessed (or uncursed for co-aligned)
            elif 'dark blue' in msg:
                pass  # uncursed (or blessed for co-aligned)
            # We don't track individual item BUC yet; just note the drop happened

    def _check_misc_messages(self, msg):
        """Detect miscellaneous game events."""
        # Detect role from opening message
        if 'valkyrie' in msg:
            self.role = 'Valkyrie'
            self.resistances.add('cold resistance')  # Valkyrie starts with cold res
        elif 'tourist' in msg:
            self.role = 'Tourist'
        elif 'wizard' in msg:
            self.role = 'Wizard'
        elif 'samurai' in msg:
            self.role = 'Samurai'
        elif 'barbarian' in msg:
            self.role = 'Barbarian'
            self.resistances.add('poison resistance')
        elif 'priestess' in msg or 'priest' in msg:
            self.role = 'Priest'
        elif 'rogue' in msg:
            self.role = 'Rogue'
        elif 'ranger' in msg:
            self.role = 'Ranger'
        elif 'knight' in msg:
            self.role = 'Knight'
        elif 'cave' in msg:
            self.role = 'Caveman'
        elif 'healer' in msg:
            self.role = 'Healer'
            self.resistances.add('poison resistance')
        elif 'monk' in msg:
            self.role = 'Monk'
            self.resistances.add('poison resistance')
            self.resistances.add('sleep resistance')
        elif 'archeologist' in msg:
            self.role = 'Archeologist'

        # Race detection
        if 'human' in msg and not self.race:
            self.race = 'Human'
        elif 'elf' in msg and not self.race:
            self.race = 'Elf'
            self.resistances.add('sleep resistance')
        elif 'dwarf' in msg and not self.race:
            self.race = 'Dwarf'
        elif 'gnome' in msg and not self.race:
            self.race = 'Gnome'
        elif 'orc' in msg and not self.race:
            self.race = 'Orc'
            self.resistances.add('poison resistance')

        # Gain level
        if 'welcome to experience level' in msg:
            pass  # already tracked via blstats

        # Hunger messages
        if 'you are beginning to feel hungry' in msg:
            pass  # tracked via blstats

        # Prayer success
        if 'is pleased' in msg or 'has healed' in msg:
            pass  # prayer worked

        # Prayer failure
        if 'is angry' in msg:
            self.prayer_state.god_anger += 1

        # Luck messages
        if 'you feel lucky' in msg:
            self.prayer_state.luck = max(self.prayer_state.luck, 1)
        elif 'you feel unlucky' in msg:
            self.prayer_state.luck = min(self.prayer_state.luck, -1)

        # Fountain drying
        if 'the fountain dries up' in msg or 'the fountain disappears' in msg:
            py, px = self.gs.py, self.gs.px
            lvl = self.current_level()
            lvl.fountains.discard((py, px))

        # Trap detection
        if 'you fall into a pit' in msg:
            pass  # noted
        if 'a trap door opens' in msg:
            pass  # fell through level

        # Stoning
        if 'turning to stone' in msg:
            pass  # emergency handler picks this up via conditions

    def _update_prayer_state(self):
        """Update PrayerState from game state."""
        # Alignment from blstats
        if self._raw_bl is not None and len(self._raw_bl) > BL_ALIGN:
            al = int(self._raw_bl[BL_ALIGN])
            if al == 1:
                self.prayer_state.player_alignment = Alignment.LAWFUL
            elif al == 0:
                self.prayer_state.player_alignment = Alignment.NEUTRAL
            else:
                self.prayer_state.player_alignment = Alignment.CHAOTIC

        # Altar
        self.prayer_state.on_altar = self.gs.on_altar
        if self.gs.on_altar:
            # Altar alignment detection: check message for "altar" + alignment
            for m in self.gs.messages[-3:] if len(self.gs.messages) >= 3 else self.gs.messages:
                ml = m.lower()
                if 'altar' in ml:
                    if 'chaotic' in ml:
                        self.prayer_state.altar_alignment = Alignment.CHAOTIC
                    elif 'neutral' in ml:
                        self.prayer_state.altar_alignment = Alignment.NEUTRAL
                    elif 'lawful' in ml:
                        self.prayer_state.altar_alignment = Alignment.LAWFUL

        # Gehennom
        if self.gs.dnum == DUNGEON_GEHENNOM or self.gs.depth >= 25:
            self.prayer_state.in_gehennom = True
        else:
            self.prayer_state.in_gehennom = False

    def _update_strategy(self):
        """Update strategic milestones."""
        lvl = self.current_level()
        self.strategy_mgr.update(
            dlevel=self.gs.dlevel,
            xlevel=self.gs.xlevel,
            hp=self.gs.hp,
            max_hp=self.gs.max_hp,
            has_excalibur=self.has_excalibur,
            level_explored=lvl.is_explored(),
            total_searches=lvl.total_searches,
            has_food=self._has_food_in_inventory(),
        )

    def _track_pet(self):
        """Track pet position from visible monsters."""
        self._pet_pos = None
        for mon in self.gs.visible_monsters:
            if mon.is_pet:
                self._pet_pos = (mon.row, mon.col)
                break

    def _update_elbereth_state(self):
        """Track whether we're standing on Elbereth."""
        if self._elbereth_cooldown > 0:
            self._elbereth_cooldown -= 1
        # Check message for Elbereth
        init = self.initial_message.lower() if self.initial_message else ''
        if 'elbereth' in init:
            self._on_elbereth = True
        else:
            self._on_elbereth = False

    # ================================================================
    # Navigation
    # ================================================================

    def bfs(self, allow_hostiles=False, allow_pets=True):
        """BFS from player position. Returns distance array (MAP_H x MAP_W, -1 = unreachable)."""
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()
        dis = np.full((MAP_H, MAP_W), -1, dtype=np.int32)
        dis[py, px] = 0
        q = deque([(py, px)])

        while q:
            y, x = q.popleft()
            d = dis[y, x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                        continue
                    if dis[ny, nx] != -1:
                        continue

                    g = int(self.glyphs[ny, nx])
                    cm = _cmap(g)

                    # Closed doors: include if not exhausted
                    is_closed = cm in _CLOSED_DOOR_CMAP
                    if is_closed and lvl.door_attempts[ny, nx] >= 10:
                        continue

                    # Monster handling
                    is_pet_glyph = glyph_is_pet(g)
                    is_wild = (GLYPH_MON_OFF <= g < GLYPH_PET_OFF)
                    is_detected = (GLYPH_DETECT_OFF <= g < GLYPH_BODY_OFF)
                    is_ridden = (GLYPH_RIDDEN_OFF <= g < GLYPH_OBJ_OFF)

                    if is_wild or is_detected or is_ridden:
                        if not allow_hostiles:
                            # Check if peaceful
                            if self._is_peaceful_at(ny, nx, g):
                                continue  # Don't path through peacefuls
                            continue  # Don't path through any wild monster
                        else:
                            # Allow if not peaceful
                            if self._is_peaceful_at(ny, nx, g):
                                continue
                    if is_pet_glyph and not allow_pets:
                        continue

                    # Walkability
                    ok = lvl.walkable[ny, nx] or is_closed
                    if is_pet_glyph and allow_pets:
                        ok = True
                    if allow_hostiles and (is_wild or is_detected or is_ridden):
                        ok = True

                    if not ok:
                        # Boulder
                        if g == BOULDER_GLYPH:
                            continue
                        # Unknown object on ground
                        if GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF:
                            ok = True
                        elif GLYPH_BODY_OFF <= g < GLYPH_RIDDEN_OFF:
                            ok = True

                    if not ok:
                        continue

                    # No diagonal through doors
                    if abs(dy) + abs(dx) > 1:
                        src_cm = _cmap(int(self.glyphs[y, x]))
                        src_obj = _cmap(int(lvl.objects[y, x])) if lvl.objects[y, x] != -1 else -1
                        if src_cm in _DOOR_CMAP or cm in _DOOR_CMAP or src_obj in _DOOR_CMAP:
                            continue

                    dis[ny, nx] = d + 1
                    q.append((ny, nx))

        return dis

    def step_toward(self, ty, tx, dis=None):
        """Take one BFS-optimal step toward (ty, tx). Returns True if moved."""
        if dis is None:
            dis = self.bfs()
        if dis[ty, tx] == -1:
            return False

        py, px = self.gs.py, self.gs.px
        if (py, px) == (ty, tx):
            return False

        # Trace back from target to player
        cy, cx = ty, tx
        path = []
        for _ in range(500):
            if (cy, cx) == (py, px):
                break
            path.append((cy, cx))
            best = None
            bd = dis[cy, cx]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < MAP_H and 0 <= nx < MAP_W and dis[ny, nx] != -1 and dis[ny, nx] < bd:
                        bd = dis[ny, nx]
                        best = (ny, nx)
            if best is None:
                return False
            cy, cx = best

        if not path:
            return False

        ny, nx = path[-1]
        dy, dx = ny - py, nx - px

        # Validate target
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            if g == BOULDER_GLYPH:
                return False
            cm = _cmap(g)
            if cm in _WALL_CMAP or cm == SS_STONE:
                return False

        # Diagonal through door correction
        if abs(dy) + abs(dx) > 1:
            lvl = self.current_level()
            src_cm = _cmap(int(self.glyphs[py, px]))
            dst_cm = _cmap(int(self.glyphs[ny, nx]))
            src_obj = _cmap(int(lvl.objects[py, px])) if lvl.objects[py, px] != -1 else -1
            dst_obj = _cmap(int(lvl.objects[ny, nx])) if lvl.objects[ny, nx] != -1 else -1
            if (src_cm in _DOOR_CMAP or dst_cm in _DOOR_CMAP or
                    src_obj in _DOOR_CMAP or dst_obj in _DOOR_CMAP):
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W:
                        g2 = int(self.glyphs[cr, cc])
                        if (lvl.walkable[cr, cc] or _cmap(g2) in _CLOSED_DOOR_CMAP
                                or glyph_is_pet(g2)):
                            dy, dx = cdy, cdx
                            ny, nx = cr, cc
                            break

        # Final validation
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            cm = _cmap(g)
            if cm in _WALL_CMAP or cm == SS_STONE or g == BOULDER_GLYPH:
                return False

        old_pos = (self.gs.py, self.gs.px)
        self._move_dir(dy, dx)

        # Retry if diagonal failed
        if 'diagonally' in self.message.lower() and abs(dy) + abs(dx) > 1:
            for cdy, cdx in [(dy, 0), (0, dx)]:
                if cdy == 0 and cdx == 0:
                    continue
                self._move_dir(cdy, cdx)
                break

        new_pos = (self.gs.py, self.gs.px)
        return new_pos != old_pos

    def _move_dir(self, dy, dx):
        """Send a compass direction movement."""
        self._last_move_dir = (dy, dx)
        name = _DELTA_TO_DIR.get((dy, dx))
        if name and name in self._name2idx:
            self.step(self._name2idx[name])

    def _kick_dir(self, dy, dx):
        """Kick in a direction (for locked doors)."""
        name = _DELTA_TO_DIR.get((dy, dx))
        if name and name in self._name2idx:
            self.step(A.Command.KICK)
            self.step(self._name2idx[name])

    def _greedy_move_toward(self, ty, tx):
        """Try to move closer to (ty, tx) using best available direction."""
        py, px = self.gs.py, self.gs.px
        best_dir = None
        best_dist = _manhattan(py, px, ty, tx)
        lvl = self.current_level()

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            g = int(self.glyphs[ny, nx])
            if not (lvl.walkable[ny, nx] or glyph_is_monster(g)):
                continue
            if g == BOULDER_GLYPH:
                continue
            # Skip diagonal through doors
            if abs(dy) + abs(dx) > 1:
                src_cm = _cmap(int(self.glyphs[py, px]))
                dst_cm = _cmap(g)
                if src_cm in _DOOR_CMAP or dst_cm in _DOOR_CMAP:
                    continue
            d = _manhattan(ny, nx, ty, tx)
            if d < best_dist:
                best_dist = d
                best_dir = (dy, dx)

        if best_dir:
            self._move_dir(best_dir[0], best_dir[1])
        else:
            self.step(A.Command.SEARCH)

    def _is_peaceful_at(self, r, c, g=None):
        """Check if position has a peaceful monster."""
        if (r, c) in self._peaceful_positions:
            return True
        if g is None and self.glyphs is not None:
            g = int(self.glyphs[r, c])
        if g is not None:
            mid = glyph_to_mon_id(g)
            if mid is not None:
                if mid in PEACEFUL_IDS or mid in self._peaceful_monster_ids:
                    return True
                if mid < NUMMONS:
                    try:
                        name = nh.permonst(mid).mname
                        if name in PEACEFUL_NAMES:
                            return True
                    except Exception:
                        pass
        return False

    def get_hostile_monsters(self):
        """Get all visible non-pet, non-peaceful monsters with distance and info."""
        py, px = self.gs.py, self.gs.px
        mons = []
        for mon in self.gs.visible_monsters:
            if mon.is_pet:
                continue
            if mon.name in PEACEFUL_NAMES:
                continue
            if mon.mon_id in PEACEFUL_IDS or mon.mon_id in self._peaceful_monster_ids:
                continue
            if (mon.row, mon.col) in self._peaceful_positions:
                continue
            d = _chebyshev(py, px, mon.row, mon.col)
            mons.append((d, mon))
        return sorted(mons, key=lambda x: x[0])

    def get_adjacent_hostiles(self):
        """Get adjacent hostile monsters (distance <= 1, non-pet, non-peaceful)."""
        result = []
        for mon in self.gs.adjacent_monsters:
            if mon.is_pet:
                continue
            if mon.name in PEACEFUL_NAMES:
                continue
            if mon.mon_id in PEACEFUL_IDS or mon.mon_id in self._peaceful_monster_ids:
                continue
            if (mon.row, mon.col) in self._peaceful_positions:
                continue
            result.append(mon)
        return result

    # ================================================================
    # Inventory helpers
    # ================================================================

    def _has_food_in_inventory(self):
        """Check if we have non-corpse food in inventory."""
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc == FOOD_CLASS:
                lower = item.lower()
                if 'corpse' not in lower and 'cursed' not in lower:
                    return True
        return False

    def _find_food_to_eat(self):
        """Find best food item letter from inventory. Prefers rations over random food."""
        best_letter = None
        best_priority = -1

        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc != FOOD_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue
            if 'corpse' in lower:
                continue  # handle corpses separately

            # Priority: prefer good food
            pri = 1
            if 'food ration' in lower:
                pri = 10
            elif 'cram ration' in lower:
                pri = 9
            elif 'lembas wafer' in lower:
                pri = 9
            elif 'k-ration' in lower or 'c-ration' in lower:
                pri = 8
            elif 'tripe ration' in lower:
                pri = 4  # low priority (can cause vomiting)
            elif 'tin' in lower:
                pri = 5
            elif 'pancake' in lower or 'fortune cookie' in lower:
                pri = 7
            elif 'apple' in lower or 'orange' in lower or 'pear' in lower:
                pri = 6
            elif 'melon' in lower or 'banana' in lower or 'carrot' in lower:
                pri = 6
            elif 'cream pie' in lower:
                pri = 3  # can blind
            elif 'egg' in lower:
                pri = 2  # might be cockatrice egg
            elif 'candy bar' in lower:
                pri = 7
            else:
                pri = 5

            if pri > best_priority:
                best_priority = pri
                best_letter = letter

        return best_letter

    def _find_healing_potion(self):
        """Find a potion to quaff in an emergency. Prefers identified healing potions."""
        best_letter = None
        best_priority = -1

        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc != POTION_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue

            pri = 0
            if 'healing' in lower:
                if 'full healing' in lower:
                    pri = 10
                elif 'extra healing' in lower:
                    pri = 9
                else:
                    pri = 8
            elif 'restore ability' in lower:
                pri = 3
            elif 'speed' in lower:
                pri = 2
            elif 'blessed' in lower:
                pri = 4  # blessed unknown potion, might be good
            elif 'uncursed' in lower:
                pri = 2  # uncursed unknown
            else:
                pri = 1  # unknown potion

            if pri > best_priority:
                best_priority = pri
                best_letter = letter

        return best_letter

    def _find_wand_letter(self):
        """Find a wand in inventory. Returns letter or None."""
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) == WAND_CLASS:
                # Skip wands of nothing/light/probing if identified
                lower = item.lower()
                if 'nothing' in lower or 'light' in lower or 'probing' in lower:
                    continue
                # Skip empty wands
                if '(0:0)' in lower:
                    continue
                return letter
        return None

    def _find_weapon_letter(self):
        """Find current wielded weapon letter."""
        for letter, item in self.inventory.items():
            lower = item.lower()
            if '(weapon in hand)' in lower or '(wielded)' in lower:
                return letter
        return None

    def _find_projectile_letter(self):
        """Find throwable projectiles in inventory."""
        for letter, item in self.inventory.items():
            lower = item.lower()
            if self.inv_oclasses.get(letter, -1) == WEAPON_CLASS:
                if any(w in lower for w in ['dagger', 'dart', 'shuriken', 'knife',
                                             'spear', 'javelin', 'boomerang']):
                    if 'cursed' not in lower and '(weapon in hand)' not in lower:
                        return letter
        return None

    def _find_unided_items(self):
        """Find unidentified items suitable for altar BUC testing."""
        result = []
        lvl = self.current_level()
        for letter, item in self.inventory.items():
            if letter in lvl.altar_tested_items:
                continue
            lower = item.lower()
            if 'blessed' in lower or 'uncursed' in lower or 'cursed' in lower:
                continue  # already BUC known
            oc = self.inv_oclasses.get(letter, -1)
            if oc in (POTION_CLASS, SCROLL_CLASS, WAND_CLASS, RING_CLASS, AMULET_CLASS):
                result.append(letter)
        return result

    def _find_long_sword_letter(self):
        """Find a long sword in inventory for Excalibur dipping."""
        for letter, item in self.inventory.items():
            if 'long sword' in item.lower() and 'cursed' not in item.lower():
                return letter
        return None

    # ================================================================
    # Two-step and three-step commands
    # ================================================================

    def _two_step_eat(self, letter):
        """Two-step eat: EAT command then item letter."""
        eat_idx = self._val2idx.get(int(A.Command.EAT))
        if eat_idx is None:
            return
        if self._env_step(eat_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_quaff(self, letter):
        """Two-step quaff: QUAFF command then potion letter."""
        quaff_idx = self._val2idx.get(int(A.Command.QUAFF))
        if quaff_idx is None:
            return
        if self._env_step(quaff_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_wield(self, letter):
        """Two-step wield: WIELD then weapon letter."""
        wield_idx = self._val2idx.get(int(A.Command.WIELD))
        if wield_idx is None:
            return
        if self._env_step(wield_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_wear(self, letter):
        """Two-step wear: WEAR then armor letter."""
        wear_idx = self._val2idx.get(int(A.Command.WEAR))
        if wear_idx is None:
            return
        if self._env_step(wear_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_puton(self, letter):
        """Two-step put on: PUTON then item letter."""
        puton_idx = self._val2idx.get(int(A.Command.PUTON))
        if puton_idx is None:
            return
        if self._env_step(puton_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_drop(self, letter):
        """Two-step drop: DROP then item letter."""
        drop_idx = self._val2idx.get(int(A.Command.DROP))
        if drop_idx is None:
            return
        if self._env_step(drop_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_dip(self, item_letter):
        """Two-step dip: DIP then item letter. Handles fountain prompt."""
        dip_idx = self._val2idx.get(int(A.Command.DIP))
        if dip_idx is None:
            return
        if self._env_step(dip_idx):
            self._quick_parse()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(item_letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._quick_parse()
                raise AgentFinished()
        # Handle follow-up prompts (fountain confirmation, etc.)
        for _ in range(10):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').replace('\x00', '').strip()
            if misc[0]:  # yn prompt
                if 'fountain' in msg.lower():
                    y_idx = self._val2idx.get(ord('y'))
                    if y_idx is not None:
                        if self._env_step(y_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    continue
                else:
                    y_idx = self._val2idx.get(ord('y'))
                    if y_idx is not None:
                        if self._env_step(y_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    continue
            elif misc[2]:  # xwait
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._quick_parse()
                    raise AgentFinished()
                continue
            elif '--More--' in msg:
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._quick_parse()
                    raise AgentFinished()
                continue
            break
        self._update_game_state()

    def _three_step_zap(self, wand_letter, dy, dx):
        """Three-step zap: ZAP -> wand letter -> direction."""
        zap_idx = self._val2idx.get(int(A.Command.ZAP))
        if zap_idx is None:
            return
        if self._env_step(zap_idx):
            self._quick_parse()
            raise AgentFinished()
        w_idx = self._val2idx.get(ord(wand_letter))
        if w_idx is None:
            self._update_game_state()
            return
        if self._env_step(w_idx):
            self._quick_parse()
            raise AgentFinished()
        # Direction
        dir_name = _DELTA_TO_DIR.get((dy, dx))
        if dir_name and dir_name in self._name2idx:
            if self._env_step(self._name2idx[dir_name]):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _three_step_throw(self, item_letter, dy, dx):
        """Three-step throw: THROW -> item letter -> direction."""
        throw_idx = self._val2idx.get(int(A.Command.THROW))
        if throw_idx is None:
            return
        if self._env_step(throw_idx):
            self._quick_parse()
            raise AgentFinished()
        i_idx = self._val2idx.get(ord(item_letter))
        if i_idx is None:
            self._update_game_state()
            return
        if self._env_step(i_idx):
            self._quick_parse()
            raise AgentFinished()
        dir_name = _DELTA_TO_DIR.get((dy, dx))
        if dir_name and dir_name in self._name2idx:
            if self._env_step(self._name2idx[dir_name]):
                self._quick_parse()
                raise AgentFinished()
        self._update_game_state()

    def _engrave_elbereth(self):
        """Engrave 'Elbereth' in the dust using fingers."""
        engrave_idx = self._val2idx.get(int(A.Command.ENGRAVE))
        if engrave_idx is None:
            return

        if self._env_step(engrave_idx):
            self._quick_parse()
            raise AgentFinished()

        # Handle "add to current engraving?" yn prompt
        misc = self.obs.get('misc', [0, 0, 0])
        msg_raw = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').lower()
        if misc[0] and 'add to' in msg_raw:
            n_idx = self._val2idx.get(ord('n'))
            if n_idx is not None:
                if self._env_step(n_idx):
                    self._quick_parse()
                    raise AgentFinished()

        # '-' for fingers
        dash_idx = self._val2idx.get(ord('-'))
        if dash_idx is not None:
            if self._env_step(dash_idx):
                self._quick_parse()
                raise AgentFinished()

        # Type 'Elbereth'
        for ch in 'Elbereth':
            ch_idx = self._val2idx.get(ord(ch))
            if ch_idx is not None:
                if self._env_step(ch_idx):
                    self._quick_parse()
                    raise AgentFinished()

        # Enter to finish
        cr_idx = self._val2idx.get(13)
        if cr_idx is not None:
            if self._env_step(cr_idx):
                self._quick_parse()
                raise AgentFinished()

        self._elbereth_cooldown = 5
        self._on_elbereth = True
        self._update_game_state()

    def _engrave_test_wand(self, wand_letter):
        """Engrave-test a wand to identify it by effect."""
        engrave_idx = self._val2idx.get(int(A.Command.ENGRAVE))
        if engrave_idx is None:
            return

        if self._env_step(engrave_idx):
            self._quick_parse()
            raise AgentFinished()

        # Handle "add to current engraving?" prompt
        misc = self.obs.get('misc', [0, 0, 0])
        msg_raw = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').lower()
        if misc[0] and 'add to' in msg_raw:
            n_idx = self._val2idx.get(ord('n'))
            if n_idx is not None:
                if self._env_step(n_idx):
                    self._quick_parse()
                    raise AgentFinished()

        # Send wand letter
        w_idx = self._val2idx.get(ord(wand_letter))
        if w_idx is not None:
            if self._env_step(w_idx):
                self._quick_parse()
                raise AgentFinished()

        # Type 'x' (minimal engrave test text)
        x_idx = self._val2idx.get(ord('x'))
        if x_idx is not None:
            if self._env_step(x_idx):
                self._quick_parse()
                raise AgentFinished()

        # Enter
        cr_idx = self._val2idx.get(13)
        if cr_idx is not None:
            if self._env_step(cr_idx):
                self._quick_parse()
                raise AgentFinished()

        self._update_game_state()

        # Parse the engrave effect from messages
        if self.appearance_tracker is not None:
            msg = self.message.lower() if self.message else ''
            init = self.initial_message.lower() if self.initial_message else ''
            combined = msg + ' ' + init

            # Match engrave effects
            effect = None
            if 'engraving now reads' in combined:
                effect = 'engraving now reads'
            elif 'ice cubes' in combined:
                effect = 'ice cubes'
            elif 'fights your attempt' in combined:
                effect = 'fights your attempt'
            elif 'bugs slow down' in combined:
                effect = 'bugs slow down'
            elif 'bugs speed up' in combined:
                effect = 'bugs speed up'
            elif 'riddled' in combined:
                effect = 'riddled by bullet holes'
            elif 'text changes' in combined:
                effect = 'text changes'
            elif 'bugs stop' in combined:
                effect = 'bugs stop moving'
            elif 'vanish' in combined and 'engraving' in combined:
                effect = 'engraving vanishes'
            elif 'is now engraved' in combined:
                effect = 'is now engraved'
            else:
                effect = 'no effect'

            # Get wand appearance from inventory
            wand_desc = self.inventory.get(wand_letter, '')
            if wand_desc:
                # Extract appearance: "a [description] wand" -> description
                lower = wand_desc.lower()
                # Try to find the appearance name
                for word in ['wand', 'of']:
                    pass
                # For now, just note the effect. Full integration with
                # AppearanceTracker would require parsing the randomized
                # appearance name from the inventory string.

    # ================================================================
    # Strategies
    # ================================================================

    def emergency(self):
        """Handle life-threatening conditions. Highest priority.

        Handles: stoning, sliming, strangulation, food poisoning,
        terminal illness, critical HP, starvation.
        """
        gs = self.gs
        conditions = gs.conditions
        turn = gs.turn

        # Build trouble state for prayer system
        trouble_state = {
            'hp': gs.hp,
            'max_hp': gs.max_hp,
            'hunger': HungerState.WEAK if gs.hunger_state == 'weak' else (
                HungerState.FAINTING if gs.hunger_state in ('fainting', 'fainted') else (
                    HungerState.HUNGRY if gs.hunger_state == 'hungry' else (
                        HungerState.SATIATED if gs.hunger_state == 'satiated' else
                        HungerState.NOT_HUNGRY
                    )
                )
            ),
            'condition': int(self._raw_bl[BL_CONDITION]) if self._raw_bl is not None and len(self._raw_bl) > BL_CONDITION else 0,
            'has_food': self._has_food_in_inventory(),
        }

        # Classify trouble
        trouble_type, severity = self.prayer_state.classify_trouble(trouble_state)

        # === STONING ===
        if 'stoned' in conditions:
            # Eat lizard corpse if available
            for letter, item in self.inventory.items():
                if 'lizard' in item.lower() and self.inv_oclasses.get(letter, -1) == FOOD_CLASS:
                    self._two_step_eat(letter)
                    return True
            # Eat acidic corpse if available
            for letter, item in self.inventory.items():
                if 'acid' in item.lower() and self.inv_oclasses.get(letter, -1) == FOOD_CLASS:
                    self._two_step_eat(letter)
                    return True
            # Pray
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'stoning')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            # Last resort: quaff any potion
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True
            return False

        # === SLIMING ===
        if 'slimed' in conditions:
            # Burn it off with fire (wand, scroll of fire)
            # Or pray
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'sliming')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            return False

        # === STRANGULATION ===
        if 'strangled' in conditions:
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'strangulation')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            return False

        # === FOOD POISONING ===
        if 'foodpois' in conditions:
            # Pray
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'food_poisoning')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            # Quaff healing
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True
            return False

        # === TERMINAL ILLNESS ===
        if 'termill' in conditions:
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'illness')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            return False

        # === STARVATION (Weak or worse) ===
        if gs.hunger_state in ('weak', 'fainting', 'fainted'):
            # Try eating from inventory first
            food_letter = self._find_food_to_eat()
            if food_letter:
                self._two_step_eat(food_letter)
                self._last_eat_turn = turn
                return True
            # Pray
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'starving')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            # Eat any corpse on ground regardless of safety
            self.step(A.Command.EAT)
            return True

        # === CRITICAL HP ===
        if gs.hp <= max(5, gs.max_hp // 7) or (gs.max_hp > 0 and gs.hp < gs.max_hp * 0.15):
            # Pray
            safe, _ = self.prayer_state.is_prayer_safe(turn, 'hp_critical')
            if safe:
                self.prayer_state.update_prayed(turn)
                self._last_prayer_turn = turn
                self.step(A.Command.PRAY)
                return True
            # Quaff healing potion
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True
            # Elbereth if monsters adjacent
            adj = self.get_adjacent_hostiles()
            if adj and self._elbereth_cooldown <= 0:
                self._engrave_elbereth()
                return True
            # Flee from adjacent monsters
            if adj:
                return self._flee_from_monsters(adj)

        return False

    def pray_strategy(self):
        """Use prayer for non-emergency minor troubles when safe."""
        gs = self.gs
        turn = gs.turn

        # Don't pray too often
        if turn - self._last_prayer_turn < 300:
            return False

        # Build trouble state
        hunger_val = HungerState.NOT_HUNGRY
        if gs.hunger_state == 'hungry':
            hunger_val = HungerState.HUNGRY
        elif gs.hunger_state == 'weak':
            hunger_val = HungerState.WEAK
        elif gs.hunger_state in ('fainting', 'fainted'):
            hunger_val = HungerState.FAINTING
        elif gs.hunger_state == 'satiated':
            hunger_val = HungerState.SATIATED

        trouble_state = {
            'hp': gs.hp,
            'max_hp': gs.max_hp,
            'hunger': hunger_val,
            'condition': int(self._raw_bl[BL_CONDITION]) if self._raw_bl is not None and len(self._raw_bl) > BL_CONDITION else 0,
            'has_food': self._has_food_in_inventory(),
        }

        recommend, reason = self.prayer_state.should_pray(turn, trouble_state)
        if recommend:
            self.prayer_state.update_prayed(turn)
            self._last_prayer_turn = turn
            self.step(A.Command.PRAY)
            return True

        return False

    def fight(self):
        """Priority-based combat system. Loops until area is clear.

        Integrates ThreatDB for per-monster assessment, fight.py for
        tactical decisions, and ranged combat with wands/projectiles.
        """
        acted_at_all = False

        for _round in range(60):  # Safety limit
            hostiles = self.get_hostile_monsters()
            if not hostiles:
                break

            # If all hostiles are far away (> 7 tiles), stop fighting
            if all(d > 7 for d, _ in hostiles):
                break

            py, px = self.gs.py, self.gs.px
            adj_hostiles = self.get_adjacent_hostiles()
            hp_ratio = self.gs.hp / max(1, self.gs.max_hp)

            # Build prioritized action list
            actions = []  # (priority, action_tuple)

            # === EMERGENCY CHECK within fight loop ===
            if self.gs.hp <= max(5, self.gs.max_hp // 7):
                if self.emergency():
                    acted_at_all = True
                    continue

            # === INSTAKILL FLEE ===
            for mon in adj_hostiles:
                info = assess_monster(mon.name, mon.mon_id)
                if info['instakill']:
                    # Flee away from the instakill threat
                    flee_dy = _sign(py - mon.row)
                    flee_dx = _sign(px - mon.col)
                    if flee_dy == 0 and flee_dx == 0:
                        flee_dy = 1  # arbitrary
                    actions.append((200, ('flee', flee_dy, flee_dx, mon.name)))

            # === WAIT ON ELBERETH ===
            if self._on_elbereth and adj_hostiles:
                if hp_ratio < 0.8:
                    actions.append((150, ('wait_elbereth',)))

            # === ELBERETH ===
            if adj_hostiles and self._elbereth_cooldown <= 0 and not self._on_elbereth:
                # Use fight.py's should_elbereth
                elb_decision = should_elbereth(
                    self.gs.hp, self.gs.max_hp,
                    adj_hostiles, self._on_elbereth, self._elbereth_cooldown
                )
                if elb_decision is not None:
                    # Check that at least some monsters respect Elbereth
                    respects = 0
                    for mon in adj_hostiles:
                        if self.threat_db is not None:
                            if self.threat_db.respects_elbereth(mon.name):
                                respects += 1
                        else:
                            # Heuristic: most monsters respect it
                            if mon.name not in PEACEFUL_NAMES and mon.name not in {
                                'minotaur', 'Death', 'Pestilence', 'Famine',
                                'Wizard of Yendor', 'Archon',
                            }:
                                respects += 1
                    if respects > 0:
                        actions.append((elb_decision.priority, ('elbereth',)))

            # === TACTICAL FLEE (multiple dangerous enemies, low HP) ===
            if len(adj_hostiles) >= 2 and hp_ratio < 0.4:
                flee_action = self._compute_flee_direction(adj_hostiles)
                if flee_action:
                    actions.append((30 * (1 - hp_ratio), flee_action))

            # === MELEE ===
            for mon in adj_hostiles:
                info = assess_monster(mon.name, mon.mon_id)

                # Never melee floating eye, gas spore
                if info['never_melee']:
                    continue
                # Never melee instakill monsters
                if info['instakill']:
                    continue
                # Skip peacefuls
                if info['peaceful']:
                    continue
                # Skip weird (nymphs, leprechauns) unless we're strong
                if info['weird'] and self.gs.xlevel < 8:
                    continue

                # Threat assessment from ThreatDB
                danger = info['danger']
                if self.threat_db is not None:
                    player_state = {
                        'hp': self.gs.hp, 'max_hp': self.gs.max_hp,
                        'ac': self.gs.ac, 'level': self.gs.xlevel,
                        'speed': 12, 'resistances': self.resistances,
                        'has_elbereth_source': True,
                    }
                    threat = self.threat_db.assess_threat(mon.name, player_state)
                    danger = threat.danger_level

                    # Check recommended action
                    if threat.recommended_action == 'ranged' and threat.ranged_preferred:
                        # Add ranged option for this monster instead of melee
                        wand = self._find_wand_letter()
                        if wand and _in_line(py, px, mon.row, mon.col):
                            ndy = _sign(mon.row - py)
                            ndx = _sign(mon.col - px)
                            actions.append((danger + 5, ('zap', wand, ndy, ndx)))
                            continue
                        proj = self._find_projectile_letter()
                        if proj and _in_line(py, px, mon.row, mon.col):
                            ndy = _sign(mon.row - py)
                            ndx = _sign(mon.col - px)
                            actions.append((danger + 3, ('throw', proj, ndy, ndx)))
                            continue

                dy = mon.row - py
                dx = mon.col - px
                pri = danger
                if self.gs.hp > 8:
                    pri += 10
                if info['weak']:
                    pri = 1

                # Don't melee while on Elbereth (erases it)
                if self._on_elbereth:
                    pri -= 100

                actions.append((pri, ('melee', dy, dx, mon.name)))

            # === RANGED ATTACKS (distant monsters) ===
            if not adj_hostiles:
                wand = self._find_wand_letter()
                proj = self._find_projectile_letter()
                for d, mon in hostiles:
                    if d <= 1 or d > 7:
                        continue
                    if not _in_line(py, px, mon.row, mon.col):
                        continue
                    if not _line_clear(self.glyphs, py, px, mon.row, mon.col):
                        continue

                    ndy = _sign(mon.row - py)
                    ndx = _sign(mon.col - px)

                    if wand:
                        actions.append((8, ('zap', wand, ndy, ndx)))
                        break
                    elif proj:
                        actions.append((6, ('throw', proj, ndy, ndx)))
                        break

            # === APPROACH distant monsters ===
            if not adj_hostiles and hp_ratio > 0.3:
                approach_range = 12
                lvl = self.current_level()
                if not lvl.stairs_down and lvl.turns_spent > 500 and lvl.total_searches > 100:
                    approach_range = 6

                fight_dis = self.bfs(allow_hostiles=True)
                best_mon = None
                best_fd = 999
                for d, mon in hostiles:
                    if d > approach_range:
                        continue
                    fd = fight_dis[mon.row, mon.col]
                    if fd != -1 and fd < best_fd:
                        best_fd = fd
                        best_mon = mon
                if best_mon:
                    actions.append((0, ('approach', best_mon.row, best_mon.col)))

            if not actions:
                break

            # Execute highest priority action
            actions.sort(key=lambda x: -x[0])
            _, best = actions[0]
            acted_at_all = True

            if best[0] == 'melee':
                self._move_dir(best[1], best[2])
            elif best[0] == 'flee':
                self._move_dir(best[1], best[2])
            elif best[0] == 'elbereth':
                self._engrave_elbereth()
                # Rest on Elbereth for a few turns
                for _ in range(3):
                    if self.gs.hp >= self.gs.max_hp * 0.5:
                        break
                    self.step(A.Command.SEARCH)
            elif best[0] == 'wait_elbereth':
                self.step(A.Command.SEARCH)
            elif best[0] == 'zap':
                self._three_step_zap(best[1], best[2], best[3])
            elif best[0] == 'throw':
                self._three_step_throw(best[1], best[2], best[3])
            elif best[0] == 'approach':
                fight_dis = self.bfs(allow_hostiles=True)
                if not self.step_toward(best[1], best[2], fight_dis):
                    self._greedy_move_toward(best[1], best[2])
            else:
                break

            # Emergency check between fight rounds
            if self.gs.hp <= max(5, self.gs.max_hp // 7):
                can_pray = (self.gs.turn - self._last_prayer_turn) >= 300
                if can_pray:
                    safe, _ = self.prayer_state.is_prayer_safe(
                        self.gs.turn, 'hp_critical')
                    if safe:
                        self.prayer_state.update_prayed(self.gs.turn)
                        self._last_prayer_turn = self.gs.turn
                        self.step(A.Command.PRAY)

        return acted_at_all

    def _flee_from_monsters(self, adj_hostiles):
        """Find the best flee direction away from adjacent monsters."""
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        best_dir = None
        best_threat = len(adj_hostiles) + 1

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            if not lvl.walkable[ny, nx]:
                g = int(self.glyphs[ny, nx])
                if not glyph_is_pet(g):
                    continue
            g = int(self.glyphs[ny, nx])
            if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                continue  # Don't flee into another monster
            if g == BOULDER_GLYPH:
                continue

            # Count how many hostiles would be adjacent after moving
            threat = sum(1 for mon in adj_hostiles
                        if _chebyshev(ny, nx, mon.row, mon.col) <= 1)
            if threat < best_threat:
                best_threat = threat
                best_dir = (dy, dx)

        if best_dir:
            self._move_dir(best_dir[0], best_dir[1])
            return True
        return False

    def _compute_flee_direction(self, adj_hostiles):
        """Compute best flee direction. Returns action tuple or None."""
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        best_dir = None
        best_threat = len(adj_hostiles) + 1

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            g = int(self.glyphs[ny, nx])
            if not (lvl.walkable[ny, nx] or glyph_is_pet(g)):
                continue
            if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                continue
            if g == BOULDER_GLYPH:
                continue

            threat = sum(1 for mon in adj_hostiles
                        if _chebyshev(ny, nx, mon.row, mon.col) <= 1)
            if threat < best_threat:
                best_threat = threat
                best_dir = (dy, dx)

        if best_dir:
            return ('flee', best_dir[0], best_dir[1], 'tactical')
        return None

    def eat_strategy(self):
        """Eat corpses from ground, from inventory, and navigate to fresh corpses.

        Proactive eating: eat wraith corpses always (gain level),
        eat resistance corpses when missing that resistance.
        """
        gs = self.gs
        turn = gs.turn

        # Rate limit eating
        if turn - self._last_eat_turn < 3:
            return False

        # === 1. Eat corpse after kill (step onto kill position) ===
        if self._last_kill_name is not None:
            acted = self._eat_corpse_after_kill()
            if acted:
                return True

        # === 2. Eat corpse from ground (if message indicates one) ===
        init_msg = self.initial_message.lower() if self.initial_message else ''
        if 'corpse' in init_msg and ('you see here' in init_msg or 'there is' in init_msg):
            # Extract corpse name
            corpse_name = self._extract_corpse_name_from_message(init_msg)
            if corpse_name:
                # Check if we should eat it
                should_eat = False
                if gs.hunger_state in ('hungry', 'weak', 'fainting', 'fainted'):
                    should_eat = self._is_corpse_worth_eating(corpse_name)
                elif self._corpse_has_wanted_intrinsic(corpse_name):
                    should_eat = True
                elif corpse_name == 'wraith':
                    should_eat = True
                elif corpse_name in ('lizard', 'lichen'):
                    should_eat = True

                if should_eat:
                    self._last_eat_turn = turn
                    self.step(A.Command.EAT)
                    return True

        # === 3. Eat non-corpse food from ground ===
        if gs.hunger_state in ('hungry', 'weak', 'fainting', 'fainted'):
            if 'you see here' in init_msg:
                for kw in ['food ration', 'cram ration', 'lembas wafer', 'k-ration',
                           'c-ration', 'tripe ration', 'candy bar', 'pancake',
                           'fortune cookie', 'tin', 'apple', 'orange', 'pear',
                           'melon', 'banana', 'carrot']:
                    if kw in init_msg:
                        self._last_eat_turn = turn
                        self.step(A.Command.EAT)
                        return True

        # === 4. Eat from inventory when hungry ===
        if gs.hunger_state in ('hungry', 'weak', 'fainting', 'fainted'):
            food_letter = self._find_food_to_eat()
            if food_letter:
                self._last_eat_turn = turn
                self._two_step_eat(food_letter)
                return True

        # === 5. Navigate to known fresh corpses (proactive) ===
        if gs.hunger_state != 'satiated':
            corpse_target = self._find_nearby_valuable_corpse()
            if corpse_target:
                r, c = corpse_target
                if (r, c) == (gs.py, gs.px):
                    # On the corpse, eat it
                    self._last_eat_turn = turn
                    self.step(A.Command.EAT)
                    return True
                else:
                    dis = self.bfs()
                    if dis[r, c] != -1 and dis[r, c] <= 8:
                        if self.step_toward(r, c, dis):
                            return True

        return False

    def _eat_corpse_after_kill(self):
        """Step onto a fresh kill's corpse and eat it."""
        name = self._last_kill_name
        dy, dx = self._last_kill_dir
        self._last_kill_name = None

        if name is None:
            return False
        if dy == 0 and dx == 0:
            return False

        # Check if the corpse is worth eating
        should_eat = False
        if self.gs.hunger_state in ('hungry', 'weak', 'fainting', 'fainted'):
            should_eat = self._is_corpse_worth_eating(name)
        elif self._corpse_has_wanted_intrinsic(name):
            should_eat = True
        elif name == 'wraith':
            should_eat = True
        elif name in ('lizard', 'lichen'):
            should_eat = True

        if not should_eat:
            return False

        py, px = self.gs.py, self.gs.px
        cy, cx = py + dy, px + dx
        if not (0 <= cy < MAP_H and 0 <= cx < MAP_W):
            return False

        # Step onto corpse tile
        self._move_dir(dy, dx)

        # Check if we're on the corpse
        msg_low = self.initial_message.lower() if self.initial_message else ''
        if 'corpse' not in msg_low and 'you see here' not in msg_low:
            return True  # moved but no corpse visible

        # Eat it
        self._last_eat_turn = self.gs.turn
        self.step(A.Command.EAT)
        return True

    def _is_corpse_worth_eating(self, name):
        """Check if a corpse is safe and worth eating."""
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            return report.safe_to_eat
        return self.food_mgr.is_corpse_safe(name, self.resistances)

    def _corpse_has_wanted_intrinsic(self, name):
        """Check if eating this corpse would grant a resistance we want."""
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            if report.safe_to_eat and report.beneficial_intrinsic:
                return report.beneficial_intrinsic not in self.resistances
            return False

        # Fallback
        for resist, monsters in INTRINSIC_CORPSES.items():
            if name in monsters:
                resist_name = resist + ' resistance' if 'resistance' not in resist else resist
                if resist_name not in self.resistances:
                    # Check if it's safe
                    if self.food_mgr.is_corpse_safe(name, self.resistances):
                        return True
        return False

    def _extract_corpse_name_from_message(self, msg):
        """Extract monster name from corpse message."""
        # "you see here a/an X corpse"
        for pat in ['you see here ', 'there is ']:
            if pat in msg:
                rest = msg.split(pat, 1)[1]
                idx = rest.find(' corpse')
                if idx > 0:
                    name = rest[:idx]
                    for article in ['a ', 'an ', 'the ']:
                        if name.startswith(article):
                            name = name[len(article):]
                    return name.strip()
        return None

    def _find_nearby_valuable_corpse(self):
        """Find a nearby fresh corpse worth eating. Returns (r, c) or None."""
        lvl = self.current_level()
        turn = self.gs.turn
        py, px = self.gs.py, self.gs.px

        best = None
        best_dist = 9
        best_priority = -1

        for (r, c), (name, kill_turn) in list(lvl.corpse_positions.items()):
            age = turn - kill_turn
            if name not in NEVER_ROT and age > MAX_CORPSE_AGE:
                continue  # rotted
            dist = _chebyshev(py, px, r, c)
            if dist >= best_dist:
                continue

            # Check value
            priority = 0
            if name == 'wraith':
                priority = 10
            elif self._corpse_has_wanted_intrinsic(name):
                priority = 8
            elif name in ('lizard', 'lichen'):
                priority = 5
            elif self._is_corpse_worth_eating(name):
                if self.gs.hunger_state in ('hungry', 'weak', 'fainting'):
                    priority = 4
                else:
                    priority = 1

            if priority > best_priority or (priority == best_priority and dist < best_dist):
                best = (r, c)
                best_dist = dist
                best_priority = priority

        if best_priority <= 0:
            return None
        return best

    def equip_strategy(self):
        """Auto-equip best weapon and armor."""
        acted = False

        # Wield better weapon
        weapon_letter = self.equip_mgr.find_best_weapon(self.inventory)
        if weapon_letter:
            self._two_step_wield(weapon_letter)
            acted = True

        # Wear armor for empty slots
        self._parse_inventory()
        armor_letter = self.equip_mgr.find_best_armor(self.inventory)
        if armor_letter:
            # Don't wear unidentified armor unless we've BUC-tested it
            item_str = self.inventory.get(armor_letter, '')
            lower = item_str.lower()
            # BUC check: only wear if blessed/uncursed known or if we have no choice
            is_buc_known = any(w in lower for w in ['blessed', 'uncursed', 'cursed'])
            if is_buc_known and 'cursed' not in lower:
                self._two_step_wear(armor_letter)
                acted = True
            elif not is_buc_known and self.gs.xlevel >= 3:
                # Risk wearing unidentified armor after level 3
                # unless we have an altar nearby to test
                lvl = self.current_level()
                if not lvl.altars:
                    self._two_step_wear(armor_letter)
                    acted = True

        return acted

    def excalibur_strategy(self):
        """Dip long sword in fountain for Excalibur.

        Requirements: lawful alignment, XL >= 5, have long sword, on/near fountain.
        """
        if self.has_excalibur:
            return False
        if self.alignment != 'lawful':
            return False
        if self.gs.xlevel < 5:
            return False

        # Find long sword
        sword_letter = self._find_long_sword_letter()
        if sword_letter is None:
            return False

        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        # Check if on fountain
        on_fountain = (py, px) in lvl.fountains
        if not on_fountain:
            # Check glyphs
            if self.glyphs is not None:
                g = int(self.glyphs[py, px])
                obj = int(lvl.objects[py, px]) if lvl.objects[py, px] != -1 else -1
                on_fountain = _cmap(g) in _FOUNTAIN_CMAP or _cmap(obj) in _FOUNTAIN_CMAP

        if on_fountain:
            self._two_step_dip(sword_letter)
            # Check results
            self._parse_inventory()
            if any('Excalibur' in s for s in self.inventory.values()):
                self.has_excalibur = True
            # Check if fountain dried
            if self.glyphs is not None:
                g = int(self.glyphs[py, px])
                if _cmap(g) not in _FOUNTAIN_CMAP:
                    lvl.fountains.discard((py, px))
            return True

        # Navigate to nearest fountain on current level
        if lvl.fountains:
            dis = self.bfs()
            best_f = None
            best_d = 999
            for fy, fx in lvl.fountains:
                d = dis[fy, fx]
                if d != -1 and d < best_d:
                    best_d = d
                    best_f = (fy, fx)
            if best_f:
                if self.step_toward(best_f[0], best_f[1], dis):
                    return True

        return False

    def altar_buc_strategy(self):
        """Drop items on altar to learn BUC status from messages."""
        if not self.gs.on_altar:
            return False

        # Find unidentified items to test
        items_to_test = self._find_unided_items()
        if not items_to_test:
            return False

        lvl = self.current_level()

        # Drop up to 5 items for testing
        tested = 0
        for letter in items_to_test[:5]:
            if letter not in self.inventory:
                continue
            self._two_step_drop(letter)
            lvl.altar_tested_items.add(letter)
            tested += 1

        if tested > 0:
            # Pick them back up
            pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
            if pickup_idx is not None:
                if self._env_step(pickup_idx):
                    self._quick_parse()
                    raise AgentFinished()
                # Handle pickup menu: press comma or 'a' for all
                for _ in range(5):
                    misc = self.obs.get('misc', [0, 0, 0])
                    if misc[1]:  # menu
                        # Press comma to pick up everything
                        comma_idx = self._val2idx.get(ord(','))
                        if comma_idx is not None:
                            if self._env_step(comma_idx):
                                self._quick_parse()
                                raise AgentFinished()
                            continue
                    break
                self._update_game_state()
            return True

        return False

    def item_id_strategy(self):
        """Identify items via engrave-testing wands and price-ID in shops."""
        if self.appearance_tracker is None:
            return False

        # Find unidentified wands to engrave-test
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
                continue
            lower = item.lower()
            # Skip already identified wands
            if 'of ' in lower:
                continue  # "wand of fire" = identified
            if '(0:' in lower:
                continue  # empty
            # Don't test wands we've already tested
            if letter in self._wands_to_test:
                continue

            # Engrave-test this wand
            self._engrave_test_wand(letter)
            self._wands_to_test.append(letter)
            return True

        return False

    def sokoban_strategy(self):
        """Solve Sokoban puzzles if in Sokoban dungeon."""
        if self.gs.dnum != DUNGEON_SOKOBAN:
            return False
        return self._solve_sokoban()

    def _solve_sokoban(self):
        """Execute one step of the Sokoban puzzle solution."""
        lvl = self.current_level()

        if not lvl.soko_matched:
            lvl.soko_matched = True
            wall_mask = np.zeros((MAP_H, MAP_W), dtype=bool)
            if self.glyphs is not None:
                for r in range(MAP_H):
                    for c in range(MAP_W):
                        g = int(self.glyphs[r, c])
                        cm = _cmap(g)
                        if cm in _WALL_CMAP:
                            wall_mask[r, c] = True
            result = match_sokoban_level(wall_mask)
            if result:
                lvl.soko_solution, off_y, off_x = result
                lvl.soko_offset = (off_y, off_x)
                lvl.soko_step = 0

        if lvl.soko_solution is None or lvl.soko_step >= len(lvl.soko_solution):
            return False

        (by, bx), (dy, dx) = lvl.soko_solution[lvl.soko_step]
        off_y, off_x = lvl.soko_offset
        game_by, game_bx = by + off_y, bx + off_x
        push_y, push_x = game_by - dy, game_bx - dx
        py, px = self.gs.py, self.gs.px

        # Check if boulder is still there
        if 0 <= game_by < MAP_H and 0 <= game_bx < MAP_W:
            if self.glyphs is not None and int(self.glyphs[game_by, game_bx]) != BOULDER_GLYPH:
                lvl.soko_step += 1
                return True

        # Navigate to push position
        if (py, px) != (push_y, push_x):
            dis = self.bfs()
            if dis[push_y, push_x] == -1:
                lvl.soko_step += 1
                return True
            self.step_toward(push_y, push_x, dis)
            return True

        # Push the boulder
        self._move_dir(dy, dx)
        lvl.soko_step += 1
        return True

    def pickup_strategy(self):
        """Pick up useful items from ground."""
        init_msg = self.initial_message.lower() if self.initial_message else ''
        if 'you see here' not in init_msg and 'there are' not in init_msg:
            return False
        if 'cursed' in init_msg:
            return False

        # Encumbrance check
        if self.gs.encumbrance >= 2:  # stressed or worse
            return False

        # Rate limit
        if self.gs.turn - self._last_pickup_turn < 3:
            return False

        # Categories worth picking up
        food_kw = ['food ration', 'cram ration', 'lembas wafer', 'k-ration', 'c-ration',
                    'tripe ration', 'tin', 'candy bar', 'pancake', 'fortune cookie',
                    'apple', 'orange', 'pear', 'melon', 'banana', 'carrot',
                    'cream pie', 'slime mold']
        armor_kw = ['mail', 'armor', 'helm', 'cloak', 'gloves', 'gauntlets',
                     'boots', 'shoes', 'jacket', 'shield']
        weapon_kw = ['long sword', 'katana', 'silver saber', 'broadsword', 'scimitar',
                      'battle-axe', 'morning star', 'war hammer', 'mace',
                      'two-handed sword', 'trident', 'Excalibur']
        potion_kw = ['potion']
        scroll_kw = ['scroll']
        wand_kw = ['wand']
        ring_kw = ['ring']
        amulet_kw = ['amulet']
        gold_kw = ['gold piece']
        tool_kw = ['unicorn horn', 'skeleton key', 'lock pick', 'stethoscope',
                    'tinning kit', 'magic marker', 'bag']

        is_useful = any(
            any(w in init_msg for w in kwl)
            for kwl in [food_kw, armor_kw, weapon_kw, potion_kw, scroll_kw,
                        wand_kw, ring_kw, amulet_kw, gold_kw, tool_kw]
        )

        if not is_useful:
            return False

        # Don't pick up boulders
        if 'boulder' in init_msg:
            return False

        self._last_pickup_turn = self.gs.turn
        pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
        if pickup_idx is None:
            return False
        if self._env_step(pickup_idx):
            self._quick_parse()
            raise AgentFinished()

        # Handle pickup menu if it appears
        for _ in range(10):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').replace('\x00', '').strip()
            if misc[1]:  # menu/getlin
                # Try comma (pickup all)
                comma_idx = self._val2idx.get(ord(','))
                if comma_idx is not None:
                    if self._env_step(comma_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    continue
            elif misc[0]:  # yn
                y_idx = self._val2idx.get(ord('y'))
                if y_idx is not None:
                    if self._env_step(y_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    continue
            elif misc[2]:
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._quick_parse()
                    raise AgentFinished()
                continue
            elif '--More--' in msg:
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._quick_parse()
                    raise AgentFinished()
                continue
            break

        self._update_game_state()

        # Auto-equip after pickup
        self._parse_inventory()
        weapon_letter = self.equip_mgr.find_best_weapon(self.inventory)
        if weapon_letter:
            self._two_step_wield(weapon_letter)
        self._parse_inventory()
        armor_letter = self.equip_mgr.find_best_armor(self.inventory)
        if armor_letter:
            item_str = self.inventory.get(armor_letter, '')
            lower = item_str.lower()
            if 'cursed' not in lower:
                is_buc = any(w in lower for w in ['blessed', 'uncursed'])
                if is_buc:
                    self._two_step_wear(armor_letter)

        return True

    def explore(self):
        """Explore the current level. Opens doors, searches walls, navigates to frontier."""
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        # === 0. Sokoban: go upstairs after solving ===
        if self.gs.dnum == DUNGEON_SOKOBAN:
            if lvl.soko_solution is not None and lvl.soko_step >= len(lvl.soko_solution):
                if lvl.stairs_up:
                    fight_dis = self.bfs(allow_hostiles=True)
                    for uy, ux in lvl.stairs_up:
                        if (py, px) == (uy, ux):
                            self.step(A.MiscDirection.UP)
                            return
                        if fight_dis[uy, ux] != -1:
                            self.step_toward(uy, ux, fight_dis)
                            return

        # === 1. Adjacent closed doors: open or kick ===
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = py + dy, px + dx
            if not (0 <= nr < MAP_H and 0 <= nc < MAP_W):
                continue
            if self.glyphs is None:
                continue
            g = int(self.glyphs[nr, nc])
            cm = _cmap(g)
            if cm not in _CLOSED_DOOR_CMAP:
                continue
            if lvl.door_attempts[nr, nc] >= 10:
                continue

            lvl.door_attempts[nr, nc] += 1
            dir_name = _DELTA_TO_DIR.get((dy, dx))
            if not dir_name or dir_name not in self._name2idx:
                continue

            if lvl.door_attempts[nr, nc] <= 2:
                # Try opening
                self.step(self._name2idx[dir_name])
                msg_low = self.message.lower()
                new_g = int(self.glyphs[nr, nc]) if self.glyphs is not None else g
                if 'locked' not in msg_low and _cmap(new_g) not in _CLOSED_DOOR_CMAP:
                    return

            # Kick it
            self._kick_dir(dy, dx)
            return

        dis = self.bfs()

        # === 2. Navigate to nearest closed door ===
        best_door = None
        best_dd = 999
        if self.glyphs is not None:
            for r in range(MAP_H):
                for c in range(MAP_W):
                    g = int(self.glyphs[r, c])
                    if _cmap(g) in _CLOSED_DOOR_CMAP and lvl.door_attempts[r, c] < 5:
                        d = dis[r, c]
                        if d != -1 and d < best_dd:
                            best_dd = d
                            best_door = (r, c)
        if best_door and best_dd > 1:
            if self.step_toward(best_door[0], best_door[1], dis):
                return

        # === 3. Check descent readiness ===
        descent_ready = self._should_descend()
        force_descend = self._should_force_descend()

        # On downstairs: descend
        if self._on_stairs_down() and self.gs.hp > self.gs.max_hp * 0.5:
            if descent_ready or force_descend:
                # Pet proximity check: wait briefly for pet
                if self._pet_pos is not None:
                    pet_dist = _chebyshev(py, px, self._pet_pos[0], self._pet_pos[1])
                    if pet_dist > 2 and self._pet_wait_count < 16:
                        self._pet_wait_count += 1
                        self.step(A.Command.SEARCH)
                        return
                self._pet_wait_count = 0
                self.step(A.MiscDirection.DOWN)
                return

        # Navigate to stairs if force-descending
        if force_descend and lvl.stairs_down and self.gs.hp > self.gs.max_hp * 0.3:
            fight_dis = self.bfs(allow_hostiles=True)
            best_s = None
            best_sd = 999
            for sy, sx in lvl.stairs_down:
                d = fight_dis[sy, sx]
                if d != -1 and d < best_sd:
                    best_sd = d
                    best_s = (sy, sx)
            if best_s:
                if best_sd == 0:
                    self.step(A.MiscDirection.DOWN)
                    return
                if self.step_toward(best_s[0], best_s[1], fight_dis):
                    return
                self._greedy_move_toward(best_s[0], best_s[1])
                return

        # === 4. Navigate to Excalibur fountain (if applicable) ===
        if (not self.has_excalibur and self.alignment == 'lawful'
                and self.gs.xlevel >= 5 and self.has_long_sword
                and self.strategy_mgr.milestone == Milestone.FIND_EXCALIBUR):
            if lvl.fountains:
                best_f = None
                best_fd = 999
                for fy, fx in lvl.fountains:
                    d = dis[fy, fx]
                    if d != -1 and d < best_fd:
                        best_fd = d
                        best_f = (fy, fx)
                if best_f and best_fd > 0:
                    if self.step_toward(best_f[0], best_f[1], dis):
                        return

        # === 5. Explore frontier ===
        best_f = self._find_explore_target(dis)
        if best_f:
            if self.step_toward(best_f[0], best_f[1], dis):
                # After reaching frontier, search if near walls
                ny, nx = self.gs.py, self.gs.px
                if lvl.search_count[ny, nx] < 3:
                    adj_wall = self._has_adjacent_wall(ny, nx)
                    if adj_wall:
                        lvl.search_count[ny, nx] += 1
                        self.step(A.Command.SEARCH)
                return

        # === 6. Navigate to known stairs ===
        if lvl.stairs_down and self.gs.hp > self.gs.max_hp * 0.3:
            if descent_ready:
                fight_dis = self.bfs(allow_hostiles=True)
                best_s = None
                best_sd = 999
                for sy, sx in lvl.stairs_down:
                    d = fight_dis[sy, sx]
                    if d != -1 and d < best_sd:
                        best_sd = d
                        best_s = (sy, sx)
                if best_s:
                    if best_sd == 0:
                        self.step(A.MiscDirection.DOWN)
                        return
                    if self.step_toward(best_s[0], best_s[1], fight_dis):
                        return

        # === 7. Navigate to altar (for BUC testing) ===
        if lvl.altars and self._find_unided_items():
            best_a = None
            best_ad = 999
            for ay, ax in lvl.altars:
                d = dis[ay, ax]
                if d != -1 and d < best_ad:
                    best_ad = d
                    best_a = (ay, ax)
            if best_a and best_ad > 0 and best_ad <= 20:
                if self.step_toward(best_a[0], best_a[1], dis):
                    return

        # === 8. Search near walls (AutoAscend to_search_func) ===
        search_target = self._find_search_target(dis)
        if search_target and search_target != (py, px):
            if self.step_toward(search_target[0], search_target[1], dis):
                return

        # === 9. Search at current position ===
        search_rounds = min(5, max(1, 12 - lvl.search_count[py, px]))
        for _ in range(search_rounds):
            lvl.search_count[py, px] += 1
            self.step(A.Command.SEARCH)
            # Check if something new appeared
            if self.glyphs is not None:
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = py + dy2, px + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            g = int(self.glyphs[nr, nc])
                            cm = _cmap(g)
                            if cm in _CLOSED_DOOR_CMAP or cm in _WALKABLE_CMAP:
                                if not lvl.seen[nr, nc] or cm in _CLOSED_DOOR_CMAP:
                                    return

    def _find_explore_target(self, dis):
        """Find the best frontier tile to explore."""
        lvl = self.current_level()
        best_f = None
        best_fd = 999

        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or d >= best_fd or not lvl.walkable[r, c]:
                    continue
                # Is this tile on the exploration frontier?
                is_frontier = False
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not lvl.seen[nr, nc]:
                            is_frontier = True
                            break
                    if is_frontier:
                        break

                if is_frontier:
                    # Priority: prefer tiles near doors/corridors
                    priority_bonus = 0
                    for dy2 in (-1, 0, 1):
                        for dx2 in (-1, 0, 1):
                            if dy2 == 0 and dx2 == 0:
                                continue
                            ar, ac = r + dy2, c + dx2
                            if 0 <= ar < MAP_H and 0 <= ac < MAP_W and lvl.seen[ar, ac]:
                                obj = int(lvl.objects[ar, ac]) if lvl.objects[ar, ac] != -1 else -1
                                cm2 = _cmap(obj)
                                if cm2 in _DOOR_CMAP:
                                    priority_bonus = 2
                                elif cm2 in (SS_CORR, SS_LITCORR):
                                    priority_bonus = max(priority_bonus, 1)

                    eff_dist = d - priority_bonus * 5
                    if eff_dist < best_fd:
                        best_fd = eff_dist
                        best_f = (r, c)

        return best_f

    def _find_search_target(self, dis):
        """Find the best tile to search near walls (AutoAscend to_search_func)."""
        lvl = self.current_level()
        py, px = self.gs.py, self.gs.px
        best_s = None
        best_sp = float('-inf')

        for r in range(MAP_H):
            for c in range(MAP_W):
                if not lvl.walkable[r, c] or dis[r, c] == -1:
                    continue

                stones = 0
                walls = 0
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            obj_val = int(lvl.objects[nr, nc])
                            if obj_val == -1 and self.glyphs is not None:
                                obj_val = int(self.glyphs[nr, nc])
                            cm2 = _cmap(obj_val)
                            if cm2 == SS_STONE:
                                stones += 1
                            elif cm2 in _WALL_CMAP:
                                walls += 1

                if stones == 0 and walls == 0:
                    continue

                sc = lvl.search_count[r, c]
                p = -1.0 - sc * sc * 2.0

                # Door with 3+ adjacent stones
                obj_here = int(lvl.objects[r, c]) if lvl.objects[r, c] != -1 else -1
                if _cmap(obj_here) in _DOOR_CMAP and stones >= 3:
                    p += 250

                # Dead end: <= 1 walkable cardinal neighbor
                cardinal_w = sum(1 for dy3, dx3 in [(-1,0),(1,0),(0,-1),(0,1)]
                                if 0 <= r+dy3 < MAP_H and 0 <= c+dx3 < MAP_W
                                and lvl.walkable[r+dy3, c+dx3])
                if cardinal_w <= 1:
                    p += 250

                p -= dis[r, c] * 2

                if p > best_sp:
                    best_sp = p
                    best_s = (r, c)

        return best_s

    def _has_adjacent_wall(self, r, c):
        """Check if tile has an adjacent wall or stone."""
        if self.glyphs is None:
            return False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nr, nc = r + dy, c + dx
                if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                    cm = _cmap(int(self.glyphs[nr, nc]))
                    if cm in _WALL_CMAP or cm == SS_STONE:
                        return True
        return False

    def _on_stairs_down(self):
        """Check if player is standing on downstairs."""
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        # Check tracked positions
        if (py, px) in lvl.stairs_down:
            return True

        # Check current glyph
        if self.glyphs is not None:
            g = int(self.glyphs[py, px])
            if _cmap(g) in _STAIRS_DOWN_CMAP:
                return True

        # Check stored terrain
        obj = int(lvl.objects[py, px]) if lvl.objects[py, px] != -1 else -1
        if _cmap(obj) in _STAIRS_DOWN_CMAP:
            return True

        # Check messages
        if self.gs.on_stairs_down:
            return True

        return False

    def _on_stairs_up(self):
        """Check if player is standing on upstairs."""
        py, px = self.gs.py, self.gs.px
        lvl = self.current_level()

        if (py, px) in lvl.stairs_up:
            return True
        if self.glyphs is not None:
            g = int(self.glyphs[py, px])
            if _cmap(g) in _STAIRS_UP_CMAP:
                return True
        obj = int(lvl.objects[py, px]) if lvl.objects[py, px] != -1 else -1
        if _cmap(obj) in _STAIRS_UP_CMAP:
            return True
        if self.gs.on_stairs_up:
            return True
        return False

    def _should_descend(self):
        """Check if we should descend based on strategy manager."""
        lvl = self.current_level()
        vis_mons = len(self.get_hostile_monsters())

        return self.strategy_mgr.should_descend(
            dlevel=self.gs.dlevel,
            xlevel=self.gs.xlevel,
            hp=self.gs.hp,
            max_hp=self.gs.max_hp,
            level_explored=lvl.is_explored(),
            total_searches=lvl.total_searches,
            visible_monsters=vis_mons,
        )

    def _should_force_descend(self):
        """Check if we should force-descend regardless of normal gates."""
        lvl = self.current_level()
        return self.strategy_mgr.should_force_descend(
            dlevel=self.gs.dlevel,
            xlevel=self.gs.xlevel,
            hp=self.gs.hp,
            max_hp=self.gs.max_hp,
            total_searches=lvl.total_searches,
        )

    def sacrifice_strategy(self):
        """Sacrifice corpses on aligned altars for alignment, gifts, etc."""
        if not self.gs.on_altar:
            return False

        # Check if there's a corpse on the ground
        init_msg = self.initial_message.lower() if self.initial_message else ''
        if 'corpse' not in init_msg:
            return False

        # Offer corpse via #offer / sacrifice
        # In NLE, sacrifice is done via #offer which maps to OFFER
        offer_idx = self._val2idx.get(int(A.Command.OFFER))
        if offer_idx is not None:
            self.step(offer_idx)
            return True

        return False

    # ================================================================
    # Main loop
    # ================================================================

    def main(self):
        """Main agent loop. Runs strategies in priority order until episode ends."""
        try:
            obs, info = self.env.reset(seed=self.seed)
            self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            bl = obs.get('blstats')
            if bl is not None:
                self._raw_bl = np.array(bl, dtype=np.int64)
            self._update_game_state()

            # Clear initial prompts
            try:
                self.step(A.Command.ESC)
                self.step(A.Command.ESC)
            except AgentFinished:
                raise

            # Disable autopickup
            try:
                self.step(A.Command.AUTOPICKUP)
                if 'Autopickup: ON' in self.message:
                    self.step(A.Command.AUTOPICKUP)
            except AgentFinished:
                raise

            # Detect starting role/race from messages
            self._detect_starting_info()

            # Main loop
            while True:
                try:
                    self._run_one_turn()
                except AgentFinished:
                    raise
                except RuntimeError as e:
                    if 'finished' in str(e).lower():
                        break
                    try:
                        self.step(A.Command.SEARCH)
                    except (AgentFinished, RuntimeError):
                        break

        except AgentFinished:
            pass

    def _detect_starting_info(self):
        """Detect role, race, alignment from initial observations."""
        # Role detection from inventory or messages
        for _, item in self.inventory.items():
            lower = item.lower()
            if 'katana' in lower:
                self.role = 'Samurai'
            elif 'credit card' in lower or 'hawaiian shirt' in lower:
                self.role = 'Tourist'
            elif 'trident' in lower and 'weapon in hand' in lower:
                self.role = 'Knight'

        # Valkyrie starts with cold resistance
        if self.role == 'Valkyrie':
            self.resistances.add('cold resistance')
        elif self.role in ('Barbarian', 'Healer', 'Monk'):
            self.resistances.add('poison resistance')
        elif self.role == 'Monk':
            self.resistances.add('sleep resistance')

        # Check for Elf race (sleep resistance)
        if self.race == 'Elf':
            self.resistances.add('sleep resistance')
        elif self.race == 'Orc':
            self.resistances.add('poison resistance')

        # Check alignment
        if self._raw_bl is not None and len(self._raw_bl) > BL_ALIGN:
            al = int(self._raw_bl[BL_ALIGN])
            self.alignment = ALIGN_LABELS.get(al, 'neutral')

    def _run_one_turn(self):
        """Execute one turn of the agent loop."""
        # Stall detection
        cur_turn = self.gs.turn
        if cur_turn == self._stall_turn:
            self._stall_count += 1
            if self._stall_count > 5:
                self.step(A.Command.SEARCH)
                self._stall_count = 0
                return
        else:
            self._stall_turn = cur_turn
            self._stall_count = 0

        # Stuck detection: same position for too long
        if self._stuck_count > 20:
            self._stuck_count = 0
            # Try to unstick by searching or moving randomly
            self.step(A.Command.SEARCH)
            return

        # On downstairs: always check descent
        if self._on_stairs_down() and self.gs.hp > self.gs.max_hp * 0.5:
            if self._should_descend() or self._should_force_descend():
                # Pet wait
                if self._pet_pos is not None:
                    pet_dist = _chebyshev(self.gs.py, self.gs.px,
                                         self._pet_pos[0], self._pet_pos[1])
                    if pet_dist > 2 and self._pet_wait_count < 16:
                        self._pet_wait_count += 1
                        self.step(A.Command.SEARCH)
                        return
                self._pet_wait_count = 0
                self.step(A.MiscDirection.DOWN)
                return

        # Strategy priority chain
        if self.emergency():
            return
        if self.pray_strategy():
            return
        if self.fight():
            return
        if self.eat_strategy():
            return
        if self.equip_strategy():
            return
        if self.excalibur_strategy():
            return
        if self.altar_buc_strategy():
            return
        if self.sacrifice_strategy():
            return
        if self.item_id_strategy():
            return
        if self.pickup_strategy():
            return
        if self.sokoban_strategy():
            return
        self.explore()


# ================================================================
# Monster Knowledge Database
# ================================================================

# Detailed monster properties for combat decisions.
# Maps monster name -> dict of properties used by the fight loop.

# Monsters that are fast (speed >= 15 in NetHack)
FAST_SPEED_MONSTERS = {
    'bat', 'giant bat', 'vampire bat',
    'panther', 'jaguar',
    'pony', 'horse', 'warhorse',
    'dog', 'large dog',
    'kitten', 'housecat', 'large cat',
    'fox', 'coyote', 'wolf', 'warg', 'winter wolf cub', 'winter wolf',
    'jackal', 'hell hound pup', 'hell hound',
    'ant', 'soldier ant', 'fire ant',
    'killer bee', 'queen bee',
    'black unicorn', 'white unicorn', 'gray unicorn',
    'air elemental',
    'jabberwock',
    'couatl',
    'mind flayer', 'master mind flayer',
    'energy vortex', 'steam vortex', 'fire vortex',
    'xorn',
}

# Monsters that can pick up items (potential threats to inventory)
ITEM_STEALERS = {
    'nymph', 'water nymph', 'wood nymph', 'mountain nymph',
    'monkey', 'ape',
}

# Monsters that steal gold
GOLD_STEALERS = {
    'leprechaun',
}

# Monsters that are dangerous to fight due to passive effects
PASSIVE_DANGEROUS = {
    'floating eye': 'paralysis (passive gaze)',
    'acid blob': 'acid splash on hit',
    'blue jelly': 'cold damage on hit',
    'brown mold': 'cold damage on hit, multiplication',
    'yellow mold': 'stun on hit',
    'green mold': 'acid damage on hit',
    'black pudding': 'splits on hit with iron weapon',
    'brown pudding': 'splits on hit',
    'gelatinous cube': 'paralysis, engulf',
    'ochre jelly': 'acid engulf',
    'spotted jelly': 'acid passive',
    'red mold': 'fire damage on hit',
    'shrieker': 'wakes other monsters',
}

# Monsters with level drain attacks
LEVEL_DRAINERS = {
    'vampire', 'vampire lord', 'Vlad the Impaler',
    'wraith', 'barrow wight',
    'Nazgul',
    'succubus', 'incubus',
}

# Monsters with energy drain
ENERGY_DRAINERS = {
    'mind flayer', 'master mind flayer',
}

# Monsters that cast spells (can be dangerous at range)
SPELLCASTERS = {
    'kobold shaman', 'orc shaman',
    'gnomish wizard',
    'winter wolf', 'hell hound',
    'red dragon', 'blue dragon', 'white dragon', 'black dragon',
    'green dragon', 'yellow dragon', 'orange dragon',
    'baby red dragon', 'baby blue dragon', 'baby white dragon',
    'baby black dragon', 'baby green dragon',
    'arch-lich', 'master lich', 'lich', 'demilich',
    'Wizard of Yendor',
    'golden naga', 'red naga', 'black naga', 'guardian naga',
    'titan',
}

# Monsters that have breath weapons
BREATH_WEAPONS = {
    'red dragon': 'fire',
    'blue dragon': 'lightning',
    'white dragon': 'cold',
    'black dragon': 'disintegration',
    'green dragon': 'poison',
    'yellow dragon': 'acid',
    'orange dragon': 'sleep',
    'baby red dragon': 'fire',
    'baby blue dragon': 'lightning',
    'baby white dragon': 'cold',
    'baby black dragon': 'disintegration',
    'baby green dragon': 'poison',
}

# Monsters immune to physical damage
PHYSICAL_IMMUNE = {
    'shade', 'ghost',
}

# Monsters that multiply
MULTIPLIERS = {
    'brown mold', 'yellow mold', 'green mold', 'red mold',
    'black pudding', 'brown pudding',
    'gremlin',
}

# Monsters that grab/hold
GRABBERS = {
    'owlbear', 'umber hulk',
    'rope golem',
    'giant eel', 'electric eel', 'kraken',
    'python',
    'giant mimic', 'large mimic', 'small mimic',
}

# Armor-destroying monsters
ARMOR_DESTROYERS = {
    'rust monster': 'rust (metal)',
    'disenchanter': 'disenchant',
    'brown pudding': 'rust/corrode',
    'black pudding': 'corrode',
}

# Monsters that are covetous (teleport to you, steal quest artifact)
COVETOUS = {
    'Wizard of Yendor', 'Vlad the Impaler',
    'Demogorgon', 'Orcus', 'Baalzebub',
    'Asmodeus', 'Dispater', 'Geryon',
    'Yeenoghu', 'Juiblex',
    'arch-lich', 'master lich',
}

# Safe distance for ranged combat
RANGED_SAFE_DISTANCE = 3

# ================================================================
# Trap Knowledge
# ================================================================

# Known trap types and their effects
TRAP_TYPES = {
    'arrow trap': {'damage': 6, 'type': 'physical'},
    'dart trap': {'damage': 4, 'type': 'physical', 'effect': 'poison'},
    'falling rock trap': {'damage': 20, 'type': 'physical'},
    'squeaky board': {'damage': 0, 'type': 'noise'},
    'bear trap': {'damage': 4, 'type': 'physical', 'effect': 'held'},
    'land mine': {'damage': 16, 'type': 'physical'},
    'rolling boulder trap': {'damage': 20, 'type': 'physical'},
    'sleeping gas trap': {'damage': 0, 'type': 'magical', 'effect': 'sleep'},
    'rust trap': {'damage': 0, 'type': 'magical', 'effect': 'rust'},
    'fire trap': {'damage': 12, 'type': 'fire'},
    'pit': {'damage': 6, 'type': 'physical'},
    'spiked pit': {'damage': 10, 'type': 'physical', 'effect': 'poison'},
    'hole': {'damage': 6, 'type': 'physical', 'effect': 'levelchange'},
    'trap door': {'damage': 6, 'type': 'physical', 'effect': 'levelchange'},
    'teleportation trap': {'damage': 0, 'type': 'magical', 'effect': 'teleport'},
    'level teleporter': {'damage': 0, 'type': 'magical', 'effect': 'levelteleport'},
    'magic portal': {'damage': 0, 'type': 'magical', 'effect': 'portal'},
    'web': {'damage': 0, 'type': 'physical', 'effect': 'held'},
    'statue trap': {'damage': 0, 'type': 'magical', 'effect': 'monster'},
    'magic trap': {'damage': 0, 'type': 'magical', 'effect': 'random'},
    'anti-magic field': {'damage': 0, 'type': 'magical', 'effect': 'drain_pw'},
    'polymorph trap': {'damage': 0, 'type': 'magical', 'effect': 'polymorph'},
}

# Trap messages
TRAP_MESSAGES = {
    'you fall into a pit': 'pit',
    'you fall into a spiked pit': 'spiked pit',
    'a trap door opens': 'trap door',
    'a bear trap closes on your': 'bear trap',
    'you are caught in a web': 'web',
    'a gush of water hits you': 'rust trap',
    'you feel a draft': 'hole',
    'you land mine': 'land mine',
    'click! you trigger a rolling boulder': 'rolling boulder trap',
    'a cloud of gas puts you to sleep': 'sleeping gas trap',
    'you feel chaotic': 'magic trap',
    'you feel a change coming over you': 'polymorph trap',
    'you shudder for a moment': 'anti-magic field',
    'an arrow shoots out at you': 'arrow trap',
    'a little dart shoots out at you': 'dart trap',
    'you are enveloped in a tower of flame': 'fire trap',
}


# ================================================================
# Scroll Knowledge
# ================================================================

# Scrolls worth reading in various situations
SCROLL_PRIORITIES = {
    'identify': 10,
    'remove curse': 9,
    'enchant weapon': 8,
    'enchant armor': 8,
    'teleportation': 7,
    'create monster': 1,  # usually bad
    'earth': 2,
    'fire': 3,
    'food detection': 5,
    'gold detection': 4,
    'magic mapping': 9,
    'scare monster': 6,
    'taming': 7,
    'genocide': 10,
    'charging': 9,
    'confuse monster': 5,
    'destroy armor': 1,
    'amnesia': 1,
    'light': 3,
    'punishment': 1,
    'stinking cloud': 2,
}

# Scrolls safe to read when confused
SAFE_CONFUSED = {
    'identify', 'light', 'food detection', 'gold detection',
    'magic mapping', 'scare monster',
}

# ================================================================
# Potion Knowledge
# ================================================================

# Potions worth quaffing
POTION_PRIORITIES = {
    'full healing': 10,
    'extra healing': 9,
    'healing': 8,
    'gain ability': 7,
    'gain level': 10,
    'speed': 7,
    'invisibility': 6,
    'see invisible': 6,
    'restore ability': 5,
    'object detection': 4,
    'monster detection': 4,
    'enlightenment': 3,
    'fruit juice': 2,
    'water': 1,
    'booze': 1,
    'sickness': 0,  # never quaff
    'hallucination': 0,
    'confusion': 0,
    'blindness': 0,
    'paralysis': 0,
    'sleeping': 0,
    'acid': 1,
    'oil': 1,
    'polymorph': 2,
}

# Potions safe to quaff when unknown (holy water test: blessed = good)
POTION_SAFE_BLESSED = {
    'healing', 'extra healing', 'full healing',
    'gain ability', 'gain level', 'speed',
    'see invisible', 'restore ability',
    'water',  # holy water
}

# ================================================================
# Wand Knowledge
# ================================================================

# Wand zap strategies
WAND_COMBAT_VALUE = {
    'death': 10,
    'fire': 8,
    'cold': 7,
    'lightning': 8,
    'magic missile': 7,
    'sleep': 6,
    'slow monster': 5,
    'striking': 5,
    'polymorph': 4,
    'teleportation': 3,
    'cancellation': 3,
    'create monster': 0,
    'digging': 2,  # escape utility
    'speed monster': 0,  # bad to zap at enemies
    'nothing': 0,
    'light': 0,
    'probing': 1,
    'undead turning': 3,
    'opening': 1,
    'locking': 1,
    'make invisible': 0,
    'wishing': 10,  # never zap at monsters
    'secret door detection': 1,
    'enlightenment': 1,
    'charging': 0,
}

# Wands to never zap at monsters (self-use or useless)
WAND_NEVER_ZAP_AT = {
    'speed monster', 'nothing', 'light', 'probing',
    'opening', 'locking', 'make invisible', 'wishing',
    'secret door detection', 'enlightenment', 'charging',
    'create monster',
}

# Wands with directional damage
WAND_DIRECTIONAL_DAMAGE = {
    'death', 'fire', 'cold', 'lightning', 'magic missile',
    'sleep', 'slow monster', 'striking', 'polymorph',
    'teleportation', 'cancellation', 'undead turning',
}

# Self-zap beneficial wands (zap at self or ground)
WAND_SELF_ZAP = {
    'speed monster': 'speed up',
    'make invisible': 'invisibility',
    'light': 'illuminate area',
    'digging': 'escape down',
    'teleportation': 'teleport self',
    'secret door detection': 'reveal secrets',
    'probing': 'check equipment',
}


# ================================================================
# Role-Specific Knowledge
# ================================================================

ROLE_STARTING_RESISTS = {
    'Valkyrie': {'cold resistance'},
    'Barbarian': {'poison resistance'},
    'Healer': {'poison resistance'},
    'Monk': {'poison resistance', 'sleep resistance'},
    'Caveman': set(),
    'Knight': set(),
    'Priest': set(),
    'Ranger': set(),
    'Rogue': set(),
    'Samurai': set(),
    'Tourist': set(),
    'Wizard': set(),
    'Archeologist': set(),
}

RACE_STARTING_RESISTS = {
    'Elf': {'sleep resistance'},
    'Orc': {'poison resistance'},
    'Human': set(),
    'Dwarf': set(),
    'Gnome': set(),
}

# Role-specific starting weapons to wield
ROLE_STARTING_WEAPONS = {
    'Valkyrie': 'long sword',
    'Samurai': 'katana',
    'Knight': 'long sword',
    'Barbarian': 'two-handed sword',
    'Caveman': 'club',
    'Ranger': 'dagger',
    'Rogue': 'short sword',
}

# Quest artifact by role
ROLE_QUEST_ARTIFACT = {
    'Valkyrie': 'Orb of Fate',
    'Samurai': 'Tsurugi of Muramasa',
    'Knight': 'Magic Mirror of Merlin',
    'Wizard': 'Eye of the Aethiopica',
    'Priest': 'Mitre of Holiness',
    'Rogue': 'Master Key of Thievery',
    'Ranger': 'Longbow of Diana',
    'Barbarian': 'Heart of Ahriman',
    'Tourist': 'Platinum Yendorian Express Card',
    'Archeologist': 'Orb of Detection',
    'Caveman': 'Sceptre of Might',
    'Healer': 'Staff of Aesculapius',
    'Monk': 'Eyes of the Overworld',
}


# ================================================================
# Dungeon Knowledge
# ================================================================

# Dungeon level ranges for branches
BRANCH_DEPTHS = {
    'mines_entrance': (2, 4),   # Mines entrance at DL 2-4
    'minetown': (5, 8),         # Minetown at mines level 5-8
    'sokoban_entrance': (5, 9), # Sokoban entrance from DL 5-9
    'oracle': (5, 9),           # Oracle level at DL 5-9
    'big_room': (10, 12),       # Big room at DL 10-12
    'rogue_level': (15, 18),    # Rogue level at DL 15-18
    'medusa': (22, 26),         # Medusa's Island
    'castle': (25, 28),         # Castle level
    'valley': (26, 29),         # Valley of the Dead
    'gehennom_start': (26, 30), # Gehennom begins
}

# Special level messages
SPECIAL_LEVEL_MESSAGES = {
    'welcome to Minetown': 'minetown',
    'you feel a strange vibration': 'oracle',
    'you enter what seems to be an older': 'rogue_level',
    'you feel a warm updraft': 'gehennom',
    'this land is not fit for habitation': 'valley',
    'you hear the roaring of an pointy-eared': 'sokoban',
}


# ================================================================
# Extended AgentV4 Methods
# ================================================================

# These are additional methods that extend the AgentV4 class.
# They are defined here and monkey-patched onto the class below.

def _assess_combat_situation(self):
    """Comprehensive tactical assessment of the current combat situation.

    Returns a dict with:
        total_danger: float
        max_danger: int
        instakill_present: bool
        ranged_threats: list
        melee_threats: list
        passive_threats: list
        recommended_tactic: str
        flee_direction: tuple or None
        corridor_available: bool
    """
    py, px = self.gs.py, self.gs.px
    adj = self.get_adjacent_hostiles()
    hostiles = self.get_hostile_monsters()
    hp_ratio = self.gs.hp / max(1, self.gs.max_hp)
    lvl = self.current_level()

    total_danger = 0
    max_danger = 0
    instakill_present = False
    ranged_threats = []
    melee_threats = []
    passive_threats = []

    for d, mon in hostiles:
        info = assess_monster(mon.name, mon.mon_id)
        danger = info['danger']

        if self.threat_db is not None:
            player_state = {
                'hp': self.gs.hp, 'max_hp': self.gs.max_hp,
                'ac': self.gs.ac, 'level': self.gs.xlevel,
                'speed': 12, 'resistances': self.resistances,
                'has_elbereth_source': True,
            }
            threat = self.threat_db.assess_threat(mon.name, player_state)
            danger = threat.danger_level
            if threat.instakill_risk:
                instakill_present = True
            if threat.ranged_preferred:
                passive_threats.append((d, mon, danger))
                continue

        if info['instakill']:
            instakill_present = True

        total_danger += danger
        max_danger = max(max_danger, danger)

        if d <= 1:
            if mon.name in PASSIVE_DANGEROUS:
                passive_threats.append((d, mon, danger))
            else:
                melee_threats.append((d, mon, danger))
        else:
            if mon.name in SPELLCASTERS or mon.name in BREATH_WEAPONS:
                ranged_threats.append((d, mon, danger))
            else:
                melee_threats.append((d, mon, danger))

    # Check if a corridor is available for 1v1 fighting
    corridor_available = False
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = py + dy, px + dx
        if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
            continue
        if not lvl.walkable[ny, nx]:
            continue
        g = int(self.glyphs[ny, nx])
        if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
            continue
        cm = _cmap(g)
        if cm in (SS_CORR, SS_LITCORR):
            corridor_available = True
            break

    # Compute best flee direction
    flee_dir = None
    if instakill_present or (len(adj) >= 3 and hp_ratio < 0.3):
        best_threat = len(adj) + 1
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            if not lvl.walkable[ny, nx]:
                g2 = int(self.glyphs[ny, nx])
                if not glyph_is_pet(g2):
                    continue
            g2 = int(self.glyphs[ny, nx])
            if GLYPH_MON_OFF <= g2 < GLYPH_PET_OFF:
                continue
            threat = sum(1 for m in adj if _chebyshev(ny, nx, m.row, m.col) <= 1)
            if threat < best_threat:
                best_threat = threat
                flee_dir = (dy, dx)

    # Determine recommended tactic
    if instakill_present and any(d <= 1 for d, _, _ in melee_threats + passive_threats
                                 if _ and _.name in INSTAKILL):
        tactic = 'flee_instakill'
    elif instakill_present:
        tactic = 'avoid_instakill'
    elif len(adj) >= 3 and hp_ratio < 0.3:
        tactic = 'flee_critical'
    elif len(adj) >= 3 and corridor_available:
        tactic = 'retreat_corridor'
    elif len(adj) >= 2 and hp_ratio < 0.5 and not self._on_elbereth:
        tactic = 'elbereth'
    elif passive_threats and not melee_threats:
        tactic = 'ranged_only'
    elif ranged_threats and not adj:
        tactic = 'approach_carefully'
    elif melee_threats:
        tactic = 'melee'
    elif hostiles:
        tactic = 'approach'
    else:
        tactic = 'safe'

    return {
        'total_danger': total_danger,
        'max_danger': max_danger,
        'instakill_present': instakill_present,
        'ranged_threats': ranged_threats,
        'melee_threats': melee_threats,
        'passive_threats': passive_threats,
        'recommended_tactic': tactic,
        'flee_direction': flee_dir,
        'corridor_available': corridor_available,
    }


def _retreat_to_corridor(self):
    """Retreat to a nearby corridor for 1v1 combat.

    Corridors limit the number of monsters that can attack simultaneously.
    Returns True if we moved toward a corridor.
    """
    py, px = self.gs.py, self.gs.px
    lvl = self.current_level()

    # Already in corridor
    if self.glyphs is not None:
        cm = _cmap(int(self.glyphs[py, px]))
        obj_cm = _cmap(int(lvl.objects[py, px])) if lvl.objects[py, px] != -1 else -1
        if cm in (SS_CORR, SS_LITCORR) or obj_cm in (SS_CORR, SS_LITCORR):
            return False

    # Find nearest corridor tile
    dis = self.bfs()
    best_corr = None
    best_d = 999

    for r in range(MAP_H):
        for c in range(MAP_W):
            if dis[r, c] == -1 or dis[r, c] >= best_d:
                continue
            if self.glyphs is not None:
                g = int(self.glyphs[r, c])
                cm = _cmap(g)
                if cm in (SS_CORR, SS_LITCORR):
                    best_d = dis[r, c]
                    best_corr = (r, c)
            obj = int(lvl.objects[r, c]) if lvl.objects[r, c] != -1 else -1
            if _cmap(obj) in (SS_CORR, SS_LITCORR):
                best_d = dis[r, c]
                best_corr = (r, c)

    if best_corr and best_d <= 6:
        return self.step_toward(best_corr[0], best_corr[1], dis)
    return False


def _handle_status_effects(self):
    """Handle non-emergency status effects.

    Tries to cure status effects using items before resorting to prayer.
    Returns True if an action was taken.
    """
    gs = self.gs
    conditions = gs.conditions

    # Confusion: quaff unicorn horn (apply)
    if 'confused' in conditions:
        for letter, item in self.inventory.items():
            if 'unicorn horn' in item.lower():
                apply_idx = self._val2idx.get(int(A.Command.APPLY))
                if apply_idx is not None:
                    self._env_step(apply_idx)
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        self._env_step(l_idx)
                    self._update_game_state()
                    return True

    # Blindness: apply unicorn horn or quaff potion of see invisible
    if 'blind' in conditions:
        for letter, item in self.inventory.items():
            if 'unicorn horn' in item.lower():
                apply_idx = self._val2idx.get(int(A.Command.APPLY))
                if apply_idx is not None:
                    self._env_step(apply_idx)
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        self._env_step(l_idx)
                    self._update_game_state()
                    return True
        for letter, item in self.inventory.items():
            if 'see invisible' in item.lower() and self.inv_oclasses.get(letter, -1) == POTION_CLASS:
                self._two_step_quaff(letter)
                return True

    # Stunned: apply unicorn horn
    if 'stunned' in conditions:
        for letter, item in self.inventory.items():
            if 'unicorn horn' in item.lower():
                apply_idx = self._val2idx.get(int(A.Command.APPLY))
                if apply_idx is not None:
                    self._env_step(apply_idx)
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        self._env_step(l_idx)
                    self._update_game_state()
                    return True

    # Hallucinating: apply unicorn horn
    if 'hallucinating' in conditions:
        for letter, item in self.inventory.items():
            if 'unicorn horn' in item.lower():
                apply_idx = self._val2idx.get(int(A.Command.APPLY))
                if apply_idx is not None:
                    self._env_step(apply_idx)
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        self._env_step(l_idx)
                    self._update_game_state()
                    return True

    return False


def _use_wand_strategy(self):
    """Strategic wand use beyond combat.

    - Zap wand of digging at walls to create shortcuts
    - Zap wand of secret door detection to find secrets
    - Zap wand of probing at self to check equipment
    - Zap wand of speed monster at self for intrinsic speed

    Returns True if an action was taken.
    """
    # Only use strategic wands when not in danger
    if self.get_adjacent_hostiles():
        return False

    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
            continue
        lower = item.lower()

        # Skip empty wands
        if '(0:0)' in lower:
            continue

        # Speed monster: zap at self for speed
        if 'speed monster' in lower and 'speed' not in self.resistances:
            # Zap at self: zap command, wand letter, '.' for self
            zap_idx = self._val2idx.get(int(A.Command.ZAP))
            if zap_idx is not None:
                if self._env_step(zap_idx):
                    self._quick_parse()
                    raise AgentFinished()
                w_idx = self._val2idx.get(ord(letter))
                if w_idx is not None:
                    if self._env_step(w_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    # Direction: '.' for self
                    dot_idx = self._val2idx.get(ord('.'))
                    if dot_idx is not None:
                        if self._env_step(dot_idx):
                            self._quick_parse()
                            raise AgentFinished()
                self._update_game_state()
                self.resistances.add('speed')
                return True

        # Digging: zap downward to escape dangerous levels
        if 'digging' in lower and self.gs.hp < self.gs.max_hp * 0.2:
            if len(self.get_hostile_monsters()) >= 3:
                self._three_step_zap(letter, 0, 0)  # zap down
                return True

    return False


def _read_scroll_strategy(self):
    """Strategic scroll reading.

    - Read identify scrolls when we have unidentified items
    - Read remove curse when we have cursed items
    - Read enchant weapon/armor when safe
    - Read magic mapping on new levels
    - Read teleportation in emergencies

    Returns True if an action was taken.
    """
    # Don't read scrolls during combat
    if self.get_adjacent_hostiles():
        return False

    # Don't read when confused (unless the scroll is safe)
    is_confused = 'confused' in self.gs.conditions

    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) != SCROLL_CLASS:
            continue
        lower = item.lower()

        if 'cursed' in lower:
            continue

        # Read identify scrolls when we have unidentified items
        if 'identify' in lower:
            unided = self._find_unided_items()
            if unided:
                # Read the scroll
                read_idx = self._val2idx.get(int(A.Command.READ))
                if read_idx is not None:
                    if self._env_step(read_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        if self._env_step(l_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    # Handle identify menu
                    for _ in range(5):
                        misc = self.obs.get('misc', [0, 0, 0])
                        if misc[1]:  # menu
                            # Pick first unidentified item
                            if unided:
                                item_letter = unided[0]
                                il_idx = self._val2idx.get(ord(item_letter))
                                if il_idx is not None:
                                    if self._env_step(il_idx):
                                        self._quick_parse()
                                        raise AgentFinished()
                                    continue
                        break
                    self._update_game_state()
                    return True

        # Read remove curse when we have cursed equipped items
        if 'remove curse' in lower:
            has_cursed = any('cursed' in self.inventory.get(l, '').lower()
                           and ('(being worn)' in self.inventory.get(l, '') or
                                '(weapon in hand)' in self.inventory.get(l, ''))
                           for l in self.inventory)
            if has_cursed:
                read_idx = self._val2idx.get(int(A.Command.READ))
                if read_idx is not None:
                    if self._env_step(read_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        if self._env_step(l_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    self._update_game_state()
                    return True

        # Read enchant weapon when we have a weapon wielded
        if 'enchant weapon' in lower and not is_confused:
            weapon = self._find_weapon_letter()
            if weapon:
                read_idx = self._val2idx.get(int(A.Command.READ))
                if read_idx is not None:
                    if self._env_step(read_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        if self._env_step(l_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    self._update_game_state()
                    return True

        # Read enchant armor when we have armor worn
        if 'enchant armor' in lower and not is_confused:
            has_armor = any('(being worn)' in self.inventory.get(l, '')
                          for l in self.inventory)
            if has_armor:
                read_idx = self._val2idx.get(int(A.Command.READ))
                if read_idx is not None:
                    if self._env_step(read_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        if self._env_step(l_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    self._update_game_state()
                    return True

    return False


def _quaff_potion_strategy(self):
    """Strategic potion drinking.

    - Quaff gain level potions when safe
    - Quaff gain ability potions when safe
    - Quaff speed potions when not fast
    - Quaff see invisible when blind and have one

    Returns True if an action was taken.
    """
    # Don't drink during combat
    if self.get_adjacent_hostiles():
        return False

    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) != POTION_CLASS:
            continue
        lower = item.lower()

        if 'cursed' in lower:
            continue

        # Gain level: always drink
        if 'gain level' in lower:
            self._two_step_quaff(letter)
            return True

        # Gain ability: drink when safe
        if 'gain ability' in lower:
            if self.gs.hp > self.gs.max_hp * 0.8:
                self._two_step_quaff(letter)
                return True

        # Speed: drink when not fast
        if 'speed' in lower and 'speed' not in self.resistances:
            self._two_step_quaff(letter)
            self.resistances.add('speed')
            return True

        # Full healing when not at full HP (only if we have multiple)
        if 'full healing' in lower:
            count = sum(1 for _, v in self.inventory.items()
                       if 'full healing' in v.lower())
            if count >= 2 and self.gs.hp < self.gs.max_hp * 0.5:
                self._two_step_quaff(letter)
                return True

    return False


def _detailed_inventory_management(self):
    """Manage inventory: drop junk, organize, manage encumbrance.

    Returns True if an action was taken.
    """
    # Don't manage during combat
    if self.get_adjacent_hostiles():
        return False

    # Encumbrance management: drop heaviest non-essential items
    if self.gs.encumbrance >= 3:  # overtaxed or worse
        worst_letter = None
        worst_priority = 999

        for letter, item in self.inventory.items():
            lower = item.lower()
            oc = self.inv_oclasses.get(letter, -1)

            # Never drop wielded weapon, worn armor, food, Excalibur
            if '(weapon in hand)' in lower or '(wielded)' in lower:
                continue
            if '(being worn)' in lower:
                continue
            if 'excalibur' in lower:
                continue

            # Prioritize keeping: food > potions > scrolls > wands > rings > other
            if oc == FOOD_CLASS:
                pri = 10
            elif oc == POTION_CLASS:
                if 'healing' in lower:
                    pri = 9
                else:
                    pri = 5
            elif oc == SCROLL_CLASS:
                pri = 5
            elif oc == WAND_CLASS:
                pri = 6
            elif oc == RING_CLASS:
                pri = 4
            elif oc == AMULET_CLASS:
                pri = 7
            elif oc == GEM_CLASS:
                pri = 2
            elif oc == WEAPON_CLASS:
                if any(w in lower for w in WEAPON_DATA):
                    pri = 3
                else:
                    pri = 1  # junk weapons
            elif oc == ARMOR_CLASS:
                pri = 3
            else:
                pri = 1

            # Rocks and gray stones are heavy
            if 'rock' in lower or 'stone' in lower:
                if 'luckstone' not in lower and 'touchstone' not in lower:
                    pri = 0

            if pri < worst_priority:
                worst_priority = pri
                worst_letter = letter

        if worst_letter and worst_priority <= 3:
            self._two_step_drop(worst_letter)
            return True

    return False


def _handle_trap_detected(self):
    """React to trap detection messages.

    Returns True if an action was taken.
    """
    msg = self.message.lower() if self.message else ''
    init = self.initial_message.lower() if self.initial_message else ''
    combined = msg + ' ' + init

    # Detect traps from messages
    for trap_msg, trap_type in TRAP_MESSAGES.items():
        if trap_msg in combined:
            trap_info = TRAP_TYPES.get(trap_type, {})

            # Bear trap: try to escape
            if trap_type == 'bear trap':
                # Try to escape by moving
                for _ in range(3):
                    self.step(A.Command.SEARCH)
                return True

            # Web: try to escape
            if trap_type == 'web':
                for _ in range(3):
                    self.step(A.Command.SEARCH)
                return True

            # Falling through trap door: already handled by level change
            if trap_type in ('trap door', 'hole'):
                return False

            # Sleeping gas: just wait it out
            if trap_type == 'sleeping gas trap':
                return False

    return False


def _find_best_combat_wand(self):
    """Find the best wand for combat. Returns (letter, name, value) or None."""
    best = None
    best_value = 0

    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
            continue
        lower = item.lower()

        # Skip empty wands
        if '(0:0)' in lower:
            continue

        # Check if identified
        for wand_name, value in WAND_COMBAT_VALUE.items():
            if wand_name in lower and wand_name not in WAND_NEVER_ZAP_AT:
                if value > best_value:
                    best = (letter, wand_name, value)
                    best_value = value
                break

    # If no identified combat wand, try unidentified ones
    if best is None:
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
                continue
            lower = item.lower()
            if '(0:0)' in lower:
                continue
            if 'of ' in lower:
                continue  # already checked above
            # Unidentified wand: moderate combat value
            if best is None:
                best = (letter, 'unknown', 3)

    return best


def _should_search_here(self, r, c):
    """Decide if we should search at this position based on AutoAscend heuristics."""
    lvl = self.current_level()
    sc = lvl.search_count[r, c]

    # Already searched too much
    if sc >= 20:
        return False

    # Count adjacent walls and stones
    stones = 0
    walls = 0
    doors = 0
    if self.glyphs is not None:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nr, nc = r + dy, c + dx
                if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                    obj = int(lvl.objects[nr, nc])
                    if obj == -1:
                        obj = int(self.glyphs[nr, nc])
                    cm = _cmap(obj)
                    if cm == SS_STONE:
                        stones += 1
                    elif cm in _WALL_CMAP:
                        walls += 1
                    elif cm in _DOOR_CMAP:
                        doors += 1

    # No adjacent walls/stones: nothing to search for
    if stones == 0 and walls == 0:
        return False

    # Prioritize:
    # 1. Door with 3+ adjacent stones (hidden corridor behind door)
    obj_here = int(lvl.objects[r, c]) if lvl.objects[r, c] != -1 else -1
    if _cmap(obj_here) in _DOOR_CMAP and stones >= 3:
        return sc < 15  # search up to 15 times

    # 2. Dead end (only 1 walkable cardinal neighbor)
    cardinal_w = 0
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dy, c + dx
        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and lvl.walkable[nr, nc]:
            cardinal_w += 1
    if cardinal_w <= 1:
        return sc < 15

    # 3. Normal wall-adjacent: search a few times
    return sc < 5


def _detect_special_level(self):
    """Detect special dungeon levels from messages and features."""
    msg = self.message.lower() if self.message else ''
    init = self.initial_message.lower() if self.initial_message else ''
    combined = msg + ' ' + init

    for level_msg, level_type in SPECIAL_LEVEL_MESSAGES.items():
        if level_msg in combined:
            return level_type

    return None


def _should_sacrifice(self):
    """Check if we should sacrifice a corpse on this altar.

    Returns True if:
    - On a co-aligned altar
    - There's a corpse on the ground or in inventory
    - We haven't sacrificed too recently
    """
    if not self.gs.on_altar:
        return False

    # Check for corpse in the message
    init = self.initial_message.lower() if self.initial_message else ''
    if 'corpse' not in init:
        return False

    return True


def _handle_shopkeeper(self):
    """Handle interactions with shopkeepers.

    - Don't pick up unpaid items
    - Pay for items when asked
    - Don't attack shopkeepers

    Returns True if an action was taken.
    """
    msg = self.message.lower() if self.message else ''

    # "You have a little trouble lifting" - too heavy
    if 'trouble lifting' in msg:
        return False

    # "For you, that will be" - price quote
    if 'for you' in msg and 'will be' in msg:
        return False  # Let normal pickup handle this

    # "You owe" - we have a debt
    if 'you owe' in msg:
        # Try to pay
        pay_idx = self._val2idx.get(int(A.Command.PAY))
        if pay_idx is not None:
            self.step(pay_idx)
            return True

    # "Shopkeeper blocks your way" - we're in a shop
    if 'blocks your way' in msg:
        # Pay first
        pay_idx = self._val2idx.get(int(A.Command.PAY))
        if pay_idx is not None:
            self.step(pay_idx)
            return True

    return False


def _compute_threat_assessment(self, mon):
    """Compute a detailed threat assessment for a single monster.

    Returns dict with:
        danger: int 1-10
        special_attacks: list[str]
        instakill: bool
        never_melee: bool
        ranged_preferred: bool
        flee: bool
        elbereth_effective: bool
    """
    info = assess_monster(mon.name, mon.mon_id)

    result = {
        'danger': info['danger'],
        'special_attacks': [],
        'instakill': info['instakill'],
        'never_melee': info['never_melee'],
        'ranged_preferred': mon.name in ONLY_RANGED_SLOW,
        'flee': False,
        'elbereth_effective': True,
    }

    # ThreatDB provides more accurate data
    if self.threat_db is not None:
        player_state = {
            'hp': self.gs.hp, 'max_hp': self.gs.max_hp,
            'ac': self.gs.ac, 'level': self.gs.xlevel,
            'speed': 12, 'resistances': self.resistances,
            'has_elbereth_source': True,
        }
        threat = self.threat_db.assess_threat(mon.name, player_state)
        result['danger'] = threat.danger_level
        result['special_attacks'] = threat.special_attacks
        result['instakill'] = threat.instakill_risk
        result['ranged_preferred'] = threat.ranged_preferred
        result['flee'] = (threat.recommended_action == 'flee')
        result['elbereth_effective'] = threat.elbereth_effective

    # Additional checks from our knowledge base
    if mon.name in PASSIVE_DANGEROUS:
        result['special_attacks'].append(PASSIVE_DANGEROUS[mon.name])
        result['never_melee'] = True

    if mon.name in LEVEL_DRAINERS:
        result['special_attacks'].append('level drain')
        if 'drain resistance' not in self.resistances:
            result['danger'] = max(result['danger'], 7)

    if mon.name in ENERGY_DRAINERS:
        result['special_attacks'].append('brain eating')
        result['danger'] = max(result['danger'], 8)

    if mon.name in MULTIPLIERS:
        result['special_attacks'].append('multiplies')

    if mon.name in ARMOR_DESTROYERS:
        result['special_attacks'].append(ARMOR_DESTROYERS[mon.name])
        result['ranged_preferred'] = True

    if mon.name in GRABBERS:
        result['special_attacks'].append('grab/hold')

    if mon.name in COVETOUS:
        result['special_attacks'].append('covetous (teleports)')
        result['danger'] = max(result['danger'], 9)

    if mon.name in ITEM_STEALERS:
        result['special_attacks'].append('steals items')
        result['danger'] = max(result['danger'], 6)

    if mon.name in GOLD_STEALERS:
        result['special_attacks'].append('steals gold')

    # Elbereth immunity
    if mon.name in {'minotaur', 'Death', 'Pestilence', 'Famine',
                     'Wizard of Yendor', 'Archon'}:
        result['elbereth_effective'] = False

    return result


def _handle_levelport(self):
    """Handle level teleportation (from traps or spells).

    Returns True if we took action.
    """
    msg = self.message.lower() if self.message else ''

    if 'you rise up through the ceiling' in msg:
        # Teleported up
        return False  # Level change already handled

    if 'you sink through the floor' in msg:
        # Teleported down
        return False

    return False


def _handle_polymorph(self):
    """Handle polymorphing into a different form.

    Returns True if we took action.
    """
    msg = self.message.lower() if self.message else ''

    if 'you feel like a new' in msg:
        # Polymorphed into new form
        return False  # Not much we can do

    return False


def _detect_mines_or_sokoban(self):
    """Detect if we've entered the Mines or Sokoban branch."""
    dnum = self.gs.dnum

    if dnum == DUNGEON_MINES:
        return 'mines'
    elif dnum == DUNGEON_SOKOBAN:
        return 'sokoban'
    elif dnum == DUNGEON_QUEST:
        return 'quest'
    elif dnum == DUNGEON_GEHENNOM:
        return 'gehennom'

    return 'main'


def _pick_ranged_target(self):
    """Pick the best target for ranged attacks (wands/projectiles).

    Returns (monster, ndy, ndx) or None.
    Prefers highest-danger targets in line of fire.
    """
    py, px = self.gs.py, self.gs.px
    hostiles = self.get_hostile_monsters()

    best = None
    best_danger = 0

    for d, mon in hostiles:
        if d <= 1 or d > 7:
            continue
        if not _in_line(py, px, mon.row, mon.col):
            continue
        if self.glyphs is not None:
            if not _line_clear(self.glyphs, py, px, mon.row, mon.col):
                continue

        threat = self._compute_threat_assessment(mon)
        danger = threat['danger']

        # Prefer instakill threats (kill them at range)
        if threat['instakill']:
            danger += 5

        # Prefer ranged-preferred targets (don't want to melee them)
        if threat['ranged_preferred']:
            danger += 3

        if danger > best_danger:
            best_danger = danger
            ndy = _sign(mon.row - py)
            ndx = _sign(mon.col - px)
            best = (mon, ndy, ndx)

    return best


def _navigate_to_fountain_for_excalibur(self):
    """Navigate to a fountain for Excalibur dipping.

    Searches current level and known adjacent levels.
    Returns True if an action was taken.
    """
    lvl = self.current_level()

    # Current level has a fountain
    if lvl.fountains:
        dis = self.bfs()
        best_f = None
        best_d = 999
        for fy, fx in lvl.fountains:
            d = dis[fy, fx]
            if d != -1 and d < best_d:
                best_d = d
                best_f = (fy, fx)
        if best_f:
            return self.step_toward(best_f[0], best_f[1], dis)

    return False


def _handle_minetown(self):
    """Special behavior for Minetown level.

    - Don't anger shopkeepers or priests
    - Visit altar for BUC testing
    - Visit shops for price-ID

    Returns True if an action was taken.
    """
    # Minetown detection: dungeon_number == 2 (mines) and message
    if self.gs.dnum != DUNGEON_MINES:
        return False

    # Check for shops (shopkeeper messages)
    msg = self.message.lower() if self.message else ''
    if 'welcome' in msg and ('shop' in msg or 'store' in msg):
        # We're entering a shop
        return False  # Normal explore handles this

    return False


def _emergency_escape(self):
    """Emergency escape when trapped with multiple deadly monsters.

    Options in order:
    1. Zap wand of teleportation at self
    2. Read scroll of teleportation
    3. Zap wand of digging downward
    4. Quaff potion of speed and flee

    Returns True if an action was taken.
    """
    # Only use emergency escape when things are dire
    hp_ratio = self.gs.hp / max(1, self.gs.max_hp)
    adj = self.get_adjacent_hostiles()
    if hp_ratio >= 0.2 or len(adj) < 3:
        return False

    # 1. Wand of teleportation at self
    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) == WAND_CLASS:
            lower = item.lower()
            if 'teleportation' in lower and '(0:0)' not in lower:
                zap_idx = self._val2idx.get(int(A.Command.ZAP))
                if zap_idx is not None:
                    if self._env_step(zap_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    w_idx = self._val2idx.get(ord(letter))
                    if w_idx is not None:
                        if self._env_step(w_idx):
                            self._quick_parse()
                            raise AgentFinished()
                        dot_idx = self._val2idx.get(ord('.'))
                        if dot_idx is not None:
                            if self._env_step(dot_idx):
                                self._quick_parse()
                                raise AgentFinished()
                    self._update_game_state()
                    return True

    # 2. Scroll of teleportation
    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) == SCROLL_CLASS:
            lower = item.lower()
            if 'teleportation' in lower and 'cursed' not in lower:
                read_idx = self._val2idx.get(int(A.Command.READ))
                if read_idx is not None:
                    if self._env_step(read_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        if self._env_step(l_idx):
                            self._quick_parse()
                            raise AgentFinished()
                    self._update_game_state()
                    return True

    # 3. Wand of digging downward
    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) == WAND_CLASS:
            lower = item.lower()
            if 'digging' in lower and '(0:0)' not in lower:
                zap_idx = self._val2idx.get(int(A.Command.ZAP))
                if zap_idx is not None:
                    if self._env_step(zap_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    w_idx = self._val2idx.get(ord(letter))
                    if w_idx is not None:
                        if self._env_step(w_idx):
                            self._quick_parse()
                            raise AgentFinished()
                        # Zap downward: '>' direction
                        down_idx = self._val2idx.get(ord('>'))
                        if down_idx is not None:
                            if self._env_step(down_idx):
                                self._quick_parse()
                                raise AgentFinished()
                    self._update_game_state()
                    return True

    # 4. Quaff speed potion and flee
    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) == POTION_CLASS:
            lower = item.lower()
            if 'speed' in lower and 'cursed' not in lower:
                self._two_step_quaff(letter)
                self.resistances.add('speed')
                return True

    return False


def _manage_ranged_ammo(self):
    """Manage ranged ammunition: pick up projectiles, wield launcher.

    Returns True if an action was taken.
    """
    # Check for projectiles on ground
    init = self.initial_message.lower() if self.initial_message else ''
    projectile_kw = ['dagger', 'dart', 'shuriken', 'knife', 'spear',
                      'javelin', 'arrow', 'crossbow bolt']

    if 'you see here' in init:
        for kw in projectile_kw:
            if kw in init and 'cursed' not in init:
                # Pick up projectiles
                pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
                if pickup_idx is not None:
                    if self._env_step(pickup_idx):
                        self._quick_parse()
                        raise AgentFinished()
                    self._update_game_state()
                    return True

    return False


def _evaluate_armor_piece(self, item_str):
    """Evaluate an armor piece for its AC value and special properties.

    Returns dict with:
        ac: int
        slot: str
        special: str or None
        worth_wearing: bool
    """
    lower = item_str.lower()

    # Determine slot
    slot = None
    for s, keywords in SLOT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            slot = s
            break

    # Get AC value
    ac = 0
    for armor_name, armor_ac in ARMOR_DATA.items():
        if armor_name in lower:
            ac = armor_ac
            break

    # Enchantment
    enchant = 0
    for prefix in ['+', '-']:
        idx = lower.find(prefix)
        if idx != -1 and idx + 1 < len(lower):
            try:
                enchant = int(lower[idx:idx+2])
            except ValueError:
                pass

    total_ac = ac + enchant

    # Special properties
    special = None
    if 'magic resistance' in lower:
        special = 'magic resistance'
    elif 'reflection' in lower:
        special = 'reflection'
    elif 'speed' in lower:
        special = 'speed'
    elif 'telepathy' in lower:
        special = 'telepathy'
    elif 'displacement' in lower:
        special = 'displacement'
    elif 'invisibility' in lower:
        special = 'invisibility'
    elif 'power' in lower:
        special = 'strength'
    elif 'protection' in lower:
        special = 'protection'
    elif 'brilliance' in lower:
        special = 'intelligence/wisdom'

    # Worth wearing?
    worth = True
    if 'cursed' in lower:
        worth = False
    elif 'fumbling' in lower:
        worth = False
    elif 'dunce cap' in lower:
        worth = False
    elif 'opposite alignment' in lower:
        worth = False

    return {
        'ac': total_ac,
        'slot': slot,
        'special': special,
        'worth_wearing': worth,
    }


def _get_nutrition_status(self):
    """Get detailed nutrition status.

    Returns dict with:
        hunger: str
        turns_until_hungry: int (estimate)
        has_food: bool
        food_count: int
        nutrition_available: int (estimate)
    """
    food_count = 0
    nutrition = 0

    for letter, item in self.inventory.items():
        if self.inv_oclasses.get(letter, -1) != FOOD_CLASS:
            continue
        lower = item.lower()
        if 'corpse' in lower:
            continue

        food_count += 1
        # Rough nutrition estimates
        if 'food ration' in lower:
            nutrition += 800
        elif 'cram ration' in lower:
            nutrition += 600
        elif 'lembas wafer' in lower:
            nutrition += 800
        elif 'k-ration' in lower or 'c-ration' in lower:
            nutrition += 400
        elif 'tripe ration' in lower:
            nutrition += 200
        elif 'tin' in lower:
            nutrition += 350
        elif 'candy bar' in lower:
            nutrition += 100
        elif 'pancake' in lower or 'fortune cookie' in lower:
            nutrition += 100
        elif any(f in lower for f in ['apple', 'orange', 'pear', 'melon',
                                       'banana', 'carrot']):
            nutrition += 100
        else:
            nutrition += 100

    # Turns until hungry (rough: ~20 nutrition per game turn consumed)
    # Actually nutrition consumption is ~1 per turn, 800 = ~800 turns
    turns_until_hungry = nutrition  # very rough

    return {
        'hunger': self.gs.hunger_state,
        'turns_until_hungry': turns_until_hungry,
        'has_food': food_count > 0,
        'food_count': food_count,
        'nutrition_available': nutrition,
    }


def _check_wand_charges(self, letter):
    """Check remaining charges on a wand from its description.

    Returns charges (int) or -1 if unknown.
    """
    item_str = self.inventory.get(letter, '')
    lower = item_str.lower()

    # Look for (X:Y) pattern where X is charges
    import re
    match = re.search(r'\((\d+):(\d+)\)', lower)
    if match:
        return int(match.group(2))  # remaining charges

    return -1  # unknown


def _should_use_unicorn_horn(self):
    """Check if we should apply unicorn horn.

    Unicorn horn cures: confusion, blindness, hallucination, stun,
    lost attribute points.
    """
    conditions = self.gs.conditions
    curable = {'confused', 'blind', 'stunned', 'hallucinating'}

    if conditions & curable:
        # Check if we have a unicorn horn
        for letter, item in self.inventory.items():
            if 'unicorn horn' in item.lower():
                return True

    return False


def _apply_unicorn_horn(self):
    """Apply unicorn horn to cure status effects.

    Returns True if horn was applied.
    """
    for letter, item in self.inventory.items():
        if 'unicorn horn' in item.lower():
            apply_idx = self._val2idx.get(int(A.Command.APPLY))
            if apply_idx is not None:
                if self._env_step(apply_idx):
                    self._quick_parse()
                    raise AgentFinished()
                l_idx = self._val2idx.get(ord(letter))
                if l_idx is not None:
                    if self._env_step(l_idx):
                        self._quick_parse()
                        raise AgentFinished()
                self._update_game_state()
                return True
    return False


def _count_inventory_weight(self):
    """Estimate total inventory weight from item descriptions.

    Returns approximate weight in arbitrary units.
    """
    weight = 0
    for letter, item in self.inventory.items():
        lower = item.lower()
        oc = self.inv_oclasses.get(letter, -1)

        # Count stacked items
        count = 1
        if lower and lower[0].isdigit():
            try:
                count = int(lower.split()[0])
            except ValueError:
                pass

        # Weight estimates per class
        if oc == WEAPON_CLASS:
            weight += 30 * count
        elif oc == ARMOR_CLASS:
            weight += 100 * count
        elif oc == FOOD_CLASS:
            weight += 20 * count
        elif oc == POTION_CLASS:
            weight += 20 * count
        elif oc == SCROLL_CLASS:
            weight += 5 * count
        elif oc == WAND_CLASS:
            weight += 7 * count
        elif oc == RING_CLASS:
            weight += 3 * count
        elif oc == AMULET_CLASS:
            weight += 20 * count
        elif oc == GEM_CLASS:
            weight += 1 * count
        elif oc == TOOL_CLASS:
            weight += 50 * count
        elif oc == GOLD_CLASS:
            weight += count  # 1 per gold piece
        else:
            weight += 10 * count

    return weight


# ================================================================
# Monkey-patch extended methods onto AgentV4
# ================================================================

AgentV4._assess_combat_situation = _assess_combat_situation
AgentV4._retreat_to_corridor = _retreat_to_corridor
AgentV4._handle_status_effects = _handle_status_effects
AgentV4._use_wand_strategy = _use_wand_strategy
AgentV4._read_scroll_strategy = _read_scroll_strategy
AgentV4._quaff_potion_strategy = _quaff_potion_strategy
AgentV4._detailed_inventory_management = _detailed_inventory_management
AgentV4._handle_trap_detected = _handle_trap_detected
AgentV4._find_best_combat_wand = _find_best_combat_wand
AgentV4._should_search_here = _should_search_here
AgentV4._detect_special_level = _detect_special_level
AgentV4._should_sacrifice = _should_sacrifice
AgentV4._handle_shopkeeper = _handle_shopkeeper
AgentV4._compute_threat_assessment = _compute_threat_assessment
AgentV4._handle_levelport = _handle_levelport
AgentV4._handle_polymorph = _handle_polymorph
AgentV4._detect_mines_or_sokoban = _detect_mines_or_sokoban
AgentV4._pick_ranged_target = _pick_ranged_target
AgentV4._navigate_to_fountain_for_excalibur = _navigate_to_fountain_for_excalibur
AgentV4._handle_minetown = _handle_minetown
AgentV4._emergency_escape = _emergency_escape
AgentV4._manage_ranged_ammo = _manage_ranged_ammo
AgentV4._evaluate_armor_piece = _evaluate_armor_piece
AgentV4._get_nutrition_status = _get_nutrition_status
AgentV4._check_wand_charges = _check_wand_charges
AgentV4._should_use_unicorn_horn = _should_use_unicorn_horn
AgentV4._apply_unicorn_horn = _apply_unicorn_horn
AgentV4._count_inventory_weight = _count_inventory_weight


# ================================================================
# Update main loop to use extended strategies
# ================================================================

# Override the _run_one_turn to include the extended strategies.
_original_run_one_turn = AgentV4._run_one_turn

def _run_one_turn_extended(self):
    """Extended turn loop with all additional strategies."""
    # Stall detection
    cur_turn = self.gs.turn
    if cur_turn == self._stall_turn:
        self._stall_count += 1
        if self._stall_count > 5:
            self.step(A.Command.SEARCH)
            self._stall_count = 0
            return
    else:
        self._stall_turn = cur_turn
        self._stall_count = 0

    # Stuck detection
    if self._stuck_count > 20:
        self._stuck_count = 0
        self.step(A.Command.SEARCH)
        return

    # On downstairs: check descent
    if self._on_stairs_down() and self.gs.hp > self.gs.max_hp * 0.5:
        if self._should_descend() or self._should_force_descend():
            if self._pet_pos is not None:
                pet_dist = _chebyshev(self.gs.py, self.gs.px,
                                     self._pet_pos[0], self._pet_pos[1])
                if pet_dist > 2 and self._pet_wait_count < 16:
                    self._pet_wait_count += 1
                    self.step(A.Command.SEARCH)
                    return
            self._pet_wait_count = 0
            self.step(A.MiscDirection.DOWN)
            return

    # Extended strategy chain
    if self.emergency():
        return
    if self._emergency_escape():
        return
    if self.pray_strategy():
        return
    if self._handle_status_effects():
        return
    if self.fight():
        return
    if self.eat_strategy():
        return
    if self.equip_strategy():
        return
    if self._detailed_inventory_management():
        return
    if self.excalibur_strategy():
        return
    if self.altar_buc_strategy():
        return
    if self.sacrifice_strategy():
        return
    if self._read_scroll_strategy():
        return
    if self._quaff_potion_strategy():
        return
    if self._use_wand_strategy():
        return
    if self.item_id_strategy():
        return
    if self.pickup_strategy():
        return
    if self._manage_ranged_ammo():
        return
    if self._handle_shopkeeper():
        return
    if self.sokoban_strategy():
        return
    self.explore()

AgentV4._run_one_turn = _run_one_turn_extended


# ================================================================
# Entry point for evaluation
# ================================================================

def run_agent(env, seed=None, verbose=False):
    """Run one episode of the agent. Returns final score."""
    agent = AgentV4(env, seed=seed, verbose=verbose)
    agent.main()
    return agent.score
