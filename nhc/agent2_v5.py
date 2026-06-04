"""LORE Agent v5: v2 interface + v4 strategies.

v2's proven NLE interface (SPT=1.0) with v4's expanded strategy chain.
All interface code copied verbatim from v2. Strategy code adapted from v4
to use v2's BLStats/Level/glyph data structures.

Strategy priority (highest first):
  emergency > status_effects > pray > fight > eat_corpse_after_kill >
  eat_ground > eat > pickup > equip > excalibur > altar_buc >
  scroll/potion/wand use > explore
"""
from __future__ import annotations

import re
import numpy as np
from collections import namedtuple, deque

import nle.nethack as nh
from nle.nethack import actions as A

from nhc.food import FoodManager
from nhc.equipment import EquipmentManager
from nhc.fight import assess_monster, NEVER_MELEE, INSTAKILL, PEACEFUL_NAMES, PEACEFUL_IDS
from nhc.sokoban import match_sokoban_level

# Optional subsystems from v4 (graceful fallback)
try:
    from nhc.fight import (
        FightDecision, ONLY_RANGED_SLOW, EXPLODING, WEAK, WEIRD,
        FAST_MONSTERS, should_elbereth, pick_melee_target, should_flee,
    )
except ImportError:
    FightDecision = None
    ONLY_RANGED_SLOW = {"floating eye", "blue jelly", "brown mold", "gas spore", "acid blob"}
    EXPLODING = {"yellow light", "gas spore", "flaming sphere", "freezing sphere", "shocking sphere"}
    WEAK = {"lichen", "newt", "shrieker", "grid bug"}
    WEIRD = {"leprechaun", "nymph"}
    FAST_MONSTERS = set()
    should_elbereth = None
    pick_melee_target = None
    should_flee = None

# ThreatDB for combat assessment
_THREAT_DB = None
try:
    from nhc.combat import ThreatDB
    try:
        _THREAT_DB = ThreatDB()
    except (FileNotFoundError, OSError):
        pass
except ImportError:
    pass

# PrayerState for prayer decisions
_PRAYER_STATE_CLASS = None
try:
    from nhc.prayer import PrayerState, TroubleSeverity, HungerState, Alignment
    _PRAYER_STATE_CLASS = PrayerState
except ImportError:
    pass

# AppearanceTracker for item ID
_APPEARANCE_TRACKER_CLASS = None
try:
    from nhc.item_id import AppearanceTracker
    _APPEARANCE_TRACKER_CLASS = AppearanceTracker
except (ImportError, FileNotFoundError, OSError):
    pass

# ================================================================
# v2 Constants (verbatim)
# ================================================================

BLStats = namedtuple('BLStats',
    'x y str_pct str dex con int wis cha score '
    'hp max_hp depth gold energy max_energy ac monster_level '
    'xl xp time hunger carrying_capacity dungeon_number level_number prop_mask')

GLYPH_MON_OFF = 0
GLYPH_PET_OFF = 381
GLYPH_BODY_OFF = 1144
GLYPH_OBJ_OFF = 1906
GLYPH_CMAP_OFF = 2359
NUMMONS = 381
MAP_H, MAP_W = 21, 79

_WALKABLE = frozenset({12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
_CLOSED_DOOR = frozenset({15, 16})
_WALL = frozenset(range(1, 12))
_DOOR = frozenset({12, 13, 14, 15, 16})
_STAIRS_DOWN = frozenset({24, 26})  # dnstair, dnladder
_STAIRS_UP = frozenset({22, 25})    # upstair, upladder
_FOUNTAIN = frozenset({31})
_ALTAR = frozenset({27})

# Hunger states
SATIATED = 0
NOT_HUNGRY = 1
HUNGRY = 2
WEAK_HUNGER = 3
FAINTING = 4

# Object classes
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

# Condition bitmask bits (from NLE blstats[25])
COND_STONE    = 0x00000001
COND_SLIME    = 0x00000002
COND_STRNGL   = 0x00000004
COND_FOODPOIS = 0x00000008
COND_TERMILL  = 0x00000010
COND_BLIND    = 0x00000020
COND_DEAF     = 0x00000040
COND_STUN     = 0x00000080
COND_CONF     = 0x00000100
COND_HALLU    = 0x00000200
COND_LEV      = 0x00000400
COND_FLY      = 0x00000800
COND_RIDE     = 0x00001000


class AgentFinished(Exception):
    pass


def _cmap(g):
    return g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1


DUNGEON_DOOM = 0
DUNGEON_GEHENNOM = 1
DUNGEON_MINES = 2
DUNGEON_SOKOBAN = 4

# Milestone progression (based on AutoAscend global_logic.py)
MILESTONE_FARM_DL1 = 0       # Stay on DL1 until XL >= 3
MILESTONE_DESCEND = 1        # Explore + descend main dungeon
MILESTONE_PUSH_DEEP = 2      # Push as deep as possible for score


# ================================================================
# v2 Level class (verbatim, with v4 additions for altar/corpse tracking)
# ================================================================

class Level:
    """Persistent per-level state. Survives when the agent leaves and returns."""
    def __init__(self, dungeon_number, level_number):
        self.dungeon_number = dungeon_number
        self.level_number = level_number
        self.seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self.search_count = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self.door_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self.stairs_down = set()
        self.stairs_up = set()
        self.fountains = set()
        self.altars = set()
        self.turns_spent = 0
        # Stair destinations: (y,x) -> (dungeon_number, level_number)
        self.stair_dest = {}
        self.soko_solution = None
        self.soko_offset = (0, 0)
        self.soko_step = 0
        self.soko_matched = False
        # v4 additions
        self.altar_tested_items = set()
        self.corpse_positions = {}  # (r, c) -> (name, turn_killed)

    def key(self):
        return (self.dungeon_number, self.level_number)

    @property
    def total_searches(self):
        return int(self.search_count.sum())


# ================================================================
# Monster knowledge databases (from v4)
# ================================================================

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

ITEM_STEALERS = {
    'nymph', 'water nymph', 'wood nymph', 'mountain nymph',
    'monkey', 'ape',
}

GOLD_STEALERS = {'leprechaun'}

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

LEVEL_DRAINERS = {
    'vampire', 'vampire lord', 'Vlad the Impaler',
    'wraith', 'barrow wight',
    'Nazgul',
    'succubus', 'incubus',
}

ENERGY_DRAINERS = {'mind flayer', 'master mind flayer'}

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

BREATH_WEAPONS = {
    'red dragon': 'fire', 'blue dragon': 'lightning',
    'white dragon': 'cold', 'black dragon': 'disintegration',
    'green dragon': 'poison', 'yellow dragon': 'acid',
    'orange dragon': 'sleep',
    'baby red dragon': 'fire', 'baby blue dragon': 'lightning',
    'baby white dragon': 'cold', 'baby black dragon': 'disintegration',
    'baby green dragon': 'poison',
}

MULTIPLIERS = {
    'brown mold', 'yellow mold', 'green mold', 'red mold',
    'black pudding', 'brown pudding', 'gremlin',
}

GRABBERS = {
    'owlbear', 'umber hulk', 'rope golem',
    'giant eel', 'electric eel', 'kraken', 'python',
    'giant mimic', 'large mimic', 'small mimic',
}

ARMOR_DESTROYERS = {
    'rust monster': 'rust (metal)',
    'disenchanter': 'disenchant',
    'brown pudding': 'rust/corrode',
    'black pudding': 'corrode',
}

COVETOUS = {
    'Wizard of Yendor', 'Vlad the Impaler',
    'Demogorgon', 'Orcus', 'Baalzebub',
    'Asmodeus', 'Dispater', 'Geryon',
    'Yeenoghu', 'Juiblex',
    'arch-lich', 'master lich',
}

ELBERETH_IMMUNE = {
    'minotaur', 'Death', 'Pestilence', 'Famine',
    'Wizard of Yendor', 'Archon',
}

# Wand combat knowledge
WAND_COMBAT_VALUE = {
    'death': 10, 'fire': 8, 'cold': 7, 'lightning': 8,
    'magic missile': 7, 'sleep': 6, 'slow monster': 5,
    'striking': 5, 'polymorph': 4, 'teleportation': 3,
    'cancellation': 3, 'undead turning': 3,
    'digging': 2,
}

WAND_NEVER_ZAP_AT = {
    'speed monster', 'nothing', 'light', 'probing',
    'opening', 'locking', 'make invisible', 'wishing',
    'secret door detection', 'enlightenment', 'charging',
    'create monster',
}

# Food knowledge (from v4)
try:
    from nhc.food import NEVER_ROT, MAX_CORPSE_AGE, INTRINSIC_CORPSES
except ImportError:
    NEVER_ROT = {'lizard', 'lichen'}
    MAX_CORPSE_AGE = 50
    INTRINSIC_CORPSES = {}

# Equipment knowledge
try:
    from nhc.equipment import WEAPON_DATA, ARMOR_DATA, SLOT_KEYWORDS
except ImportError:
    WEAPON_DATA = {}
    ARMOR_DATA = {}
    SLOT_KEYWORDS = {}


# ================================================================
# Helper functions
# ================================================================

def _chebyshev(r1, c1, r2, c2):
    return max(abs(r1 - r2), abs(c1 - c2))

def _sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def _in_line(r1, c1, r2, c2):
    dr = r2 - r1
    dc = c2 - c1
    if dr == 0 or dc == 0:
        return True
    return abs(dr) == abs(dc)

def _line_clear(glyphs, r1, c1, r2, c2):
    dr = _sign(r2 - r1)
    dc = _sign(c2 - c1)
    r, c = r1 + dr, c1 + dc
    while (r, c) != (r2, c2):
        if not (0 <= r < MAP_H and 0 <= c < MAP_W):
            return False
        g = int(glyphs[r, c])
        cm = _cmap(g)
        if cm in _WALL or cm in _CLOSED_DOOR or cm == 0:
            return False
        if g == GLYPH_OBJ_OFF + 447:
            return False
        r += dr
        c += dc
    return True

_DIR_DELTAS = {
    'N':  (-1,  0), 'S': ( 1,  0), 'E': ( 0,  1), 'W': ( 0, -1),
    'NE': (-1,  1), 'SE': ( 1,  1), 'SW': ( 1, -1), 'NW': (-1, -1),
}
_DELTA_TO_DIR = {v: k for k, v in _DIR_DELTAS.items()}


# ================================================================
# Agent
# ================================================================

class AgentV5:
    def __init__(self, env, seed=None, verbose=False):
        self.env = env
        self.seed = seed
        self.verbose = verbose

        # Build action lookups (v2 verbatim)
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

        # Subsystems (v2)
        self.food = FoodManager()
        self.equip = EquipmentManager()

        # v4 optional subsystems
        self.threat_db = _THREAT_DB
        self.prayer_state = _PRAYER_STATE_CLASS() if _PRAYER_STATE_CLASS else None
        self.appearance_tracker = None
        if _APPEARANCE_TRACKER_CLASS is not None:
            try:
                self.appearance_tracker = _APPEARANCE_TRACKER_CLASS()
            except (FileNotFoundError, OSError):
                pass

        # State (v2 verbatim)
        self.obs = None
        self.blstats = None
        self.glyphs = None
        self.message = ''
        self.initial_message = ''
        self.score = 0.0
        self.step_count = 0
        self._last_turn = -1

        # Multi-level state: persistent Level objects
        self.levels = {}
        self._raw_bl = None
        self._prev_level_key = None

        # Inventory
        self.inventory = {}
        self.inv_oclasses = {}

        # Character state (v2 base + v4 extensions)
        self.resistances = {"cold resistance"}
        self.has_excalibur = False
        self.has_long_sword = False
        self._last_prayer_turn = -1000
        self._last_eat_turn = -100
        self._peaceful_positions = set()
        self._peaceful_monster_ids = set()
        self._last_move_dir = (0, 0)

        # v4 additions
        self.alignment = 'neutral'
        self.role = ''
        self.race = ''
        self._elbereth_cooldown = 0
        self._on_elbereth = False
        self._kill_count = 0
        self._last_kill_name = None
        self._last_kill_dir = (0, 0)
        self._wands_tested = set()
        self._last_pickup_turn = -100
        self._stuck_count = 0
        self._last_pos = (-1, -1)

        # Debug
        self._ugs_count = 0
        self._prompt_steps = 0
        self._yn_count = 0
        self._xwait_count = 0
        self._getlin_count = 0
        self._more_count = 0

        # Milestone system (v2 verbatim)
        self.milestone = MILESTONE_FARM_DL1

    def current_level(self):
        """Get or create the Level object for the current dungeon position."""
        if self.blstats is None:
            key = (0, 1)
        else:
            key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        return self.levels[key]

    # Convenience accessors for current level maps (v2 verbatim)
    @property
    def seen(self):
        return self.current_level().seen
    @property
    def walkable(self):
        return self.current_level().walkable
    @property
    def objects(self):
        return self.current_level().objects
    @property
    def search_count(self):
        return self.current_level().search_count
    @property
    def door_attempts(self):
        return self.current_level().door_attempts
    @property
    def _stairs_down(self):
        return self.current_level().stairs_down
    @property
    def _stairs_up(self):
        return self.current_level().stairs_up
    @property
    def _fountains(self):
        return self.current_level().fountains
    @property
    def _altars(self):
        return self.current_level().altars
    @property
    def _level_turns(self):
        return self.current_level().turns_spent

    # ================================================================
    # Condition helpers (v4 style, using v2 data)
    # ================================================================

    def _get_conditions(self):
        """Get condition bitmask from blstats prop_mask."""
        if self._raw_bl is not None and len(self._raw_bl) > 25:
            return int(self._raw_bl[25])
        return 0

    def _get_condition_set(self):
        """Return set of active condition strings."""
        cond = self._get_conditions()
        result = set()
        if cond & COND_STONE:    result.add('stoned')
        if cond & COND_SLIME:    result.add('slimed')
        if cond & COND_STRNGL:   result.add('strangled')
        if cond & COND_FOODPOIS: result.add('foodpois')
        if cond & COND_TERMILL:  result.add('termill')
        if cond & COND_BLIND:    result.add('blind')
        if cond & COND_DEAF:     result.add('deaf')
        if cond & COND_STUN:     result.add('stunned')
        if cond & COND_CONF:     result.add('confused')
        if cond & COND_HALLU:    result.add('hallucinating')
        return result

    def _get_alignment(self):
        """Get alignment from raw blstats index 26."""
        if self._raw_bl is not None and len(self._raw_bl) > 26:
            al = int(self._raw_bl[26])
            if al == 1: return 'lawful'
            if al == 0: return 'neutral'
            return 'chaotic'
        return self.alignment

    def _get_encumbrance(self):
        """Get encumbrance level from blstats carrying_capacity."""
        if self.blstats is not None:
            return self.blstats.carrying_capacity
        return 0

    def _get_hunger_state_str(self):
        """Get hunger state as string."""
        if self.blstats is None:
            return 'not_hungry'
        h = self.blstats.hunger
        if h == 0: return 'satiated'
        if h == 1: return 'not_hungry'
        if h == 2: return 'hungry'
        if h == 3: return 'weak'
        if h == 4: return 'fainting'
        if h >= 5: return 'fainted'
        return 'not_hungry'

    # ================================================================
    # Core: step / update (v2 VERBATIM)
    # ================================================================

    def _env_step(self, idx):
        """Raw env.step with observation copy."""
        obs, reward, done, truncated, info = self.env.step(idx)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        bl = self.obs.get('blstats')
        if bl is not None and int(bl[20]) > 0:
            self._raw_bl = np.array(bl, dtype=np.int64)
        self.score += reward
        self.step_count += 1
        return done or truncated

    def step(self, action, gen=None):
        """Send action to env, handle prompts iteratively.

        action: NLE action object, string char, or int action value
        gen: optional generator yielding responses for multi-step actions
        """
        if isinstance(action, str):
            assert len(action) == 1
            idx = self._val2idx.get(ord(action))
        elif type(action) is int and action < len(self.actions):
            # Plain int: treat as action index
            idx = action
        else:
            # NLE action enum or other: look up by value
            idx = self._val2idx.get(int(action))

        if idx is None:
            return

        if self._env_step(idx):
            self._parse_blstats()
            raise AgentFinished()

        # Save initial message before prompt handling
        raw_msg = self.obs.get('message', b'')
        self.initial_message = bytes(raw_msg).decode('latin-1', errors='replace').replace('\x00', '').strip()

        # Handle prompts iteratively
        prompt_count = 0
        for _ in range(200):
            msg_raw = self.obs.get('message', b'')
            self.message = bytes(msg_raw).decode('latin-1', errors='replace').replace('\x00', '').strip()
            misc = self.obs.get('misc', [0, 0, 0])

            # Generator for multi-step actions (eat letter, engrave text, etc.)
            if gen is not None:
                try:
                    next_action = next(gen)
                    if isinstance(next_action, str):
                        next_idx = self._val2idx.get(ord(next_action))
                    elif isinstance(next_action, int) and next_action < len(self.actions):
                        next_idx = next_action
                    else:
                        next_idx = self._val2idx.get(int(next_action))
                    if next_idx is not None:
                        if self._env_step(next_idx):
                            self._parse_blstats()
                            raise AgentFinished()
                        continue
                except StopIteration:
                    gen = None

            # yn prompt (misc[0] = in_yn_function)
            if misc[0]:
                self._yn_count += 1
                resp = self._handle_yn_prompt()
                if resp is not None:
                    if self._env_step(resp):
                        self._parse_blstats()
                        raise AgentFinished()
                continue

            # Text entry (misc[1] = in_getlin)
            if misc[1]:
                self._getlin_count += 1
                # Handle eat menu: type food letter
                if 'What do you want to eat' in self.message:
                    food_letter = self._find_food_to_eat()
                    if food_letter:
                        resp_idx = self._val2idx.get(ord(food_letter))
                        if resp_idx is not None:
                            if self._env_step(resp_idx):
                                self._parse_blstats()
                                raise AgentFinished()
                            continue
                # Handle wield menu
                if 'What do you want to wield' in self.message:
                    weapon_letter = self.equip.find_best_weapon(self.inventory)
                    if weapon_letter:
                        resp_idx = self._val2idx.get(ord(weapon_letter))
                        if resp_idx is not None:
                            if self._env_step(resp_idx):
                                self._parse_blstats()
                                raise AgentFinished()
                            continue
                # Handle wear menu
                if 'What do you want to wear' in self.message:
                    armor_letter = self.equip.find_best_armor(self.inventory)
                    if armor_letter:
                        resp_idx = self._val2idx.get(ord(armor_letter))
                        if resp_idx is not None:
                            if self._env_step(resp_idx):
                                self._parse_blstats()
                                raise AgentFinished()
                            continue
                # Handle quaff menu
                if 'What do you want to drink' in self.message or 'What do you want to quaff' in self.message:
                    potion_letter = self._find_healing_potion()
                    if potion_letter:
                        resp_idx = self._val2idx.get(ord(potion_letter))
                        if resp_idx is not None:
                            if self._env_step(resp_idx):
                                self._parse_blstats()
                                raise AgentFinished()
                            continue
                # Default: ESC out of text entry
                if self._env_step(self._val2idx.get(27, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # xwait (misc[2])
            if misc[2]:
                self._xwait_count += 1
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # --More-- in message text (backup check)
            if '--More--' in self.message:
                self._more_count += 1
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            break

        if prompt_count > 3 and self.verbose:
            print(f"  PROMPT_LOOP: {prompt_count} iterations, msg={self.message[:60]}")
        self._prompt_steps += prompt_count
        self._update_game_state()

    def _handle_yn_prompt(self):
        """Decide response to yn prompt. Returns action index."""
        msg = self.message
        # Don't attack peacefuls - record their position AND monster ID
        if 'Really attack' in msg:
            if self.blstats and self._last_move_dir != (0, 0):
                dy, dx = self._last_move_dir
                py, px = self.blstats.y, self.blstats.x
                ty, tx = py + dy, px + dx
                if 0 <= ty < MAP_H and 0 <= tx < MAP_W:
                    self._peaceful_positions.add((ty, tx))
                    g = int(self.glyphs[ty, tx]) if self.glyphs is not None else 0
                    if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                        self._peaceful_monster_ids.add(g - GLYPH_MON_OFF)
            return self._val2idx.get(ord('n'))
        # Don't force locks
        if 'force the lock' in msg:
            return self._val2idx.get(ord('n'))
        # Eat it? check corpse safety (v4 improvement)
        if 'eat it?' in msg.lower() or 'eat this?' in msg.lower():
            if self._is_current_corpse_safe():
                return self._val2idx.get(ord('y'))
            return self._val2idx.get(ord('n'))
        # Menus that need ESC
        if any(s in msg for s in ['What do you want to eat',
                                   'What do you want to dip',
                                   'What do you want to drop',
                                   'What do you want to throw',
                                   'What do you want to wield',
                                   'What do you want to wear',
                                   'What do you want to take off',
                                   'What do you want to put on',
                                   'What do you want to remove',
                                   'What do you want to call',
                                   'What do you want to use']):
            return self._val2idx.get(27)  # ESC
        # Don't pray when unsafe
        if 'Are you sure' in msg and 'pray' in msg:
            return self._val2idx.get(ord('y'))
        # Sacrifice confirmation
        if 'sacrifice' in msg.lower():
            return self._val2idx.get(ord('y'))
        # Don't pick up if carrying too much
        if 'carrying too much' in msg.lower():
            return self._val2idx.get(ord('n'))
        # Stop eating - no, keep eating
        if 'Stop eating' in msg:
            return self._val2idx.get(ord('n'))
        # Still climb stairs
        if 'Still climb' in msg:
            return self._val2idx.get(ord('y'))
        # Don't loot containers (wastes turns)
        if 'loot' in msg.lower():
            return self._val2idx.get(ord('n'))
        # Don't drink from fountain
        if 'Drink from' in msg and 'fountain' in msg:
            return self._val2idx.get(ord('n'))
        # Dip in fountain - yes
        if 'Dip' in msg and 'fountain' in msg:
            return self._val2idx.get(ord('y'))
        # Add to engraving - no
        if 'add to' in msg.lower() and 'engraving' in msg.lower():
            return self._val2idx.get(ord('n'))
        # Shall I pick up - no (we handle pickup explicitly)
        if 'Shall I pick' in msg:
            return self._val2idx.get(ord('n'))
        # Default: yes
        return self._val2idx.get(ord('y'))

    def _is_current_corpse_safe(self):
        """Check if the corpse mentioned in current message is safe to eat."""
        msg = self.message.lower()
        name = None
        for pat in ['eat it?', 'eat this?']:
            if pat in msg:
                idx = msg.find('corpse')
                if idx > 0:
                    prefix = msg[:idx].strip()
                    for article in ['a ', 'an ', 'the ']:
                        aidx = prefix.rfind(article)
                        if aidx >= 0:
                            name = prefix[aidx + len(article):].strip()
                            break
                    if name is None:
                        name = prefix.split()[-1] if prefix.split() else None
                break
        if name is None:
            return self.blstats is not None and self.blstats.hunger >= HUNGRY
        # ThreatDB assessment
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            if report.safe_to_eat:
                if name == 'wraith':
                    return True
                if report.beneficial_intrinsic and report.beneficial_intrinsic not in self.resistances:
                    return True
                if self.blstats is not None and self.blstats.hunger >= HUNGRY:
                    return True
                if name in ('lizard', 'lichen'):
                    return True
                return False
            return False
        return self.food.is_corpse_safe(name, self.resistances)

    # ================================================================
    # State update (v2 VERBATIM core + v4 extensions)
    # ================================================================

    def _update_game_state(self):
        """Parse observation into full game state."""
        self._ugs_count += 1
        self._parse_blstats()
        if self.blstats is None:
            return

        # Level change detection
        cur_key = (self.blstats.dungeon_number, self.blstats.level_number)
        if self._prev_level_key is None:
            self._prev_level_key = cur_key
        elif cur_key != self._prev_level_key:
            old_key = self._prev_level_key
            self._prev_level_key = cur_key
            self._peaceful_positions = set()
            self.food.on_level_change()
            self._on_elbereth = False
            self._elbereth_cooldown = 0
            # Record stair connections between levels
            lvl = self.current_level()
            went_down = (cur_key[1] > old_key[1] if cur_key[0] == old_key[0]
                         else cur_key[0] != old_key[0])
            if went_down:
                lvl.stairs_up.add((self.blstats.y, self.blstats.x))
            else:
                lvl.stairs_down.add((self.blstats.y, self.blstats.x))

        # Track turns on this level
        if self.blstats.time != self._last_turn:
            self.current_level().turns_spent += 1
            self._last_turn = self.blstats.time

        self.glyphs = self.obs['glyphs']
        self._update_maps()
        self._parse_inventory()
        self._parse_messages()

        # Update alignment from blstats
        self.alignment = self._get_alignment()

        # Update prayer state
        if self.prayer_state is not None:
            self._update_prayer_state()

        # Update Elbereth state
        if self._elbereth_cooldown > 0:
            self._elbereth_cooldown -= 1
        init = self.initial_message.lower() if self.initial_message else ''
        self._on_elbereth = 'elbereth' in init

        # Stuck detection
        cur_pos = (self.blstats.y, self.blstats.x)
        if cur_pos == self._last_pos:
            self._stuck_count += 1
        else:
            self._stuck_count = 0
            self._last_pos = cur_pos

    def _parse_blstats(self):
        bl = self._raw_bl
        if bl is not None and len(bl) >= 26:
            self.blstats = BLStats(*[int(v) for v in bl[:26]])

    def _update_maps(self):
        g = self.glyphs
        py, px = self.blstats.y, self.blstats.x

        for r in range(MAP_H):
            for c in range(MAP_W):
                v = int(g[r, c])
                cm = _cmap(v)
                if cm in _WALKABLE:
                    self.seen[r, c] = True
                    self.walkable[r, c] = True
                    # Only store terrain glyphs in objects (not monster/player glyphs)
                    self.objects[r, c] = v
                    # Accumulate stairs and fountains (persist until level change)
                    if cm in _STAIRS_DOWN:
                        self._stairs_down.add((r, c))
                    if cm in _STAIRS_UP:
                        self._stairs_up.add((r, c))
                    if cm in _FOUNTAIN:
                        self._fountains.add((r, c))
                    if cm in _ALTAR:
                        self._altars.add((r, c))
                elif cm in _WALL:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                    self.objects[r, c] = v
                elif cm in _CLOSED_DOOR:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                    self.objects[r, c] = v
                elif v == GLYPH_OBJ_OFF + 447:  # boulder
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif GLYPH_MON_OFF <= v < GLYPH_CMAP_OFF:
                    # Monster on tile: mark seen/walkable but DON'T overwrite objects
                    self.seen[r, c] = True
                    if self.objects[r, c] == -1:
                        self.walkable[r, c] = True
                elif cm == 0:
                    if abs(r - py) <= 1 and abs(c - px) <= 1:
                        self.seen[r, c] = True
                        self.walkable[r, c] = False
        self.walkable[py, px] = True
        self.seen[py, px] = True

        # Also detect stairs from chars observation (more reliable than glyph CMAP)
        chars = self.obs.get('chars')
        if chars is not None:
            ys, xs = (chars == ord('>')).nonzero()
            for y, x in zip(ys, xs):
                self._stairs_down.add((int(y), int(x)))
            ys, xs = (chars == ord('<')).nonzero()
            for y, x in zip(ys, xs):
                self._stairs_up.add((int(y), int(x)))

    def _parse_inventory(self):
        inv_strs = self.obs.get('inv_strs')
        inv_letters = self.obs.get('inv_letters')
        inv_oclasses = self.obs.get('inv_oclasses')
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
        if any('Excalibur' in s for s in self.inventory.values()):
            self.has_excalibur = True
        self.has_long_sword = any('long sword' in s.lower() for s in self.inventory.values())

    def _parse_messages(self):
        msg = self.message.lower()
        init_msg = self.initial_message.lower() if self.initial_message else ''

        # Resistance detection (v2 + v4 expanded)
        resists = {
            "you feel especially healthy": "poison resistance",
            "you feel a momentary chill": "cold resistance",
            "you feel warm": "fire resistance",
            "you feel full of energy": "shock resistance",
            "you feel wide awake": "sleep resistance",
            "you feel very firm": "disintegration resistance",
        }
        for frag, r in resists.items():
            if frag in msg or frag in init_msg:
                self.resistances.add(r)

        # Telepathy, teleport control, speed, see invisible (v4)
        if "you feel a strange mental acuity" in msg:
            self.resistances.add("telepathy")
        if "you feel in control of yourself" in msg:
            self.resistances.add("teleport control")
        if "you feel very fast" in msg or "you speed up" in msg:
            self.resistances.add("speed")
        if "you feel transparent" in msg:
            self.resistances.add("see invisible")

        if "your sword has a bright" in msg or "excalibur" in msg:
            self.has_excalibur = True

        # Role/race detection (v4)
        for m in (msg, init_msg):
            self._detect_role_race(m)

        # Prayer results (v4)
        if 'is angry' in msg and self.prayer_state is not None:
            self.prayer_state.god_anger += 1
        if 'you feel lucky' in msg and self.prayer_state is not None:
            self.prayer_state.luck = max(self.prayer_state.luck, 1)

        # Fountain drying (v4)
        if 'the fountain dries up' in msg or 'the fountain disappears' in msg:
            py, px = self.blstats.y, self.blstats.x
            self._fountains.discard((py, px))

        # Kill tracking with corpse position (v2 verbatim + v4 corpse tracking)
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
                name = name.rstrip('.')
                for article in ['a ', 'an ', 'the ']:
                    if name.startswith(article):
                        name = name[len(article):]
                py, px = self.blstats.y, self.blstats.x
                self.food.on_kill(name, py, px, self.blstats.time, self.resistances)
                self._last_kill_name = name
                self._last_kill_dir = self._last_move_dir
                self._kill_count += 1
                # Record in level corpse tracker
                dy, dx = self._last_move_dir
                kill_r, kill_c = py + dy, px + dx
                lvl = self.current_level()
                lvl.corpse_positions[(kill_r, kill_c)] = (name, self.blstats.time)
                break

    def _detect_role_race(self, msg):
        """Detect role and race from message text."""
        if not self.role:
            roles = {
                'valkyrie': ('Valkyrie', {'cold resistance'}),
                'tourist': ('Tourist', set()),
                'wizard': ('Wizard', set()),
                'samurai': ('Samurai', set()),
                'barbarian': ('Barbarian', {'poison resistance'}),
                'priestess': ('Priest', set()),
                'priest': ('Priest', set()),
                'rogue': ('Rogue', set()),
                'ranger': ('Ranger', set()),
                'knight': ('Knight', set()),
                'cave': ('Caveman', set()),
                'healer': ('Healer', {'poison resistance'}),
                'monk': ('Monk', {'poison resistance', 'sleep resistance'}),
                'archeologist': ('Archeologist', set()),
            }
            for kw, (rname, rresists) in roles.items():
                if kw in msg:
                    self.role = rname
                    self.resistances |= rresists
                    break
        if not self.race:
            races = {
                'elf': ('Elf', {'sleep resistance'}),
                'orc': ('Orc', {'poison resistance'}),
                'dwarf': ('Dwarf', set()),
                'gnome': ('Gnome', set()),
                'human': ('Human', set()),
            }
            for kw, (rname, rresists) in races.items():
                if kw in msg:
                    self.race = rname
                    self.resistances |= rresists
                    break

    def _update_prayer_state(self):
        """Update PrayerState from blstats."""
        if self.prayer_state is None:
            return
        if self._raw_bl is not None and len(self._raw_bl) > 26:
            al = int(self._raw_bl[26])
            if al == 1:
                self.prayer_state.player_alignment = Alignment.LAWFUL
            elif al == 0:
                self.prayer_state.player_alignment = Alignment.NEUTRAL
            else:
                self.prayer_state.player_alignment = Alignment.CHAOTIC
        # Gehennom flag
        if self.blstats.dungeon_number == DUNGEON_GEHENNOM or self.blstats.depth >= 25:
            self.prayer_state.in_gehennom = True
        else:
            self.prayer_state.in_gehennom = False

    # ================================================================
    # Navigation (v2 VERBATIM)
    # ================================================================

    def _is_peaceful_glyph(self, g):
        """Check if a monster glyph represents a peaceful monster."""
        if not (GLYPH_MON_OFF <= g < GLYPH_PET_OFF):
            return False
        mid = g - GLYPH_MON_OFF
        if mid in PEACEFUL_IDS:
            return True
        if mid < NUMMONS:
            name = nh.permonst(mid).mname
            if name in PEACEFUL_NAMES:
                return True
        return False

    def bfs(self):
        py, px = self.blstats.y, self.blstats.x
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
                    if not (0 <= ny < MAP_H and 0 <= nx < MAP_W) or dis[ny, nx] != -1:
                        continue
                    g = int(self.glyphs[ny, nx])
                    cm = _cmap(g)
                    # Closed doors: only include if not exhausted (locked/stuck)
                    is_closed_door = cm in _CLOSED_DOOR
                    if is_closed_door and self.door_attempts[ny, nx] >= 5:
                        continue
                    # Pets: can swap places, include them. All other monsters: exclude.
                    is_pet = GLYPH_PET_OFF <= g < GLYPH_BODY_OFF
                    is_wild_monster = GLYPH_MON_OFF <= g < GLYPH_PET_OFF
                    if is_wild_monster:
                        continue  # Don't path through ANY wild monster
                    ok = self.walkable[ny, nx] or is_closed_door or is_pet
                    if not ok:
                        continue
                    # No diagonal through doors
                    if abs(dy) + abs(dx) > 1:
                        if _cmap(int(self.glyphs[y, x])) in _DOOR or cm in _DOOR:
                            continue
                    dis[ny, nx] = d + 1
                    q.append((ny, nx))
        return dis

    def step_toward(self, ty, tx, dis):
        """Take one BFS-optimal step toward (ty, tx). Returns True if stepped."""
        if dis[ty, tx] == -1:
            return False
        py, px = self.blstats.y, self.blstats.x
        # Trace back from target to player
        path = []
        cy, cx = ty, tx
        for _ in range(300):
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
            if g == GLYPH_OBJ_OFF + 447:  # boulder
                return False
            cm = _cmap(g)
            if cm in _WALL or cm == 0:
                return False

        # Diagonal through door: use cardinal instead
        if abs(dy) + abs(dx) > 1:
            src_cm = _cmap(int(self.glyphs[py, px]))
            dst_cm = _cmap(int(self.glyphs[ny, nx]))
            src_obj = _cmap(int(self.objects[py, px])) if self.objects[py, px] != -1 else -1
            dst_obj = _cmap(int(self.objects[ny, nx])) if self.objects[ny, nx] != -1 else -1
            if src_cm in _DOOR or dst_cm in _DOOR or src_obj in _DOOR or dst_obj in _DOOR:
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W and (self.walkable[cr, cc] or _cmap(int(self.glyphs[cr, cc])) in _CLOSED_DOOR):
                        dy, dx = cdy, cdx
                        ny, nx = cr, cc
                        break

        # Final validation
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            cm = _cmap(g)
            if cm in _WALL or cm == 0 or g == GLYPH_OBJ_OFF + 447:
                return False

        old_pos = (self.blstats.y, self.blstats.x)
        self._move_dir(dy, dx)
        # Retry if diagonal failed
        if 'diagonally' in self.message.lower() and abs(dy) + abs(dx) > 1:
            for cdy, cdx in [(dy, 0), (0, dx)]:
                if cdy == 0 and cdx == 0:
                    continue
                self._move_dir(cdy, cdx)
                break
        # Check if we actually moved (or at least attempted valid action)
        new_pos = (self.blstats.y, self.blstats.x) if self.blstats else old_pos
        return new_pos != old_pos

    def _move_dir(self, dy, dx):
        """Send a compass direction."""
        self._last_move_dir = (dy, dx)
        dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W',
                (-1,1):'NE',(1,1):'SE',(1,-1):'SW',(-1,-1):'NW'}
        name = dmap.get((dy, dx))
        if name and name in self._name2idx:
            self.step(self._name2idx[name])

    def _kick_dir(self, dy, dx):
        """Kick in a direction (for locked doors)."""
        dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W'}
        name = dmap.get((dy, dx))
        if name and name in self._name2idx:
            self.step(A.Command.KICK)
            self.step(self._name2idx[name])

    def get_monsters(self):
        """Visible non-pet monsters with distance."""
        py, px = self.blstats.y, self.blstats.x
        mons = []
        for r in range(MAP_H):
            for c in range(MAP_W):
                if r == py and c == px:
                    continue
                g = int(self.glyphs[r, c])
                if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                    mid = g - GLYPH_MON_OFF
                    name = nh.permonst(mid).mname if mid < NUMMONS else f"mon{mid}"
                    d = max(abs(r - py), abs(c - px))
                    mons.append((d, r, c, name, mid))
        return sorted(mons)

    def _bfs_allow_hostiles(self):
        """BFS that treats hostile monsters as walkable (can fight through them)."""
        py, px = self.blstats.y, self.blstats.x
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
                    if not (0 <= ny < MAP_H and 0 <= nx < MAP_W) or dis[ny, nx] != -1:
                        continue
                    g = int(self.glyphs[ny, nx])
                    cm = _cmap(g)
                    is_closed_door = cm in _CLOSED_DOOR
                    if is_closed_door and self.door_attempts[ny, nx] >= 10:
                        continue
                    # Include hostile monsters and pets, exclude peacefuls
                    is_monster = GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF
                    if is_monster and self._is_peaceful_glyph(g):
                        continue
                    if is_monster and (ny, nx) in self._peaceful_positions:
                        continue
                    ok = self.walkable[ny, nx] or is_closed_door or is_monster
                    if not ok:
                        continue
                    if abs(dy) + abs(dx) > 1:
                        if _cmap(int(self.glyphs[y, x])) in _DOOR or cm in _DOOR:
                            continue
                    dis[ny, nx] = d + 1
                    q.append((ny, nx))
        return dis

    def _greedy_move_toward(self, ty, tx):
        """Try to move closer to (ty, tx) using cardinal directions."""
        py, px = self.blstats.y, self.blstats.x
        best_dir = None
        best_dist = abs(ty - py) + abs(tx - px)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            if not self.walkable[ny, nx]:
                g = int(self.glyphs[ny, nx])
                if not (GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF):
                    continue
            # Skip diagonal through doors
            if abs(dy) + abs(dx) > 1:
                src_cm = _cmap(int(self.glyphs[py, px]))
                dst_cm = _cmap(int(self.glyphs[ny, nx]))
                if src_cm in _DOOR or dst_cm in _DOOR:
                    continue
            d = abs(ty - ny) + abs(tx - nx)
            if d < best_dist:
                best_dist = d
                best_dir = (dy, dx)
        if best_dir:
            self._move_dir(best_dir[0], best_dir[1])
        else:
            self.step(A.Command.SEARCH)

    def _engrave_elbereth(self):
        """Engrave 'Elbereth' in the dust using fingers. (v2 verbatim)"""
        engrave_idx = self._val2idx.get(int(A.Command.ENGRAVE))
        if engrave_idx is None:
            return
        if self._env_step(engrave_idx):
            self._parse_blstats()
            raise AgentFinished()

        # Check for "add to current engraving?" yn prompt
        misc = self.obs.get('misc', [0, 0, 0])
        if misc[0] and 'add to' in bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').lower():
            n_idx = self._val2idx.get(ord('n'))
            if n_idx and self._env_step(n_idx):
                self._parse_blstats()
                raise AgentFinished()

        # '-' (fingers)
        dash_idx = self._val2idx.get(ord('-'))
        if dash_idx is not None:
            if self._env_step(dash_idx):
                self._parse_blstats()
                raise AgentFinished()

        # Type E,l,b,e,r,e,t,h
        for ch in 'Elbereth':
            ch_idx = self._val2idx.get(ord(ch))
            if ch_idx is not None:
                if self._env_step(ch_idx):
                    self._parse_blstats()
                    raise AgentFinished()

        # Enter to finish
        cr_idx = self._val2idx.get(13)
        if cr_idx is not None:
            if self._env_step(cr_idx):
                self._parse_blstats()
                raise AgentFinished()

        self._elbereth_cooldown = 5
        self._on_elbereth = True
        self._update_game_state()

    def _solve_sokoban(self):
        """Execute one step of the Sokoban puzzle solution. (v2 verbatim)"""
        lvl = self.current_level()
        if not lvl.soko_matched:
            lvl.soko_matched = True
            wall_mask = np.zeros((MAP_H, MAP_W), dtype=bool)
            for r in range(MAP_H):
                for c in range(MAP_W):
                    if _cmap(int(self.glyphs[r, c])) in _WALL:
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
        py, px = self.blstats.y, self.blstats.x

        # Check boulder still there
        if 0 <= game_by < MAP_H and 0 <= game_bx < MAP_W:
            if int(self.glyphs[game_by, game_bx]) != GLYPH_OBJ_OFF + 447:
                lvl.soko_step += 1
                return True

        if (py, px) != (push_y, push_x):
            dis = self.bfs()
            if dis[push_y, push_x] == -1:
                lvl.soko_step += 1
                return True
            self.step_toward(push_y, push_x, dis)
            return True

        self._move_dir(dy, dx)
        lvl.soko_step += 1
        return True

    def _on_stairs_down(self):
        """Check if player is standing on downstairs. (v2 verbatim)"""
        py, px = self.blstats.y, self.blstats.x
        if (py, px) in self._stairs_down:
            return True
        g_here = int(self.glyphs[py, px]) if self.glyphs is not None else 0
        cm_here = _cmap(g_here)
        if cm_here in _STAIRS_DOWN:
            return True
        obj_here = int(self.objects[py, px]) if self.objects[py, px] != -1 else -1
        cm_obj = _cmap(obj_here) if obj_here != -1 else -1
        return cm_obj in _STAIRS_DOWN

    def _has_frontier(self):
        """Check if there are REACHABLE unexplored tiles. (v2 verbatim)"""
        dis = self.bfs()
        for r in range(MAP_H):
            for c in range(MAP_W):
                if not self.walkable[r, c] or dis[r, c] == -1:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not self.seen[nr, nc]:
                            return True
        return False

    # ================================================================
    # Two-step and three-step commands (from v4, adapted to v2 interface)
    # ================================================================

    def _two_step_eat(self, letter):
        """Two-step eat: EAT command then item letter."""
        eat_idx = self._val2idx.get(int(A.Command.EAT))
        if eat_idx is None:
            return
        if self._env_step(eat_idx):
            self._parse_blstats()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._parse_blstats()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_quaff(self, letter):
        """Two-step quaff: QUAFF command then potion letter."""
        quaff_idx = self._val2idx.get(int(A.Command.QUAFF))
        if quaff_idx is None:
            return
        if self._env_step(quaff_idx):
            self._parse_blstats()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._parse_blstats()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_wield(self, letter):
        """Two-step wield: WIELD then weapon letter."""
        wield_idx = self._val2idx.get(int(A.Command.WIELD))
        if wield_idx is None:
            return
        if self._env_step(wield_idx):
            self._parse_blstats()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._parse_blstats()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_wear(self, letter):
        """Two-step wear: WEAR then armor letter."""
        wear_idx = self._val2idx.get(int(A.Command.WEAR))
        if wear_idx is None:
            return
        if self._env_step(wear_idx):
            self._parse_blstats()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._parse_blstats()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_drop(self, letter):
        """Two-step drop: DROP then item letter."""
        drop_idx = self._val2idx.get(int(A.Command.DROP))
        if drop_idx is None:
            return
        if self._env_step(drop_idx):
            self._parse_blstats()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._parse_blstats()
                raise AgentFinished()
        self._update_game_state()

    def _two_step_dip(self, item_letter):
        """Two-step dip: DIP then item letter. Handles fountain prompt."""
        dip_idx = self._val2idx.get(int(A.Command.DIP))
        if dip_idx is None:
            return
        if self._env_step(dip_idx):
            self._parse_blstats()
            raise AgentFinished()
        letter_idx = self._val2idx.get(ord(item_letter))
        if letter_idx is not None:
            if self._env_step(letter_idx):
                self._parse_blstats()
                raise AgentFinished()
        # Handle follow-up prompts
        for _ in range(10):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').replace('\x00', '').strip()
            if misc[0]:
                y_idx = self._val2idx.get(ord('y'))
                if y_idx is not None:
                    if self._env_step(y_idx):
                        self._parse_blstats()
                        raise AgentFinished()
                continue
            elif misc[2] or '--More--' in msg:
                sp_idx = self._val2idx.get(32, 0)
                if self._env_step(sp_idx):
                    self._parse_blstats()
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
            self._parse_blstats(); raise AgentFinished()
        w_idx = self._val2idx.get(ord(wand_letter))
        if w_idx is None:
            self._update_game_state(); return
        if self._env_step(w_idx):
            self._parse_blstats(); raise AgentFinished()
        dir_name = _DELTA_TO_DIR.get((dy, dx))
        if dir_name and dir_name in self._name2idx:
            if self._env_step(self._name2idx[dir_name]):
                self._parse_blstats(); raise AgentFinished()
        self._update_game_state()

    def _three_step_throw(self, item_letter, dy, dx):
        """Three-step throw: THROW -> item letter -> direction."""
        throw_idx = self._val2idx.get(int(A.Command.THROW))
        if throw_idx is None:
            return
        if self._env_step(throw_idx):
            self._parse_blstats(); raise AgentFinished()
        i_idx = self._val2idx.get(ord(item_letter))
        if i_idx is None:
            self._update_game_state(); return
        if self._env_step(i_idx):
            self._parse_blstats(); raise AgentFinished()
        dir_name = _DELTA_TO_DIR.get((dy, dx))
        if dir_name and dir_name in self._name2idx:
            if self._env_step(self._name2idx[dir_name]):
                self._parse_blstats(); raise AgentFinished()
        self._update_game_state()

    def _apply_item(self, letter):
        """Two-step apply: APPLY then item letter."""
        apply_idx = self._val2idx.get(int(A.Command.APPLY))
        if apply_idx is None:
            return
        if self._env_step(apply_idx):
            self._parse_blstats()
            raise AgentFinished()
        l_idx = self._val2idx.get(ord(letter))
        if l_idx is not None:
            if self._env_step(l_idx):
                self._parse_blstats()
                raise AgentFinished()
        self._update_game_state()

    def _read_scroll(self, letter):
        """Two-step read: READ then scroll letter."""
        read_idx = self._val2idx.get(int(A.Command.READ))
        if read_idx is None:
            return
        if self._env_step(read_idx):
            self._parse_blstats()
            raise AgentFinished()
        l_idx = self._val2idx.get(ord(letter))
        if l_idx is not None:
            if self._env_step(l_idx):
                self._parse_blstats()
                raise AgentFinished()
        # Handle follow-up menus (identify, etc.)
        for _ in range(5):
            misc = self.obs.get('misc', [0, 0, 0])
            if misc[1]:
                # Pick first item from inventory for identify
                for inv_letter in self.inventory:
                    il_idx = self._val2idx.get(ord(inv_letter))
                    if il_idx is not None:
                        if self._env_step(il_idx):
                            self._parse_blstats()
                            raise AgentFinished()
                        break
                continue
            break
        self._update_game_state()

    # ================================================================
    # Inventory helpers (from v4)
    # ================================================================

    def _has_food_in_inventory(self):
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc == FOOD_CLASS:
                lower = item.lower()
                if 'corpse' not in lower and 'cursed' not in lower:
                    return True
        return False

    def _find_food_to_eat(self):
        """Find best food item letter from inventory."""
        best_letter = None
        best_priority = -1
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc != FOOD_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower or 'corpse' in lower:
                continue
            pri = 1
            if 'food ration' in lower: pri = 10
            elif 'cram ration' in lower: pri = 9
            elif 'lembas wafer' in lower: pri = 9
            elif 'k-ration' in lower or 'c-ration' in lower: pri = 8
            elif 'candy bar' in lower: pri = 7
            elif 'pancake' in lower or 'fortune cookie' in lower: pri = 7
            elif any(f in lower for f in ['apple', 'orange', 'pear', 'melon', 'banana', 'carrot']): pri = 6
            elif 'tin' in lower: pri = 5
            elif 'tripe ration' in lower: pri = 4
            elif 'cream pie' in lower: pri = 3
            elif 'egg' in lower: pri = 2
            else: pri = 5
            if pri > best_priority:
                best_priority = pri
                best_letter = letter
        return best_letter

    def _find_healing_potion(self):
        """Find a potion to quaff in an emergency."""
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
            if 'full healing' in lower: pri = 10
            elif 'extra healing' in lower: pri = 9
            elif 'healing' in lower: pri = 8
            elif 'restore ability' in lower: pri = 3
            elif 'speed' in lower: pri = 2
            elif 'blessed' in lower: pri = 4
            elif 'uncursed' in lower: pri = 2
            else: pri = 1
            if pri > best_priority:
                best_priority = pri
                best_letter = letter
        return best_letter

    def _find_wand_letter(self):
        """Find a usable wand in inventory."""
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) == WAND_CLASS:
                lower = item.lower()
                if 'nothing' in lower or 'light' in lower or 'probing' in lower:
                    continue
                if '(0:0)' in lower:
                    continue
                return letter
        return None

    def _find_combat_wand(self):
        """Find the best wand for combat."""
        best = None
        best_value = 0
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
                continue
            lower = item.lower()
            if '(0:0)' in lower:
                continue
            for wand_name, value in WAND_COMBAT_VALUE.items():
                if wand_name in lower and wand_name not in WAND_NEVER_ZAP_AT:
                    if value > best_value:
                        best = letter
                        best_value = value
                    break
            else:
                # Unidentified wand
                if 'of ' not in lower and best is None:
                    best = letter
                    best_value = 3
        return best

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

    def _find_weapon_letter(self):
        """Find current wielded weapon letter."""
        for letter, item in self.inventory.items():
            lower = item.lower()
            if '(weapon in hand)' in lower or '(wielded)' in lower:
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
                continue
            oc = self.inv_oclasses.get(letter, -1)
            if oc in (POTION_CLASS, SCROLL_CLASS, WAND_CLASS, RING_CLASS, AMULET_CLASS):
                result.append(letter)
        return result

    def _find_long_sword_letter(self):
        for letter, item in self.inventory.items():
            if 'long sword' in item.lower() and 'cursed' not in item.lower():
                return letter
        return None

    # ================================================================
    # Corpse helpers (from v4)
    # ================================================================

    def _is_corpse_worth_eating(self, name):
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            return report.safe_to_eat
        return self.food.is_corpse_safe(name, self.resistances)

    def _corpse_has_wanted_intrinsic(self, name):
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            if report.safe_to_eat and report.beneficial_intrinsic:
                return report.beneficial_intrinsic not in self.resistances
            return False
        if hasattr(self, '_intrinsic_corpses_cache'):
            ic = self._intrinsic_corpses_cache
        else:
            ic = INTRINSIC_CORPSES
        for resist, monsters in ic.items():
            if name in monsters:
                resist_name = resist + ' resistance' if 'resistance' not in resist else resist
                if resist_name not in self.resistances:
                    if self.food.is_corpse_safe(name, self.resistances):
                        return True
        return False

    def _extract_corpse_name_from_message(self, msg):
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
        """Find a nearby fresh corpse worth eating."""
        lvl = self.current_level()
        turn = self.blstats.time
        py, px = self.blstats.y, self.blstats.x
        best = None
        best_dist = 9
        best_priority = -1
        for (r, c), (name, kill_turn) in list(lvl.corpse_positions.items()):
            age = turn - kill_turn
            if name not in NEVER_ROT and age > MAX_CORPSE_AGE:
                continue
            dist = _chebyshev(py, px, r, c)
            if dist >= best_dist:
                continue
            priority = 0
            if name == 'wraith': priority = 10
            elif self._corpse_has_wanted_intrinsic(name): priority = 8
            elif name in ('lizard', 'lichen'): priority = 5
            elif self._is_corpse_worth_eating(name):
                if self.blstats.hunger >= HUNGRY: priority = 4
                else: priority = 1
            if priority > best_priority or (priority == best_priority and dist < best_dist):
                best = (r, c)
                best_dist = dist
                best_priority = priority
        if best_priority <= 0:
            return None
        return best

    # ================================================================
    # Strategies
    # ================================================================

    def emergency(self):
        """Handle life-threatening conditions. v4 version with condition bitmask."""
        bl = self.blstats
        if bl is None:
            return False

        conditions = self._get_condition_set()
        turn = bl.time
        can_pray = (turn - self._last_prayer_turn) >= 300

        # === STONING ===
        if 'stoned' in conditions:
            for letter, item in self.inventory.items():
                if 'lizard' in item.lower() and self.inv_oclasses.get(letter, -1) == FOOD_CLASS:
                    self._two_step_eat(letter)
                    return True
            for letter, item in self.inventory.items():
                if 'acid' in item.lower() and self.inv_oclasses.get(letter, -1) == FOOD_CLASS:
                    self._two_step_eat(letter)
                    return True
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True
            return False

        # === SLIMING ===
        if 'slimed' in conditions:
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            return False

        # === STRANGULATION ===
        if 'strangled' in conditions:
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            return False

        # === FOOD POISONING ===
        if 'foodpois' in conditions:
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True
            return False

        # === TERMINAL ILLNESS ===
        if 'termill' in conditions:
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            return False

        # === STARVATION (Weak or worse) ===
        if bl.hunger >= WEAK_HUNGER:
            food_letter = self._find_food_to_eat()
            if food_letter:
                self._two_step_eat(food_letter)
                self._last_eat_turn = turn
                return True
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            # Eat any corpse on ground
            self.step(A.Command.EAT)
            return True

        # === CRITICAL HP ===
        if bl.hp <= max(5, bl.max_hp // 7) or (bl.max_hp > 0 and bl.hp < bl.max_hp * 0.15):
            if can_pray:
                self._last_prayer_turn = turn
                if self.prayer_state:
                    self.prayer_state.update_prayed(turn)
                self.step(A.Command.PRAY)
                return True
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True
            # Elbereth if monsters adjacent
            mons = [(d,r,c,n,m) for d,r,c,n,m in self.get_monsters()
                    if n not in PEACEFUL_NAMES and m not in PEACEFUL_IDS
                    and m not in self._peaceful_monster_ids
                    and (r, c) not in self._peaceful_positions and d <= 1]
            if mons and self._elbereth_cooldown <= 0:
                self._engrave_elbereth()
                return True

        # HP critical fallback (v2 compatible): pray at HP < max/3
        if can_pray and bl.hp < max(8, bl.max_hp // 3):
            self._last_prayer_turn = turn
            if self.prayer_state:
                self.prayer_state.update_prayed(turn)
            self.step(A.Command.PRAY)
            return True

        # Last resort: quaff a potion when HP critical and can't pray
        if not can_pray and bl.hp < max(6, bl.max_hp // 4):
            potion = self._find_healing_potion()
            if potion:
                self._two_step_quaff(potion)
                return True

        return False

    def handle_status_effects(self):
        """Handle non-emergency status effects using unicorn horn. From v4."""
        conditions = self._get_condition_set()
        curable = conditions & {'confused', 'blind', 'stunned', 'hallucinating'}
        if not curable:
            return False
        for letter, item in self.inventory.items():
            if 'unicorn horn' in item.lower():
                self._apply_item(letter)
                return True
        # For blindness, try see invisible potion
        if 'blind' in conditions:
            for letter, item in self.inventory.items():
                if 'see invisible' in item.lower() and self.inv_oclasses.get(letter, -1) == POTION_CLASS:
                    self._two_step_quaff(letter)
                    return True
        return False

    def pray_strategy(self):
        """Use prayer for non-emergency minor troubles when safe. From v4."""
        if self.prayer_state is None:
            return False
        bl = self.blstats
        if bl is None:
            return False
        turn = bl.time
        if turn - self._last_prayer_turn < 300:
            return False

        # Build trouble state
        hunger_val = HungerState.NOT_HUNGRY
        hunger_str = self._get_hunger_state_str()
        if hunger_str == 'hungry': hunger_val = HungerState.HUNGRY
        elif hunger_str == 'weak': hunger_val = HungerState.WEAK
        elif hunger_str in ('fainting', 'fainted'): hunger_val = HungerState.FAINTING
        elif hunger_str == 'satiated': hunger_val = HungerState.SATIATED

        trouble_state = {
            'hp': bl.hp, 'max_hp': bl.max_hp,
            'hunger': hunger_val,
            'condition': self._get_conditions(),
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
        """Priority-based combat system. v4 ThreatDB + corridor retreat,
        built on v2's data structures."""
        acted = False
        for _round in range(50):
            mons = [(d,r,c,n,m) for d,r,c,n,m in self.get_monsters()
                    if n not in PEACEFUL_NAMES and m not in PEACEFUL_IDS
                    and m not in self._peaceful_monster_ids
                    and (r, c) not in self._peaceful_positions]
            if not mons or all(d > 7 for d,_,_,_,_ in mons):
                break

            acted = True
            py, px = self.blstats.y, self.blstats.x
            hp_ratio = self.blstats.hp / max(1, self.blstats.max_hp)

            adj = [(d,r,c,n,m) for d,r,c,n,m in mons if d <= 1]
            actions = []

            # === EMERGENCY CHECK within fight loop ===
            if self.blstats.hp <= max(5, self.blstats.max_hp // 7):
                if self.emergency():
                    continue

            # === INSTAKILL FLEE ===
            for d, r, c, n, m in adj:
                if n in INSTAKILL:
                    flee_dy = _sign(py - r)
                    flee_dx = _sign(px - c)
                    if flee_dy == 0 and flee_dx == 0:
                        flee_dy = 1
                    actions.append((200, ('flee', flee_dy, flee_dx)))

            # === WAIT ON ELBERETH ===
            if self._on_elbereth and adj and hp_ratio < 0.8:
                actions.append((150, ('wait',)))

            # === ELBERETH ===
            if adj and self._elbereth_cooldown <= 0 and not self._on_elbereth:
                adj_threat = 0
                respects = 0
                for d, r, c, n, m in adj:
                    if n in NEVER_MELEE or n in PASSIVE_DANGEROUS:
                        continue
                    adj_threat += 1.0
                    mlevel = nh.permonst(m).mlevel if m < NUMMONS else 0
                    if mlevel > self.blstats.xl + 3:
                        adj_threat += 2.0
                    if n not in ELBERETH_IMMUNE:
                        respects += 1
                if adj_threat > 0 and respects > 0:
                    if self.blstats.hp < 30 or (len(adj) >= 2 and hp_ratio < 0.5):
                        elb_pri = -15 + 20 * adj_threat * (1 - hp_ratio**0.5)
                        actions.append((elb_pri, ('elbereth',)))

            # === TACTICAL FLEE ===
            melee_adj = [x for x in adj if x[3] not in NEVER_MELEE and x[3] not in INSTAKILL]
            if len(melee_adj) >= 2 and hp_ratio < 0.4:
                best_flee = None
                best_threat = len(melee_adj)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < MAP_H and 0 <= nx < MAP_W and self.walkable[ny, nx]:
                        g = int(self.glyphs[ny, nx])
                        if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                            continue
                        if g == GLYPH_OBJ_OFF + 447:
                            continue
                        threat = sum(1 for _,r2,c2,_,_ in melee_adj
                                    if max(abs(r2-ny), abs(c2-nx)) <= 1)
                        if threat < best_threat:
                            best_threat = threat
                            best_flee = (dy, dx)
                if best_flee:
                    flee_pri = 5 + (1 - hp_ratio) * 15
                    actions.append((flee_pri, ('flee', best_flee[0], best_flee[1])))

            # === CORRIDOR RETREAT ===
            if len(adj) >= 3 and hp_ratio < 0.6:
                # Check if a corridor tile is adjacent
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = py + dy, px + dx
                    if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                        continue
                    if not self.walkable[ny, nx]:
                        continue
                    g = int(self.glyphs[ny, nx])
                    if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                        continue
                    cm = _cmap(g)
                    obj_cm = _cmap(int(self.objects[ny, nx])) if self.objects[ny, nx] != -1 else -1
                    if cm in (21, 22) or obj_cm in (21, 22):  # corridors (litcorr, corr)
                        # Count how many adj hostiles would reach us there
                        threat = sum(1 for _,r2,c2,_,_ in adj
                                    if max(abs(r2-ny), abs(c2-nx)) <= 1)
                        if threat < len(adj):
                            actions.append((25, ('flee', dy, dx)))
                            break

            # === MELEE ===
            for d, r, c, n, m in adj:
                info = assess_monster(n, m)
                if info['never_melee']:
                    continue
                if info['instakill']:
                    continue
                if info['peaceful']:
                    continue
                if n in WEIRD and self.blstats.xl < 8:
                    continue

                danger = info['danger']
                # ThreatDB assessment
                if self.threat_db is not None:
                    player_state = {
                        'hp': self.blstats.hp, 'max_hp': self.blstats.max_hp,
                        'ac': self.blstats.ac, 'level': self.blstats.xl,
                        'speed': 12, 'resistances': self.resistances,
                        'has_elbereth_source': True,
                    }
                    threat = self.threat_db.assess_threat(n, player_state)
                    danger = threat.danger_level
                    # If ThreatDB recommends ranged, try that instead
                    if threat.recommended_action == 'ranged' and threat.ranged_preferred:
                        wand = self._find_combat_wand()
                        if wand and _in_line(py, px, r, c):
                            ndy = _sign(r - py)
                            ndx = _sign(c - px)
                            actions.append((danger + 5, ('zap', wand, ndy, ndx)))
                            continue
                        proj = self._find_projectile_letter()
                        if proj and _in_line(py, px, r, c):
                            ndy = _sign(r - py)
                            ndx = _sign(c - px)
                            actions.append((danger + 3, ('throw', proj, ndy, ndx)))
                            continue

                # Level drainers get priority bump
                if n in LEVEL_DRAINERS and 'drain resistance' not in self.resistances:
                    danger = max(danger, 7)
                if n in ENERGY_DRAINERS:
                    danger = max(danger, 8)
                if n in ITEM_STEALERS:
                    danger = max(danger, 6)
                if n in ARMOR_DESTROYERS:
                    danger = max(danger, 6)

                dy, dx = r - py, c - px
                pri = danger
                if self.blstats.hp > 8:
                    pri += 10
                if info.get('weak'):
                    pri = 1
                # Don't melee while on Elbereth
                if self._on_elbereth:
                    pri -= 100
                actions.append((pri, ('melee', dy, dx)))

            # === RANGED ATTACKS (distant) ===
            if not adj:
                wand = self._find_combat_wand()
                proj = self._find_projectile_letter()
                for d, r, c, n, m in mons:
                    if d <= 1 or d > 7:
                        continue
                    if not _in_line(py, px, r, c):
                        continue
                    if not _line_clear(self.glyphs, py, px, r, c):
                        continue
                    ndy = _sign(r - py)
                    ndx = _sign(c - px)
                    if wand:
                        actions.append((8, ('zap', wand, ndy, ndx)))
                        break
                    elif proj:
                        actions.append((6, ('throw', proj, ndy, ndx)))
                        break

            # === APPROACH ===
            if not adj and self.blstats.hp > self.blstats.max_hp * 0.3:
                fight_dis = self._bfs_allow_hostiles()
                approach_range = 12
                total_searches = int(self.search_count.sum())
                if (not self._stairs_down and self._level_turns > 500
                        and total_searches > 100):
                    approach_range = 6
                best_mon = None
                best_d = 999
                for d, r, c, n, m in mons:
                    fd = fight_dis[r, c]
                    if fd != -1 and fd <= approach_range and fd < best_d:
                        best_d = fd
                        best_mon = (r, c)
                if best_mon:
                    actions.append((0, ('approach', best_mon[0], best_mon[1])))

            if not actions:
                break

            # Execute highest priority action
            actions.sort(key=lambda x: -x[0])
            _, best = actions[0]

            if best[0] == 'melee':
                self._move_dir(best[1], best[2])
            elif best[0] == 'flee':
                self._move_dir(best[1], best[2])
            elif best[0] == 'elbereth':
                self._engrave_elbereth()
                for _ in range(3):
                    if self.blstats.hp >= self.blstats.max_hp * 0.5:
                        break
                    self.step(A.Command.SEARCH)
            elif best[0] == 'wait':
                self.step(A.Command.SEARCH)
            elif best[0] == 'zap':
                self._three_step_zap(best[1], best[2], best[3])
            elif best[0] == 'throw':
                self._three_step_throw(best[1], best[2], best[3])
            elif best[0] == 'approach':
                fight_dis = self._bfs_allow_hostiles()
                if not self.step_toward(best[1], best[2], fight_dis):
                    self._greedy_move_toward(best[1], best[2])
            else:
                break

            # Emergency check between fight rounds
            bl = self.blstats
            can_pray_now = (bl.time - self._last_prayer_turn) >= 300
            if can_pray_now and bl.hp < max(8, bl.max_hp // 3):
                self._last_prayer_turn = bl.time
                if self.prayer_state:
                    self.prayer_state.update_prayed(bl.time)
                self.step(A.Command.PRAY)

        return acted

    def eat_corpse_after_kill(self):
        """Step onto a fresh kill's corpse and eat it. v4 expanded."""
        if self._last_kill_name is None:
            return False
        bl = self.blstats
        if bl is None:
            return False
        name = self._last_kill_name
        dy, dx = self._last_kill_dir
        self._last_kill_name = None

        # v4: eat wraith/intrinsic corpses even when not hungry
        should_eat = False
        if bl.hunger >= HUNGRY:
            should_eat = self._is_corpse_worth_eating(name)
        elif name == 'wraith':
            should_eat = True
        elif name in ('lizard', 'lichen'):
            should_eat = True
        elif self._corpse_has_wanted_intrinsic(name):
            should_eat = True

        if not should_eat:
            return False
        if dy == 0 and dx == 0:
            return False
        py, px = bl.y, bl.x
        cy, cx = py + dy, px + dx
        if not (0 <= cy < MAP_H and 0 <= cx < MAP_W):
            return False
        self._move_dir(dy, dx)
        msg_low = self.initial_message.lower() if hasattr(self, 'initial_message') else ''
        if 'corpse' not in msg_low and 'you see here' not in msg_low:
            return True
        self._last_eat_turn = bl.time
        self.step(A.Command.EAT)
        return True

    def eat_ground(self):
        """Eat corpse/food from ground. v4 expanded with intrinsic eating."""
        bl = self.blstats
        if bl is None:
            return False
        if bl.time - self._last_eat_turn < 3:
            return False
        msg = self.initial_message.lower() if hasattr(self, 'initial_message') else ''
        if 'you see here' not in msg and 'there is' not in msg:
            return False

        # Corpse on ground
        if 'corpse' in msg:
            corpse_name = self._extract_corpse_name_from_message(msg)
            if corpse_name:
                should_eat = False
                if bl.hunger >= HUNGRY:
                    should_eat = self._is_corpse_worth_eating(corpse_name)
                elif corpse_name == 'wraith':
                    should_eat = True
                elif corpse_name in ('lizard', 'lichen'):
                    should_eat = True
                elif self._corpse_has_wanted_intrinsic(corpse_name):
                    should_eat = True
                if should_eat:
                    self._last_eat_turn = bl.time
                    self.step(A.Command.EAT)
                    return True
            elif bl.hunger >= HUNGRY:
                if self.food.is_corpse_safe(corpse_name or '', self.resistances):
                    self._last_eat_turn = bl.time
                    self.step(A.Command.EAT)
                    return True
            return False

        # Non-corpse food on ground
        if bl.hunger >= HUNGRY:
            food_kw = ['food ration', 'cram ration', 'lembas wafer', 'k-ration',
                       'c-ration', 'tripe ration', 'candy bar', 'pancake',
                       'fortune cookie', 'tin', 'apple', 'orange', 'pear',
                       'melon', 'banana', 'carrot']
            for kw in food_kw:
                if kw in msg:
                    self._last_eat_turn = bl.time
                    self.step(A.Command.EAT)
                    return True
        return False

    def eat(self):
        """Eat from inventory when hungry. v2 base."""
        bl = self.blstats
        if bl is None or bl.hunger < HUNGRY:
            return False
        if bl.time - self._last_eat_turn < 5:
            return False
        food_letter = self._find_food_to_eat()
        if food_letter is None:
            return False
        self._last_eat_turn = bl.time
        self._two_step_eat(food_letter)
        return True

    def pickup_useful(self):
        """Pick up useful items from ground. v4 expanded categories."""
        msg = self.initial_message.lower() if hasattr(self, 'initial_message') else ''
        if 'you see here' not in msg and 'there are' not in msg:
            return False
        if 'cursed' in msg:
            return False
        # Encumbrance check
        if self._get_encumbrance() >= 2:
            return False
        if self.blstats and self.blstats.time - self._last_pickup_turn < 3:
            return False

        food_kw = ['food ration', 'cram ration', 'lembas wafer', 'k-ration', 'c-ration',
                    'tripe ration', 'tin', 'candy bar', 'pancake', 'fortune cookie',
                    'apple', 'orange', 'pear', 'melon', 'banana', 'carrot']
        armor_kw = ['mail', 'armor', 'helm', 'cloak', 'gloves', 'gauntlets',
                     'boots', 'shoes', 'jacket', 'shield']
        weapon_kw = ['long sword', 'katana', 'silver saber', 'broadsword', 'scimitar',
                      'battle-axe', 'morning star', 'war hammer', 'mace',
                      'two-handed sword', 'trident']
        misc_kw = ['potion', 'scroll', 'wand', 'ring', 'amulet', 'gold piece',
                   'unicorn horn', 'skeleton key', 'lock pick', 'stethoscope',
                   'tinning kit', 'magic marker', 'bag']

        is_useful = (any(w in msg for w in food_kw) or
                     any(w in msg for w in armor_kw) or
                     any(w in msg for w in weapon_kw) or
                     any(w in msg for w in misc_kw))
        if not is_useful:
            return False
        if 'boulder' in msg:
            return False

        self._last_pickup_turn = self.blstats.time
        pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
        if pickup_idx is None:
            return False
        if self._env_step(pickup_idx):
            self._parse_blstats()
            raise AgentFinished()
        self._update_game_state()
        # Auto-equip
        self._auto_equip_check()
        return True

    def _auto_equip_check(self):
        """Equip best available armor and weapons from inventory. v4 expanded."""
        self._parse_inventory()
        acted = False

        # Wield better weapon
        weapon_letter = self.equip.find_best_weapon(self.inventory)
        if weapon_letter:
            self._two_step_wield(weapon_letter)
            acted = True

        # Wear armor (with BUC safety from v4)
        self._parse_inventory()
        armor_letter = self.equip.find_best_armor(self.inventory)
        if armor_letter:
            item_str = self.inventory.get(armor_letter, '')
            lower = item_str.lower()
            is_buc_known = any(w in lower for w in ['blessed', 'uncursed', 'cursed'])
            if is_buc_known and 'cursed' not in lower:
                self._two_step_wear(armor_letter)
                acted = True
            elif not is_buc_known and self.blstats.xl >= 3:
                # Risk wearing unidentified armor after level 3 if no altar
                if not self._altars:
                    self._two_step_wear(armor_letter)
                    acted = True
        return acted

    def dip_excalibur(self):
        """Dip long sword in fountain for Excalibur. v4 with alignment check."""
        bl = self.blstats
        if bl is None or self.has_excalibur:
            return False
        if self.alignment != 'lawful':
            return False
        if bl.xl < 5:
            return False

        sword_letter = self._find_long_sword_letter()
        if sword_letter is None:
            return False

        py, px = bl.y, bl.x
        on_fountain = (py, px) in self._fountains
        if not on_fountain:
            g_here = int(self.glyphs[py, px])
            obj_here = int(self.objects[py, px]) if self.objects[py, px] != -1 else -1
            on_fountain = _cmap(g_here) in _FOUNTAIN or _cmap(obj_here) in _FOUNTAIN

        if on_fountain:
            self._two_step_dip(sword_letter)
            self._parse_inventory()
            if any('Excalibur' in s for s in self.inventory.values()):
                self.has_excalibur = True
            if (py, px) in self._fountains:
                g = int(self.glyphs[py, px]) if self.glyphs is not None else 0
                if _cmap(g) not in _FOUNTAIN:
                    self._fountains.discard((py, px))
            return True

        # Navigate to nearest fountain
        if self._fountains:
            dis = self.bfs()
            best_f = None
            best_d = 999
            for fy, fx in self._fountains:
                d = dis[fy, fx]
                if d != -1 and d < best_d:
                    best_d = d
                    best_f = (fy, fx)
            if best_f:
                if self.step_toward(best_f[0], best_f[1], dis):
                    return True
        return False

    def altar_buc_strategy(self):
        """Drop items on altar to learn BUC status. From v4."""
        py, px = self.blstats.y, self.blstats.x
        if (py, px) not in self._altars:
            return False
        items_to_test = self._find_unided_items()
        if not items_to_test:
            return False
        lvl = self.current_level()
        tested = 0
        for letter in items_to_test[:5]:
            if letter not in self.inventory:
                continue
            self._two_step_drop(letter)
            lvl.altar_tested_items.add(letter)
            tested += 1
        if tested > 0:
            pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
            if pickup_idx is not None:
                if self._env_step(pickup_idx):
                    self._parse_blstats()
                    raise AgentFinished()
                self._update_game_state()
            return True
        return False

    def inventory_management(self):
        """Drop junk when encumbered. From v4."""
        mons = self.get_monsters()
        adj = [x for x in mons if x[0] <= 1]
        if adj:
            return False
        if self._get_encumbrance() < 3:
            return False
        worst_letter = None
        worst_priority = 999
        for letter, item in self.inventory.items():
            lower = item.lower()
            oc = self.inv_oclasses.get(letter, -1)
            if '(weapon in hand)' in lower or '(wielded)' in lower:
                continue
            if '(being worn)' in lower:
                continue
            if 'excalibur' in lower:
                continue
            if oc == FOOD_CLASS: pri = 10
            elif oc == POTION_CLASS:
                pri = 9 if 'healing' in lower else 5
            elif oc == SCROLL_CLASS: pri = 5
            elif oc == WAND_CLASS: pri = 6
            elif oc == RING_CLASS: pri = 4
            elif oc == AMULET_CLASS: pri = 7
            elif oc == GEM_CLASS: pri = 2
            elif oc == WEAPON_CLASS: pri = 3
            elif oc == ARMOR_CLASS: pri = 3
            else: pri = 1
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

    def read_scroll_strategy(self):
        """Strategic scroll reading. From v4."""
        mons = self.get_monsters()
        adj = [x for x in mons if x[0] <= 1]
        if adj:
            return False
        conditions = self._get_condition_set()
        is_confused = 'confused' in conditions
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != SCROLL_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue
            # Identify
            if 'identify' in lower:
                unided = self._find_unided_items()
                if unided:
                    self._read_scroll(letter)
                    return True
            # Remove curse
            if 'remove curse' in lower:
                has_cursed = any('cursed' in self.inventory.get(l, '').lower()
                               and ('(being worn)' in self.inventory.get(l, '') or
                                    '(weapon in hand)' in self.inventory.get(l, ''))
                               for l in self.inventory)
                if has_cursed:
                    self._read_scroll(letter)
                    return True
            # Enchant weapon
            if 'enchant weapon' in lower and not is_confused:
                if self._find_weapon_letter():
                    self._read_scroll(letter)
                    return True
            # Enchant armor
            if 'enchant armor' in lower and not is_confused:
                has_armor = any('(being worn)' in self.inventory.get(l, '')
                              for l in self.inventory)
                if has_armor:
                    self._read_scroll(letter)
                    return True
        return False

    def quaff_potion_strategy(self):
        """Strategic potion drinking. From v4."""
        mons = self.get_monsters()
        adj = [x for x in mons if x[0] <= 1]
        if adj:
            return False
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != POTION_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue
            if 'gain level' in lower:
                self._two_step_quaff(letter)
                return True
            if 'gain ability' in lower:
                if self.blstats.hp > self.blstats.max_hp * 0.8:
                    self._two_step_quaff(letter)
                    return True
            if 'speed' in lower and 'speed' not in self.resistances:
                self._two_step_quaff(letter)
                self.resistances.add('speed')
                return True
        return False

    def use_wand_strategy(self):
        """Strategic wand use. From v4."""
        mons = self.get_monsters()
        adj = [x for x in mons if x[0] <= 1]
        if adj:
            return False
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
                continue
            lower = item.lower()
            if '(0:0)' in lower:
                continue
            # Speed monster at self
            if 'speed monster' in lower and 'speed' not in self.resistances:
                zap_idx = self._val2idx.get(int(A.Command.ZAP))
                if zap_idx is not None:
                    if self._env_step(zap_idx):
                        self._parse_blstats(); raise AgentFinished()
                    w_idx = self._val2idx.get(ord(letter))
                    if w_idx is not None:
                        if self._env_step(w_idx):
                            self._parse_blstats(); raise AgentFinished()
                        dot_idx = self._val2idx.get(ord('.'))
                        if dot_idx is not None:
                            if self._env_step(dot_idx):
                                self._parse_blstats(); raise AgentFinished()
                    self._update_game_state()
                    self.resistances.add('speed')
                    return True
        return False

    def engrave_test_wand(self):
        """Engrave-test unidentified wands. From v4."""
        if self.appearance_tracker is None:
            return False
        mons = self.get_monsters()
        adj = [x for x in mons if x[0] <= 1]
        if adj:
            return False
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) != WAND_CLASS:
                continue
            lower = item.lower()
            if 'of ' in lower:
                continue
            if '(0:' in lower:
                continue
            if letter in self._wands_tested:
                continue
            # Engrave-test this wand
            engrave_idx = self._val2idx.get(int(A.Command.ENGRAVE))
            if engrave_idx is None:
                continue
            if self._env_step(engrave_idx):
                self._parse_blstats(); raise AgentFinished()
            misc = self.obs.get('misc', [0, 0, 0])
            msg_raw = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').lower()
            if misc[0] and 'add to' in msg_raw:
                n_idx = self._val2idx.get(ord('n'))
                if n_idx is not None:
                    if self._env_step(n_idx):
                        self._parse_blstats(); raise AgentFinished()
            w_idx = self._val2idx.get(ord(letter))
            if w_idx is not None:
                if self._env_step(w_idx):
                    self._parse_blstats(); raise AgentFinished()
            x_idx = self._val2idx.get(ord('x'))
            if x_idx is not None:
                if self._env_step(x_idx):
                    self._parse_blstats(); raise AgentFinished()
            cr_idx = self._val2idx.get(13)
            if cr_idx is not None:
                if self._env_step(cr_idx):
                    self._parse_blstats(); raise AgentFinished()
            self._update_game_state()
            self._wands_tested.add(letter)
            return True
        return False

    def emergency_escape(self):
        """Emergency escape when trapped with many deadly monsters. From v4."""
        bl = self.blstats
        if bl is None:
            return False
        hp_ratio = bl.hp / max(1, bl.max_hp)
        mons = self.get_monsters()
        adj = [(d,r,c,n,m) for d,r,c,n,m in mons
               if n not in PEACEFUL_NAMES and m not in PEACEFUL_IDS
               and d <= 1]
        if hp_ratio >= 0.2 or len(adj) < 3:
            return False

        # Wand of teleportation at self
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) == WAND_CLASS:
                lower = item.lower()
                if 'teleportation' in lower and '(0:0)' not in lower:
                    zap_idx = self._val2idx.get(int(A.Command.ZAP))
                    if zap_idx is not None:
                        if self._env_step(zap_idx):
                            self._parse_blstats(); raise AgentFinished()
                        w_idx = self._val2idx.get(ord(letter))
                        if w_idx is not None:
                            if self._env_step(w_idx):
                                self._parse_blstats(); raise AgentFinished()
                            dot_idx = self._val2idx.get(ord('.'))
                            if dot_idx is not None:
                                if self._env_step(dot_idx):
                                    self._parse_blstats(); raise AgentFinished()
                        self._update_game_state()
                        return True

        # Scroll of teleportation
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) == SCROLL_CLASS:
                lower = item.lower()
                if 'teleportation' in lower and 'cursed' not in lower:
                    self._read_scroll(letter)
                    return True

        # Wand of digging downward
        for letter, item in self.inventory.items():
            if self.inv_oclasses.get(letter, -1) == WAND_CLASS:
                lower = item.lower()
                if 'digging' in lower and '(0:0)' not in lower:
                    zap_idx = self._val2idx.get(int(A.Command.ZAP))
                    if zap_idx is not None:
                        if self._env_step(zap_idx):
                            self._parse_blstats(); raise AgentFinished()
                        w_idx = self._val2idx.get(ord(letter))
                        if w_idx is not None:
                            if self._env_step(w_idx):
                                self._parse_blstats(); raise AgentFinished()
                            down_idx = self._val2idx.get(ord('>'))
                            if down_idx is not None:
                                if self._env_step(down_idx):
                                    self._parse_blstats(); raise AgentFinished()
                        self._update_game_state()
                        return True
        return False

    def explore(self):
        """Explore: open doors, go to frontier, find stairs, search, descend.
        v2 structure with v4 additions (altar nav, better descent)."""
        py, px = self.blstats.y, self.blstats.x

        # Sokoban: solve puzzle if in Sokoban dungeon
        if self.blstats.dungeon_number == DUNGEON_SOKOBAN:
            if self._solve_sokoban():
                return
            lvl = self.current_level()
            if lvl.soko_solution is not None and lvl.soko_step >= len(lvl.soko_solution):
                if self._stairs_up:
                    fight_dis = self._bfs_allow_hostiles()
                    for uy, ux in self._stairs_up:
                        if (py, px) == (uy, ux):
                            self.step(A.MiscDirection.UP)
                            return
                        if fight_dis[uy, ux] != -1:
                            self.step_toward(uy, ux, fight_dis)
                            return

        # Update milestones
        if self.milestone == MILESTONE_FARM_DL1 and self.blstats.xl >= 3:
            self.milestone = MILESTONE_DESCEND
        elif self.milestone == MILESTONE_DESCEND and self.blstats.depth >= 5:
            self.milestone = MILESTONE_PUSH_DEEP

        # 0. On downstairs: descend based on milestone
        if self._on_stairs_down() and self.blstats.hp > self.blstats.max_hp * 0.5:
            descent_timer = max(40, 200 - self.blstats.depth * 30)
            time_ok = self._level_turns > descent_timer

            if self.milestone == MILESTONE_FARM_DL1:
                xl_ok = self.blstats.xl >= 5
            elif self.milestone == MILESTONE_DESCEND:
                xl_ok = self.blstats.xl >= self.blstats.depth
            else:
                xl_ok = True

            if xl_ok and time_ok:
                self.step(A.MiscDirection.DOWN)
                return

        # Excalibur: only dip if already on fountain
        if not self.has_excalibur and self.alignment == 'lawful' and self.blstats.xl >= 5:
            if (py, px) in self._fountains:
                self.dip_excalibur()
                return

        dis = self.bfs()

        # 1. Adjacent closed doors: open or kick them
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = py+dy, px+dx
            if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                g = int(self.glyphs[nr, nc])
                if _cmap(g) in _CLOSED_DOOR and self.door_attempts[nr, nc] < 10:
                    self.door_attempts[nr, nc] += 1
                    dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W'}
                    name = dmap.get((dy, dx))
                    if name and name in self._name2idx:
                        if self.door_attempts[nr, nc] <= 2:
                            self.step(self._name2idx[name])
                            msg_low = self.message.lower()
                            if 'locked' not in msg_low and _cmap(int(self.glyphs[nr, nc])) not in _CLOSED_DOOR:
                                return
                        self._kick_dir(dy, dx)
                        if _cmap(int(self.glyphs[nr, nc])) not in _CLOSED_DOOR:
                            return
                        return

        # 2. Go to nearest closed door
        best_door, best_dd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if _cmap(int(self.glyphs[r, c])) in _CLOSED_DOOR and self.door_attempts[r, c] < 5:
                    d = dis[r, c]
                    if d != -1 and d < best_dd:
                        best_dd = d
                        best_door = (r, c)
        if best_door and best_dd > 1:
            if self.step_toward(best_door[0], best_door[1], dis):
                return

        # 3. Check if we should force descent
        if self.milestone == MILESTONE_FARM_DL1:
            xl_ready = self.blstats.xl >= 3
        elif self.milestone == MILESTONE_DESCEND:
            xl_ready = self.blstats.xl >= self.blstats.depth
        else:
            xl_ready = True
        descent_timer = max(40, 200 - self.blstats.depth * 30)
        force_descend = self._level_turns > descent_timer and xl_ready

        # 3a. Navigate to stairs if force_descend
        if force_descend and self._stairs_down and self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self._bfs_allow_hostiles()
            best_s, best_sd = None, 999
            for sy, sx in self._stairs_down:
                d = fight_dis[sy, sx]
                if d != -1 and d < best_sd:
                    best_sd = d
                    best_s = (sy, sx)
            if best_s:
                if best_sd == 0:
                    self.step(A.MiscDirection.DOWN)
                    return
                moved = self.step_toward(best_s[0], best_s[1], fight_dis)
                if moved:
                    return
                self._greedy_move_toward(best_s[0], best_s[1])
                return

        # 3b. Navigate to Excalibur fountain (from v4)
        if (not self.has_excalibur and self.alignment == 'lawful'
                and self.blstats.xl >= 5 and self.has_long_sword
                and self._fountains):
            best_f = None
            best_fd = 999
            for fy, fx in self._fountains:
                d = dis[fy, fx]
                if d != -1 and d < best_fd:
                    best_fd = d
                    best_f = (fy, fx)
            if best_f and best_fd > 0:
                if self.step_toward(best_f[0], best_f[1], dis):
                    return

        # 4. Go to nearest frontier
        best_f, best_fd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or d >= best_fd or not self.walkable[r, c]:
                    continue
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not self.seen[nr, nc]:
                            best_fd = d
                            best_f = (r, c)
                            break
                    if best_f and dis[best_f[0], best_f[1]] == best_fd:
                        break
        if best_f:
            if self.step_toward(best_f[0], best_f[1], dis):
                ny, nx = self.blstats.y, self.blstats.x
                if self.search_count[ny, nx] < 3:
                    adj_wall = False
                    for dy2 in (-1, 0, 1):
                        for dx2 in (-1, 0, 1):
                            if dy2 == 0 and dx2 == 0:
                                continue
                            nr2, nc2 = ny + dy2, nx + dx2
                            if 0 <= nr2 < MAP_H and 0 <= nc2 < MAP_W:
                                cm2 = _cmap(int(self.glyphs[nr2, nc2]))
                                if cm2 in _WALL or cm2 == 0:
                                    adj_wall = True
                                    break
                        if adj_wall:
                            break
                    if adj_wall:
                        self.search_count[ny, nx] += 1
                        self.step(A.Command.SEARCH)
                return

        # 5. No frontier: navigate to stairs
        if self._stairs_down and self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self._bfs_allow_hostiles()
            best_s, best_sd = None, 999
            for sy, sx in self._stairs_down:
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

        # 5a. Navigate to altar (for BUC testing, from v4)
        if self._altars and self._find_unided_items():
            best_a = None
            best_ad = 999
            for ay, ax in self._altars:
                d = dis[ay, ax]
                if d != -1 and d < best_ad:
                    best_ad = d
                    best_a = (ay, ax)
            if best_a and best_ad > 0 and best_ad <= 20:
                if self.step_toward(best_a[0], best_a[1], dis):
                    return

        # 6. Search near walls (v2 verbatim)
        best_s, best_sp = None, float('-inf')
        for r in range(MAP_H):
            for c in range(MAP_W):
                if not self.walkable[r, c] or dis[r, c] == -1:
                    continue
                stones = 0
                walls = 0
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            obj_val = int(self.objects[nr, nc])
                            if obj_val == -1:
                                obj_val = int(self.glyphs[nr, nc])
                            cm2 = _cmap(obj_val)
                            if cm2 == 0:
                                stones += 1
                            elif cm2 in _WALL:
                                walls += 1
                if stones == 0 and walls == 0:
                    continue
                sc = self.search_count[r, c]
                p = -1.0 - sc * sc * 2
                obj_here = int(self.objects[r, c]) if self.objects[r, c] != -1 else -1
                if _cmap(obj_here) in _DOOR and stones >= 3:
                    p += 250
                cardinal_w = sum(1 for dy3, dx3 in [(-1,0),(1,0),(0,-1),(0,1)]
                                if 0 <= r+dy3 < MAP_H and 0 <= c+dx3 < MAP_W
                                and self.walkable[r+dy3, c+dx3])
                if cardinal_w <= 1:
                    p += 250
                p -= dis[r, c] * 2
                if p > best_sp:
                    best_sp = p
                    best_s = (r, c)

        if best_s and best_s != (py, px):
            if self.step_toward(best_s[0], best_s[1], dis):
                return

        # 7. Search at current position
        search_rounds = min(5, max(1, 12 - self.search_count[py, px]))
        for _ in range(search_rounds):
            self.search_count[py, px] += 1
            self.step(A.Command.SEARCH)
            for dy2 in (-1, 0, 1):
                for dx2 in (-1, 0, 1):
                    if dy2 == 0 and dx2 == 0:
                        continue
                    nr, nc = py + dy2, px + dx2
                    if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                        g = int(self.glyphs[nr, nc])
                        cm = _cmap(g)
                        if cm in _CLOSED_DOOR or cm in _WALKABLE:
                            if not self.seen[nr, nc] or cm in _CLOSED_DOOR:
                                return

    # ================================================================
    # Main loop
    # ================================================================

    def main(self):
        try:
            obs, info = self.env.reset(seed=self.seed)
            self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            bl = obs.get('blstats')
            if bl is not None:
                self._raw_bl = bl.copy()
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

            stall_turn = -1
            stall_count = 0
            while True:
                try:
                    # Check descent after every action
                    if self._on_stairs_down() and self._level_turns > max(40, 200 - self.blstats.depth * 30):
                        if self.milestone == MILESTONE_FARM_DL1:
                            xl_ok = self.blstats.xl >= 5
                        elif self.milestone == MILESTONE_DESCEND:
                            xl_ok = self.blstats.xl >= self.blstats.depth
                        else:
                            xl_ok = True
                        if xl_ok and self.blstats.hp > self.blstats.max_hp * 0.5:
                            self.step(A.MiscDirection.DOWN)
                            continue

                    # Stall detection
                    cur_turn = self.blstats.time if self.blstats else 0
                    if cur_turn == stall_turn:
                        stall_count += 1
                        if stall_count > 3:
                            self.step(A.Command.SEARCH)
                            stall_count = 0
                            continue
                    else:
                        stall_turn = cur_turn
                        stall_count = 0

                    # Stuck detection (v4)
                    if self._stuck_count > 20:
                        self._stuck_count = 0
                        self.step(A.Command.SEARCH)
                        continue

                    # ===== STRATEGY CHAIN (v4 expanded) =====
                    if self.emergency():
                        continue
                    if self.emergency_escape():
                        continue
                    if self.handle_status_effects():
                        continue
                    if self.pray_strategy():
                        continue
                    if self.fight():
                        continue
                    if self.eat_corpse_after_kill():
                        continue
                    if self.eat_ground():
                        continue
                    if self.eat():
                        continue
                    if self.pickup_useful():
                        continue
                    if self._auto_equip_check():
                        continue
                    if self.inventory_management():
                        continue
                    if self.dip_excalibur():
                        continue
                    if self.altar_buc_strategy():
                        continue
                    if self.read_scroll_strategy():
                        continue
                    if self.quaff_potion_strategy():
                        continue
                    if self.use_wand_strategy():
                        continue
                    if self.engrave_test_wand():
                        continue
                    self.explore()
                except RuntimeError as e:
                    if 'finished' in str(e).lower():
                        break
                    try:
                        self.step(A.Command.SEARCH)
                    except (AgentFinished, RuntimeError):
                        break

        except (AgentFinished, RuntimeError):
            pass
        return self.score


# Entry point for evaluation
def run_agent(env, seed=None, verbose=False):
    """Run one episode of the agent. Returns final score."""
    agent = AgentV5(env, seed=seed, verbose=verbose)
    agent.main()
    return agent.score
