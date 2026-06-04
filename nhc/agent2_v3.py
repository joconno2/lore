"""LORE Agent v3: Full AutoAscend-level expert system.

Clean room rewrite integrating ALL subsystems: combat, navigation, food,
equipment, prayer, item identification, Sokoban, Excalibur, strategy.

The agent DRIVES the env. step() calls env.step() and handles all prompts
internally (yn, getlin, xwait, --More--). The main loop runs strategies
in priority order until the episode ends.

Architecture mirrors AutoAscend's agent.py + global_logic.py:
  emergency > pray > fight > eat > equip > excalibur > sokoban > explore > descend
"""
from __future__ import annotations

import numpy as np
from collections import namedtuple, deque

import nle.nethack as nh
from nle.nethack import actions as A

from nhc.food import (
    FoodManager, UNSAFE_CORPSES, POISONOUS_CORPSES, NO_CORPSE,
    UNDEAD_FRAGMENTS, NEVER_ROT, MAX_CORPSE_AGE, INTRINSIC_CORPSES,
)
from nhc.equipment import EquipmentManager, WEAPON_DATA, ARMOR_DATA, SLOT_KEYWORDS
from nhc.fight import (
    assess_monster, NEVER_MELEE, INSTAKILL, PEACEFUL_NAMES, PEACEFUL_IDS,
    ONLY_RANGED_SLOW, EXPLODING, WEAK, WEIRD, FAST_MONSTERS,
)
from nhc.sokoban import match_sokoban_level

# ================================================================
# Constants
# ================================================================

BLStats = namedtuple('BLStats',
    'x y str_pct str dex con int wis cha score '
    'hp max_hp depth gold energy max_energy ac monster_level '
    'xl xp time hunger carrying_capacity dungeon_number level_number '
    'condition align')

GLYPH_MON_OFF = 0
GLYPH_PET_OFF = 381
GLYPH_INVIS_OFF = 762
GLYPH_DETECT_OFF = 763
GLYPH_BODY_OFF = 1144
GLYPH_RIDDEN_OFF = 1525
GLYPH_OBJ_OFF = 1906
GLYPH_CMAP_OFF = 2359
NUMMONS = 381
MAP_H, MAP_W = 21, 79

# CMAP indices (offset from GLYPH_CMAP_OFF)
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
SS_NDOOR = 12
SS_VODOOR = 13
SS_HODOOR = 14
SS_VCDOOR = 15
SS_HCDOOR = 16
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

_WALKABLE = frozenset({
    SS_NDOOR, SS_VODOOR, SS_HODOOR,
    SS_ROOM, SS_DARKROOM, SS_CORR, SS_LITCORR,
    SS_UPSTAIR, SS_DNSTAIR, SS_UPLADDER, SS_DNLADDER,
    SS_ALTAR, SS_GRAVE, SS_THRONE, SS_SINK, SS_FOUNTAIN,
    SS_ICE,
})
_CLOSED_DOOR = frozenset({SS_VCDOOR, SS_HCDOOR})
_WALL = frozenset(range(1, 12))
_DOOR = frozenset({SS_NDOOR, SS_VODOOR, SS_HODOOR, SS_VCDOOR, SS_HCDOOR})
_STAIRS_DOWN = frozenset({SS_DNSTAIR, SS_DNLADDER})
_STAIRS_UP = frozenset({SS_UPSTAIR, SS_UPLADDER})
_FOUNTAIN = frozenset({SS_FOUNTAIN})
_ALTAR = frozenset({SS_ALTAR})
_POOL_LAVA = frozenset({SS_POOL, SS_LAVA, 41})  # pool, lava, water

# Hunger states
SATIATED = 0
NOT_HUNGRY = 1
HUNGRY = 2
WEAK = 3
FAINTING = 4

# Object classes
FOOD_CLASS = 7
WEAPON_CLASS = 3
ARMOR_CLASS = 6
POTION_CLASS = 10
WAND_CLASS = 9
SCROLL_CLASS = 8
RING_CLASS = 5
AMULET_CLASS = 4
TOOL_CLASS = 11
GEM_CLASS = 15
GOLD_CLASS = 16

# Condition bitmask flags (NLE)
COND_STONE = 0x00000001
COND_SLIME = 0x00000002
COND_STRNGL = 0x00000004
COND_FOODPOIS = 0x00000008
COND_TERMILL = 0x00000010
COND_BLIND = 0x00000020
COND_DEAF = 0x00000040
COND_STUN = 0x00000080
COND_CONF = 0x00000100
COND_HALLU = 0x00000200

# Boulder glyph
BOULDER_GLYPH = GLYPH_OBJ_OFF + 447

# Dungeon identifiers
DUNGEON_DOOM = 0
DUNGEON_MINES = 2
DUNGEON_SOKOBAN = 4


class AgentFinished(Exception):
    pass


def _cmap(g):
    """Return CMAP index if glyph is a dungeon feature, else -1."""
    idx = g - GLYPH_CMAP_OFF
    if 0 <= idx < 87:
        return idx
    return -1


def _is_monster(g):
    """True for wild monster glyphs."""
    return GLYPH_MON_OFF <= g < GLYPH_PET_OFF


def _is_pet(g):
    """True for pet glyphs."""
    return GLYPH_PET_OFF <= g < GLYPH_INVIS_OFF


def _is_any_monster(g):
    """True for any creature glyph (wild, pet, detected, ridden)."""
    if GLYPH_MON_OFF <= g < GLYPH_INVIS_OFF:
        return True
    if GLYPH_DETECT_OFF <= g < GLYPH_BODY_OFF:
        return True
    if GLYPH_RIDDEN_OFF <= g < GLYPH_OBJ_OFF:
        return True
    return False


def _mon_id(g):
    """Extract monster ID from glyph, or -1."""
    if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
        return g - GLYPH_MON_OFF
    if GLYPH_PET_OFF <= g < GLYPH_INVIS_OFF:
        return g - GLYPH_PET_OFF
    if GLYPH_DETECT_OFF <= g < GLYPH_BODY_OFF:
        return g - GLYPH_DETECT_OFF
    if GLYPH_RIDDEN_OFF <= g < GLYPH_OBJ_OFF:
        return g - GLYPH_RIDDEN_OFF
    return -1


def _mon_name(mid):
    """Get monster name from ID."""
    if 0 <= mid < NUMMONS:
        return nh.permonst(mid).mname
    return f"mon{mid}"


# Elbereth-immune monsters (from combat.py)
_ELBERETH_IMMUNE_NAMES = {
    "Death", "Pestilence", "Famine",
    "Wizard of Yendor", "Medusa", "minotaur",
    "Demogorgon", "Asmodeus", "Baalzebub", "Orcus", "Juiblex",
    "Yeenoghu", "Dispater", "Geryon",
    "shopkeeper", "guard", "aligned priest", "high priest",
    "Archon",
}

# Additional dangerous corpses beyond UNSAFE_CORPSES
_EXTRA_UNSAFE = {
    "bat", "giant bat",  # stun
    "yellow mold",  # hallucination
    "small mimic", "large mimic", "giant mimic",  # paralysis
}

# Monster DB for melee priority
_MONSTER_MLEVEL = {}
_MONSTER_SPEED = {}
_MONSTER_AC = {}


def _init_monster_db():
    """Cache monster stats from NLE."""
    if _MONSTER_MLEVEL:
        return
    for i in range(NUMMONS):
        m = nh.permonst(i)
        _MONSTER_MLEVEL[m.mname] = m.mlevel
        _MONSTER_SPEED[m.mname] = m.mmove
        _MONSTER_AC[m.mname] = m.ac


# ================================================================
# Milestone system (AutoAscend global_logic.py)
# ================================================================

MILESTONE_FARM_DL1 = 0
MILESTONE_FIND_EXCALIBUR = 1
MILESTONE_EXPLORE_MINES = 2
MILESTONE_SOKOBAN = 3
MILESTONE_PUSH_DEEP = 4


# ================================================================
# Level class: persistent per-level state
# ================================================================

class Level:
    """Persistent per-level state. Survives when the agent leaves and returns."""
    __slots__ = (
        'dungeon_number', 'level_number',
        'seen', 'walkable', 'objects', 'search_count', 'door_attempts',
        'stairs_down', 'stairs_up', 'fountains', 'altars',
        'turns_spent', 'stair_dest',
        'soko_solution', 'soko_offset', 'soko_step', 'soko_matched',
        'items_on_ground', 'corpse_positions',
    )

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
        self.stair_dest = {}
        self.soko_solution = None
        self.soko_offset = (0, 0)
        self.soko_step = 0
        self.soko_matched = False
        self.items_on_ground = {}  # (y, x) -> set of glyph IDs
        self.corpse_positions = {}  # (y, x) -> (monster_name, turn_killed)

    def key(self):
        return (self.dungeon_number, self.level_number)


# ================================================================
# AgentV3: the full expert system
# ================================================================

class AgentV3:
    def __init__(self, env, seed=None, verbose=False):
        self.env = env
        self.seed = seed
        self.verbose = verbose

        # Build action lookups from NLE canonical action list
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

        _init_monster_db()

        # Subsystems
        self.food = FoodManager()
        self.equip = EquipmentManager()

        # Core state
        self.obs = None
        self.blstats = None
        self.glyphs = None
        self.chars = None
        self.message = ''
        self.initial_message = ''
        self.score = 0.0
        self.step_count = 0
        self._last_turn = -1
        self._raw_bl = None

        # Multi-level state
        self.levels = {}  # (dungeon_number, level_number) -> Level
        self._prev_level_key = None

        # Inventory
        self.inventory = {}
        self.inv_oclasses = {}
        self.inv_glyphs = {}

        # Character state
        self.resistances = {"cold resistance"}  # Valkyrie starts with cold res
        self.has_excalibur = False
        self.alignment = 1  # lawful (Valkyrie)

        # Prayer tracking
        self._last_prayer_turn = -1000
        self._prayer_timeout = 300  # initial timeout

        # Food tracking
        self._last_eat_turn = -100

        # Combat state
        self._peaceful_positions = set()
        self._peaceful_monster_ids = set()
        self._last_move_dir = (0, 0)
        self._last_kill_name = None
        self._last_kill_dir = (0, 0)
        self._elbereth_turns = 0  # turns since last Elbereth write

        # Equipment state
        self._wielded_letter = None
        self._worn_slots = {}  # slot_name -> letter
        self._cursed_slots = set()  # letters known cursed

        # Pet tracking
        self._pet_pos = None
        self._pet_alive = True

        # Strategy
        self.milestone = MILESTONE_FARM_DL1
        self._mines_entrance = None  # (dungeon_number, level_number)
        self._sokoban_entrance = None

        # Ranged combat
        self._projectile_letters = {}  # letter -> item_name
        self._wand_letters = {}  # letter -> item_name

        # Item identification (lightweight, no JSON dependency)
        self._identified_items = {}  # appearance -> identity
        self._buc_status = {}  # inv letter -> 'blessed'/'uncursed'/'cursed'/None
        self._shop_prices = {}  # appearance -> observed_price

        # Stall detection
        self._stall_turn = -1
        self._stall_count = 0

        # Debug counters
        self._yn_count = 0
        self._prompt_steps = 0

    # ================================================================
    # Level management
    # ================================================================

    def current_level(self):
        if self.blstats is None:
            key = (0, 1)
        else:
            key = (self.blstats.dungeon_number, self.blstats.level_number)
        if key not in self.levels:
            self.levels[key] = Level(*key)
        return self.levels[key]

    @property
    def lvl(self):
        return self.current_level()

    # ================================================================
    # Core: env stepping and prompt handling
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
        """Send action to env, handle prompts iteratively."""
        if isinstance(action, str):
            assert len(action) == 1
            idx = self._val2idx.get(ord(action))
        elif type(action) is int and action < len(self.actions):
            idx = action
        else:
            idx = self._val2idx.get(int(action))

        if idx is None:
            return

        if self._env_step(idx):
            self._parse_blstats()
            raise AgentFinished()

        raw_msg = self.obs.get('message', b'')
        self.initial_message = bytes(raw_msg).decode('latin-1', errors='replace').replace('\x00', '').strip()

        for _ in range(200):
            msg_raw = self.obs.get('message', b'')
            self.message = bytes(msg_raw).decode('latin-1', errors='replace').replace('\x00', '').strip()
            misc = self.obs.get('misc', [0, 0, 0])

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

            # yn prompt
            if misc[0]:
                self._yn_count += 1
                resp = self._handle_yn_prompt()
                if resp is not None:
                    if self._env_step(resp):
                        self._parse_blstats()
                        raise AgentFinished()
                continue

            # getlin prompt
            if misc[1]:
                resp = self._handle_getlin_prompt()
                if self._env_step(resp):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # xwait
            if misc[2]:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # --More--
            if '--More--' in self.message:
                if self._env_step(self._val2idx.get(13, self._val2idx.get(32, 0))):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            break

        self._update_game_state()

    def _handle_yn_prompt(self):
        """Decide response to yn prompt. Returns action index."""
        msg = self.message

        # Don't attack peacefuls
        if 'Really attack' in msg:
            if self.blstats and self._last_move_dir != (0, 0):
                dy, dx = self._last_move_dir
                py, px = self.blstats.y, self.blstats.x
                ty, tx = py + dy, px + dx
                if 0 <= ty < MAP_H and 0 <= tx < MAP_W:
                    self._peaceful_positions.add((ty, tx))
                    g = int(self.glyphs[ty, tx]) if self.glyphs is not None else 0
                    if _is_monster(g):
                        self._peaceful_monster_ids.add(_mon_id(g))
            return self._val2idx.get(ord('n'))

        # Don't force locks
        if 'force the lock' in msg:
            return self._val2idx.get(ord('n'))

        # Confirm prayer
        if 'pray' in msg.lower() and ('Are you sure' in msg or 'really pray' in msg.lower()):
            return self._val2idx.get(ord('y'))

        # Don't eat if satiated (choking risk)
        if ('eat' in msg.lower() and 'satiated' in msg.lower()) or 'choke' in msg.lower():
            return self._val2idx.get(ord('n'))

        # Don't eat rotten/tainted food
        if 'old ' in msg.lower() or 'rotten' in msg.lower() or 'tainted' in msg.lower():
            return self._val2idx.get(ord('n'))

        # Eat corpses from ground: confirm
        if 'eat it?' in msg.lower() or 'eat this?' in msg.lower():
            return self._val2idx.get(ord('y'))

        # "What do you want to..." menus: ESC out if unexpected
        menu_prefixes = [
            'What do you want to call',
            'What do you want to use',
            'What do you want to name',
        ]
        if any(s in msg for s in menu_prefixes):
            return self._val2idx.get(27)  # ESC

        # Confirm sacrifice
        if 'sacrifice' in msg.lower():
            return self._val2idx.get(ord('y'))

        # Don't overeat
        if 'stop eating' in msg.lower() or 'Continue eating' in msg:
            return self._val2idx.get(ord('y'))

        # Loot: don't bother
        if 'loot' in msg.lower() and '[yn]' in msg:
            return self._val2idx.get(ord('n'))

        # Default: yes
        return self._val2idx.get(ord('y'))

    def _handle_getlin_prompt(self):
        """Handle text-entry prompts. Returns action index."""
        msg = self.message

        # Eat prompt: find food letter
        if 'What do you want to eat' in msg:
            food_letter = self._find_best_food_letter()
            if food_letter:
                return self._val2idx.get(ord(food_letter), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)  # ESC

        # Wield prompt
        if 'What do you want to wield' in msg:
            wep = self._find_best_wield_letter()
            if wep:
                return self._val2idx.get(ord(wep), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Wear prompt
        if 'What do you want to wear' in msg or 'What do you want to put on' in msg:
            arm = self._find_best_wear_letter()
            if arm:
                return self._val2idx.get(ord(arm), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Take off prompt
        if 'What do you want to take off' in msg or 'What do you want to remove' in msg:
            return self._val2idx.get(27, 0)  # ESC, we handle removal explicitly

        # Dip prompt (for Excalibur)
        if 'What do you want to dip' in msg:
            sword = self._find_long_sword_letter()
            if sword:
                return self._val2idx.get(ord(sword), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Throw prompt
        if 'What do you want to throw' in msg:
            proj = self._find_projectile_letter()
            if proj:
                return self._val2idx.get(ord(proj), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Zap prompt
        if 'What do you want to zap' in msg:
            wand = self._find_best_wand_letter()
            if wand:
                return self._val2idx.get(ord(wand), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Quaff prompt
        if 'What do you want to drink' in msg or 'What do you want to quaff' in msg:
            pot = self._find_healing_potion_letter()
            if pot:
                return self._val2idx.get(ord(pot), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Drop prompt
        if 'What do you want to drop' in msg:
            drop = self._find_droppable_letter()
            if drop:
                return self._val2idx.get(ord(drop), self._val2idx.get(27, 0))
            return self._val2idx.get(27, 0)

        # Default: ESC
        return self._val2idx.get(27, 0)

    # ================================================================
    # State update
    # ================================================================

    def _update_game_state(self):
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
            # Record stair connections
            lvl = self.current_level()
            went_down = (cur_key[1] > old_key[1] if cur_key[0] == old_key[0]
                         else cur_key[0] != old_key[0])
            if went_down:
                lvl.stairs_up.add((self.blstats.y, self.blstats.x))
            else:
                lvl.stairs_down.add((self.blstats.y, self.blstats.x))
            # Record destination in source level
            if old_key in self.levels:
                src = self.levels[old_key]
                py, px = self.blstats.y, self.blstats.x
                # The stair we came from in the old level
                # is approximately where we were standing
                src.stair_dest[(py, px)] = cur_key

        # Track turns on level
        if self.blstats.time != self._last_turn:
            self.current_level().turns_spent += 1
            self._last_turn = self.blstats.time

        self.glyphs = self.obs['glyphs']
        self.chars = self.obs.get('chars')
        self._update_maps()
        self._parse_inventory()
        self._parse_messages()
        self._update_pet_position()
        self._update_milestone()

    def _parse_blstats(self):
        bl = self._raw_bl
        if bl is not None and len(bl) >= 27:
            self.blstats = BLStats(*[int(v) for v in bl[:27]])
            self.alignment = self.blstats.align
        elif bl is not None and len(bl) >= 26:
            # Fallback: no alignment field, pad with 0
            vals = [int(v) for v in bl[:26]]
            vals.append(0)  # align
            self.blstats = BLStats(*vals)

    def _update_maps(self):
        """Update per-level seen/walkable/objects maps from glyphs."""
        g = self.glyphs
        py, px = self.blstats.y, self.blstats.x
        lvl = self.current_level()

        for r in range(MAP_H):
            for c in range(MAP_W):
                v = int(g[r, c])
                cm = _cmap(v)

                if cm in _WALKABLE:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = True
                    lvl.objects[r, c] = v
                    if cm in _STAIRS_DOWN:
                        lvl.stairs_down.add((r, c))
                    if cm in _STAIRS_UP:
                        lvl.stairs_up.add((r, c))
                    if cm in _FOUNTAIN:
                        lvl.fountains.add((r, c))
                    if cm in _ALTAR:
                        lvl.altars.add((r, c))

                elif cm in _WALL:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False
                    lvl.objects[r, c] = v

                elif cm in _CLOSED_DOOR:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False
                    lvl.objects[r, c] = v

                elif v == BOULDER_GLYPH:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False

                elif _is_any_monster(v) or (GLYPH_OBJ_OFF <= v < GLYPH_CMAP_OFF):
                    # Monster or object on tile: mark seen, keep previous walkable
                    lvl.seen[r, c] = True
                    if lvl.objects[r, c] == -1:
                        lvl.walkable[r, c] = True

                elif cm == SS_STONE:
                    # Only mark adjacent-to-player stone as confirmed
                    if abs(r - py) <= 1 and abs(c - px) <= 1:
                        lvl.seen[r, c] = True
                        lvl.walkable[r, c] = False

                elif cm in _POOL_LAVA:
                    lvl.seen[r, c] = True
                    lvl.walkable[r, c] = False
                    lvl.objects[r, c] = v

        lvl.walkable[py, px] = True
        lvl.seen[py, px] = True

        # Detect stairs from chars (more reliable backup)
        if self.chars is not None:
            ys, xs = (self.chars == ord('>')).nonzero()
            for y, x in zip(ys, xs):
                lvl.stairs_down.add((int(y), int(x)))
            ys, xs = (self.chars == ord('<')).nonzero()
            for y, x in zip(ys, xs):
                lvl.stairs_up.add((int(y), int(x)))

    def _parse_inventory(self):
        inv_strs = self.obs.get('inv_strs')
        inv_letters = self.obs.get('inv_letters')
        inv_oclasses = self.obs.get('inv_oclasses')
        inv_glyphs_arr = self.obs.get('inv_glyphs')
        if inv_strs is None or inv_letters is None:
            return
        self.inventory = {}
        self.inv_oclasses = {}
        self.inv_glyphs = {}
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
                if inv_glyphs_arr is not None:
                    self.inv_glyphs[ch] = int(inv_glyphs_arr[i])

        # Update weapon/armor tracking
        self._wielded_letter = None
        self._worn_slots = {}
        self._projectile_letters = {}
        self._wand_letters = {}
        for letter, item in self.inventory.items():
            lower = item.lower()
            oc = self.inv_oclasses.get(letter, -1)
            if '(weapon in hand)' in lower or '(wielded)' in lower:
                self._wielded_letter = letter
            if '(being worn)' in lower:
                for slot, keywords in SLOT_KEYWORDS.items():
                    if any(kw in lower for kw in keywords):
                        self._worn_slots[slot] = letter
                        break
            if 'excalibur' in lower:
                self.has_excalibur = True
            # Track projectiles
            if oc == WEAPON_CLASS and 'cursed' not in lower:
                for proj_name in ['dagger', 'dart', 'shuriken', 'spear', 'javelin', 'knife']:
                    if proj_name in lower and '(weapon in hand)' not in lower and '(wielded)' not in lower:
                        self._projectile_letters[letter] = proj_name
                        break
            # Track wands
            if oc == WAND_CLASS:
                self._wand_letters[letter] = item
            # Track BUC from inventory strings
            if 'blessed' in lower:
                self._buc_status[letter] = 'blessed'
            elif 'uncursed' in lower:
                self._buc_status[letter] = 'uncursed'
            elif 'cursed' in lower:
                self._buc_status[letter] = 'cursed'
                self._cursed_slots.add(letter)

    def _parse_messages(self):
        msg = self.message.lower()

        # Resistance tracking
        resists = {
            "you feel especially healthy": "poison resistance",
            "you feel a momentary chill": "cold resistance",
            "you feel warm": "fire resistance",
            "you feel full of energy": "shock resistance",
            "you feel very firm": "disintegration resistance",
            "you feel wide awake": "sleep resistance",
        }
        for frag, r in resists.items():
            if frag in msg:
                self.resistances.add(r)

        # Excalibur detection
        if "your sword has a bright" in msg or "excalibur" in msg:
            self.has_excalibur = True

        # Kill tracking
        self._last_kill_name = None
        self._last_kill_dir = (0, 0)
        for prefix in ["you kill the ", "you kill ", "you destroy the ", "you destroy "]:
            if prefix in msg:
                name = msg.split(prefix, 1)[1].split("!")[0].split(".")[0].strip()
                py, px = self.blstats.y, self.blstats.x
                self.food.on_kill(name, py, px, self.blstats.time, self.resistances)
                self._last_kill_name = name
                self._last_kill_dir = self._last_move_dir
                break

        # Door handling
        if 'the door opens' in msg:
            pass  # walkability updated by map scan
        if 'this door is locked' in msg:
            py, px = self.blstats.y, self.blstats.x
            dy, dx = self._last_move_dir
            ty, tx = py + dy, px + dx
            if 0 <= ty < MAP_H and 0 <= tx < MAP_W:
                self.lvl.door_attempts[ty, tx] += 1

        # Shop detection (for price-based ID later)
        if 'welcome to' in msg and 'shop' in msg:
            pass  # track shop location

        # Elbereth feedback
        if 'you see a message' in msg and 'elbereth' in msg:
            self._elbereth_turns = 0

        # Fountain dried up
        if 'the fountain dries up' in msg or 'water doesn\'t come' in msg:
            py, px = self.blstats.y, self.blstats.x
            self.lvl.fountains.discard((py, px))

        # Pet death
        if 'is killed' in msg and ('your ' in msg or 'the ' in msg):
            # Rough detection; pet glyph check is more reliable
            pass

        # Alignment tracking from messages
        if 'you feel that' in msg and 'is pleased' in msg:
            pass  # alignment went up
        if 'you have sinned' in msg:
            pass  # alignment went down

    def _update_pet_position(self):
        """Scan for pet glyphs."""
        if self.glyphs is None:
            return
        pet_found = False
        py, px = self.blstats.y, self.blstats.x
        best_dist = 999
        best_pos = None
        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(self.glyphs[r, c])
                if _is_pet(g):
                    pet_found = True
                    d = max(abs(r - py), abs(c - px))
                    if d < best_dist:
                        best_dist = d
                        best_pos = (r, c)
        if pet_found:
            self._pet_pos = best_pos
            self._pet_alive = True
        else:
            self._pet_alive = False

    def _update_milestone(self):
        """Update strategic milestone based on game state."""
        bl = self.blstats
        if bl is None:
            return

        if self.milestone == MILESTONE_FARM_DL1:
            # Stay on DL1 until XL >= 5 (farm for combat readiness)
            if bl.xl >= 5:
                if self.alignment == 1 and not self.has_excalibur:
                    self.milestone = MILESTONE_FIND_EXCALIBUR
                else:
                    self.milestone = MILESTONE_PUSH_DEEP

        elif self.milestone == MILESTONE_FIND_EXCALIBUR:
            if self.has_excalibur:
                self.milestone = MILESTONE_PUSH_DEEP
            # Give up on Excalibur after DL8 or XL12
            if bl.depth >= 8 or bl.xl >= 12:
                self.milestone = MILESTONE_PUSH_DEEP

        elif self.milestone == MILESTONE_SOKOBAN:
            if bl.dungeon_number != DUNGEON_SOKOBAN:
                self.milestone = MILESTONE_PUSH_DEEP

    # ================================================================
    # Navigation
    # ================================================================

    def _is_peaceful(self, g, y, x):
        """Check if a monster glyph is peaceful."""
        if not _is_monster(g):
            return False
        mid = _mon_id(g)
        if mid in PEACEFUL_IDS:
            return True
        if mid in self._peaceful_monster_ids:
            return True
        if (y, x) in self._peaceful_positions:
            return True
        if 0 <= mid < NUMMONS:
            name = nh.permonst(mid).mname
            if name in PEACEFUL_NAMES:
                return True
        return False

    def bfs(self, allow_hostiles=False, allow_pets=True):
        """BFS from player position. Returns distance array.

        allow_hostiles: if True, treat hostile monsters as walkable (can fight through)
        allow_pets: if True, treat pets as walkable (can swap)
        """
        py, px = self.blstats.y, self.blstats.x
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
                    if not (0 <= ny < MAP_H and 0 <= nx < MAP_W) or dis[ny, nx] != -1:
                        continue
                    g = int(self.glyphs[ny, nx])
                    cm = _cmap(g)

                    # Closed doors: walkable if not exhausted
                    is_closed = cm in _CLOSED_DOOR
                    if is_closed and lvl.door_attempts[ny, nx] >= 10:
                        continue

                    # Pet handling
                    is_pet = _is_pet(g)
                    if is_pet and not allow_pets:
                        continue

                    # Hostile monster handling
                    is_hostile = _is_monster(g) and not self._is_peaceful(g, ny, nx)
                    if is_hostile and not allow_hostiles:
                        continue

                    # Peaceful: never path through
                    if _is_monster(g) and self._is_peaceful(g, ny, nx):
                        continue

                    # Boulder: block
                    if g == BOULDER_GLYPH:
                        continue

                    # Pool/lava: block
                    if cm in _POOL_LAVA:
                        continue

                    ok = lvl.walkable[ny, nx] or is_closed or is_pet or (is_hostile and allow_hostiles)
                    if not ok:
                        continue

                    # No diagonal through doors
                    if abs(dy) + abs(dx) > 1:
                        src_cm = _cmap(int(self.glyphs[y, x]))
                        src_obj = _cmap(int(lvl.objects[y, x])) if lvl.objects[y, x] != -1 else -1
                        dst_obj = _cmap(int(lvl.objects[ny, nx])) if lvl.objects[ny, nx] != -1 else -1
                        if (src_cm in _DOOR or cm in _DOOR or
                            src_obj in _DOOR or dst_obj in _DOOR):
                            continue

                    dis[ny, nx] = d + 1
                    q.append((ny, nx))
        return dis

    def step_toward(self, ty, tx, dis):
        """Take one BFS-optimal step toward (ty, tx). Returns True if stepped."""
        if dis[ty, tx] == -1:
            return False
        py, px = self.blstats.y, self.blstats.x
        lvl = self.current_level()

        # Trace back from target to player
        path = []
        cy, cx = ty, tx
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

        # Validate: don't walk into walls/boulders
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            if g == BOULDER_GLYPH:
                return False
            cm = _cmap(g)
            if cm in _WALL or cm == SS_STONE:
                return False

        # Diagonal through door: use cardinal instead
        if abs(dy) + abs(dx) > 1:
            src_cm = _cmap(int(self.glyphs[py, px]))
            dst_cm = _cmap(int(self.glyphs[ny, nx]))
            src_obj = _cmap(int(lvl.objects[py, px])) if lvl.objects[py, px] != -1 else -1
            dst_obj = _cmap(int(lvl.objects[ny, nx])) if lvl.objects[ny, nx] != -1 else -1
            if src_cm in _DOOR or dst_cm in _DOOR or src_obj in _DOOR or dst_obj in _DOOR:
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W:
                        if lvl.walkable[cr, cc] or _cmap(int(self.glyphs[cr, cc])) in _CLOSED_DOOR:
                            dy, dx = cdy, cdx
                            ny, nx = cr, cc
                            break

        old_pos = (self.blstats.y, self.blstats.x)
        self._move_dir(dy, dx)

        # Retry if diagonal failed
        if 'diagonally' in self.message.lower() and abs(dy) + abs(dx) > 1:
            for cdy, cdx in [(dy, 0), (0, dx)]:
                if cdy == 0 and cdx == 0:
                    continue
                self._move_dir(cdy, cdx)
                break

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
        """Visible non-pet monsters sorted by distance."""
        py, px = self.blstats.y, self.blstats.x
        mons = []
        for r in range(MAP_H):
            for c in range(MAP_W):
                if r == py and c == px:
                    continue
                g = int(self.glyphs[r, c])
                if _is_monster(g):
                    mid = _mon_id(g)
                    name = _mon_name(mid)
                    d = max(abs(r - py), abs(c - px))
                    mons.append((d, r, c, name, mid))
        return sorted(mons)

    def get_hostile_monsters(self):
        """Visible hostile monsters sorted by distance."""
        return [(d, r, c, n, m) for d, r, c, n, m in self.get_monsters()
                if n not in PEACEFUL_NAMES
                and m not in PEACEFUL_IDS
                and m not in self._peaceful_monster_ids
                and (r, c) not in self._peaceful_positions]

    # ================================================================
    # Strategy priorities
    # ================================================================

    def emergency(self):
        """P0: Handle HP critical, starvation, stoning, illness."""
        bl = self.blstats
        if bl is None:
            return False

        # Condition bitmask from blstats
        cond = int(bl.condition)

        # Stoning: eat lizard corpse or pray
        if cond & COND_STONE:
            for letter, item in self.inventory.items():
                if 'lizard corpse' in item.lower():
                    self._two_step_eat(letter)
                    return True
            if self._can_pray():
                self._do_pray()
                return True

        # Sliming: pray
        if cond & COND_SLIME:
            if self._can_pray():
                self._do_pray()
                return True

        # Food poisoning / terminal illness: pray
        if cond & (COND_FOODPOIS | COND_TERMILL):
            if self._can_pray():
                self._do_pray()
                return True

        # Strangulation: pray
        if cond & COND_STRNGL:
            if self._can_pray():
                self._do_pray()
                return True

        # HP critical: pray at HP < max/3 or HP < 8
        can_pray = self._can_pray()
        if can_pray and bl.hp < max(8, bl.max_hp // 3):
            self._do_pray()
            return True

        # Weak/fainting from hunger: pray
        if can_pray and bl.hunger >= WEAK:
            self._do_pray()
            return True

        # Last resort: quaff a healing potion when HP critical and can't pray
        if not can_pray and bl.hp < max(6, bl.max_hp // 4):
            pot = self._find_healing_potion_letter()
            if pot:
                self._two_step_quaff(pot)
                return True

        return False

    def pray_strategy(self):
        """P1: Proactive prayer for minor troubles when safe."""
        bl = self.blstats
        if bl is None or not self._can_pray():
            return False

        # Only pray proactively if timeout fully expired (safe for minor trouble)
        if bl.time - self._last_prayer_turn < 400:
            return False

        # Hunger: pray if weak+ and no food
        if bl.hunger >= WEAK:
            has_food = any(self.inv_oclasses.get(l, -1) == FOOD_CLASS
                          for l in self.inventory)
            if not has_food:
                self._do_pray()
                return True

        return False

    def fight(self):
        """P2: Priority-based combat system (AutoAscend fight2 port).

        Loops until all nearby hostile monsters are dead or fled.
        Re-evaluates each round. Uses continuous fight loop.
        """
        acted = False

        for _round in range(80):
            mons = self.get_hostile_monsters()
            if not mons:
                break

            py, px = self.blstats.y, self.blstats.x
            hp_ratio = self.blstats.hp / max(1, self.blstats.max_hp)

            # Adjacent hostiles
            adj = [(d, r, c, n, m) for d, r, c, n, m in mons if d <= 1]

            # No adjacent and no close hostiles: stop fighting
            close = [(d, r, c, n, m) for d, r, c, n, m in mons if d <= 7]
            if not adj and not close:
                break

            # Check emergency between rounds
            if self.blstats.hp < max(8, self.blstats.max_hp // 3):
                if self._can_pray():
                    self._do_pray()
                    acted = True
                    continue

            # Build action candidates
            actions = []

            # === INSTAKILL FLEE ===
            for d, r, c, n, m in adj:
                if n in INSTAKILL:
                    dy, dx = py - r, px - c  # direction away
                    actions.append((100, ('flee', dy, dx, n)))

            # === MELEE ACTIONS ===
            for d, r, c, n, m in adj:
                if n in NEVER_MELEE or n in INSTAKILL:
                    continue
                dy, dx = r - py, c - px
                pri = 1
                if self.blstats.hp > 8:
                    pri += 15
                # Danger-based priority
                mlevel = _MONSTER_MLEVEL.get(n, 0)
                if n in ONLY_RANGED_SLOW:
                    pri -= 100  # don't melee these
                if n in EXPLODING:
                    pri -= 17
                if n in WEAK:
                    pri += 5  # easy kill, do it
                # Higher level monsters first (prevent them from hitting more)
                pri += min(mlevel, 10)
                actions.append((pri, ('melee', dy, dx, n)))

            # === ELBERETH ===
            if adj and self.blstats.hp < 30 and not self._can_pray():
                adj_threat = 0.0
                for d, r, c, n, m in adj:
                    if n in ONLY_RANGED_SLOW:
                        continue
                    adj_threat += 1.0
                    mlevel = _MONSTER_MLEVEL.get(n, 0)
                    if mlevel > self.blstats.xl + 3:
                        adj_threat += 2.0
                    if n in FAST_MONSTERS:
                        adj_threat += 0.5
                if adj_threat > 0:
                    # Check Elbereth immunity
                    all_immune = all(n in _ELBERETH_IMMUNE_NAMES for _, _, _, n, _ in adj)
                    if not all_immune:
                        elb_pri = -15 + 20 * adj_threat * (1 - hp_ratio**0.5)
                        actions.append((elb_pri, ('elbereth',)))

            # === WAIT ON ELBERETH ===
            if self._elbereth_turns < 3:
                # Recently wrote Elbereth, wait for HP recovery
                msg_low = self.initial_message.lower() if self.initial_message else ''
                if 'elbereth' in msg_low or self._elbereth_turns == 0:
                    wait_pri = 30 - hp_ratio * 40
                    actions.append((wait_pri, ('wait',)))
                    # Suppress melee while on Elbereth
                    actions = [(p - 100 if a[0] == 'melee' else p, a) for p, a in actions]

            # === TACTICAL FLEE ===
            melee_adj = [x for x in adj if x[3] not in NEVER_MELEE and x[3] not in INSTAKILL]
            if len(melee_adj) >= 2 and hp_ratio < 0.4:
                best_flee = self._find_flee_direction(py, px, melee_adj)
                if best_flee:
                    flee_pri = 5 + (1 - hp_ratio) * 15
                    actions.append((flee_pri, ('flee', best_flee[0], best_flee[1], 'tactical')))

            # === ZAP WAND at distant monster ===
            if not adj and self._wand_letters:
                wand_letter = next(iter(self._wand_letters))
                for d, r, c, n, m in mons:
                    if d > 7 or d <= 1:
                        continue
                    dy, dx = r - py, c - px
                    if dy != 0 and dx != 0 and abs(dy) != abs(dx):
                        continue
                    ndy = (1 if dy > 0 else -1) if dy != 0 else 0
                    ndx = (1 if dx > 0 else -1) if dx != 0 else 0
                    actions.append((3, ('zap', wand_letter, ndy, ndx, n)))
                    break

            # === THROW at distant monster ===
            if not adj and self._projectile_letters:
                proj_letter = next(iter(self._projectile_letters))
                for d, r, c, n, m in mons:
                    if d > 5 or d <= 1:
                        continue
                    dy, dx = r - py, c - px
                    # Must be in a line (cardinal or diagonal)
                    if dy != 0 and dx != 0 and abs(dy) != abs(dx):
                        continue
                    ndy = (1 if dy > 0 else -1) if dy != 0 else 0
                    ndx = (1 if dx > 0 else -1) if dx != 0 else 0
                    actions.append((2, ('throw', proj_letter, ndy, ndx, n)))
                    break

            # === APPROACH distant monsters ===
            if not adj and self.blstats.hp > self.blstats.max_hp * 0.3:
                fight_dis = self.bfs(allow_hostiles=True)
                approach_range = 12
                if not self.lvl.stairs_down and self.lvl.turns_spent > 500:
                    approach_range = 6
                best_mon = None
                best_d = 999
                for d, r, c, n, m in mons:
                    fd = fight_dis[r, c]
                    if fd != -1 and fd <= approach_range and fd < best_d:
                        best_d = fd
                        best_mon = (r, c, n)
                if best_mon:
                    actions.append((0, ('approach', best_mon[0], best_mon[1], best_mon[2])))

            if not actions:
                break

            # Execute highest priority action
            actions.sort(key=lambda x: -x[0])
            _, best = actions[0]
            acted = True

            if best[0] == 'melee':
                self._move_dir(best[1], best[2])
            elif best[0] == 'flee':
                dy, dx = best[1], best[2]
                # Normalize flee direction
                if dy != 0:
                    dy = 1 if dy > 0 else -1
                if dx != 0:
                    dx = 1 if dx > 0 else -1
                self._move_dir(dy, dx)
            elif best[0] == 'elbereth':
                self._engrave_elbereth()
                self._elbereth_turns = 0
                # Rest on Elbereth for a few turns
                for _ in range(min(3, max(0, 3 - int(hp_ratio * 5)))):
                    if self.blstats.hp >= self.blstats.max_hp * 0.5:
                        break
                    self.step(A.Command.SEARCH)
            elif best[0] == 'wait':
                self.step(A.Command.SEARCH)
                self._elbereth_turns += 1
            elif best[0] == 'zap':
                self._three_step_zap(best[1], best[2], best[3])
            elif best[0] == 'throw':
                self._three_step_throw(best[1], best[2], best[3])
            elif best[0] == 'approach':
                fight_dis = self.bfs(allow_hostiles=True)
                self.step_toward(best[1], best[2], fight_dis)

        return acted

    def _find_flee_direction(self, py, px, melee_adj):
        """Find best direction to flee from multiple melee threats."""
        best_dir = None
        best_threat = len(melee_adj)
        lvl = self.current_level()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            if not lvl.walkable[ny, nx]:
                continue
            g = int(self.glyphs[ny, nx])
            if _is_monster(g) or g == BOULDER_GLYPH:
                continue
            threat = sum(1 for _, r, c, _, _ in melee_adj
                        if max(abs(r - ny), abs(c - nx)) <= 1)
            if threat < best_threat:
                best_threat = threat
                best_dir = (dy, dx)
        return best_dir

    def eat_corpse_after_kill(self):
        """P3a: Step onto a fresh kill's corpse and eat it."""
        if self._last_kill_name is None:
            return False
        bl = self.blstats
        if bl is None:
            return False

        name = self._last_kill_name
        dy, dx = self._last_kill_dir
        self._last_kill_name = None

        # Only eat when hungry or for intrinsics
        should_eat = bl.hunger >= HUNGRY
        # Also eat for valuable intrinsics even when not hungry
        if not should_eat and name in INTRINSIC_CORPSES:
            for resist_type, monsters in INTRINSIC_CORPSES.items():
                if name in monsters and f"{resist_type} resistance" not in self.resistances:
                    should_eat = True
                    break
        if name == 'wraith':  # always eat wraith (gain level)
            should_eat = True
        if name == 'floating eye' and 'telepathy' not in self.resistances:
            # Only if blind (free action prevents paralysis) or have free action
            should_eat = False  # too dangerous without free action

        if not should_eat:
            return False
        if not self.food.is_corpse_safe(name, self.resistances):
            return False
        if name in _EXTRA_UNSAFE:
            return False
        if dy == 0 and dx == 0:
            return False

        py, px = bl.y, bl.x
        cy, cx = py + dy, px + dx
        if not (0 <= cy < MAP_H and 0 <= cx < MAP_W):
            return False

        self._move_dir(dy, dx)
        msg_low = self.initial_message.lower() if self.initial_message else ''
        if 'corpse' not in msg_low and 'you see here' not in msg_low:
            return True  # moved but no corpse visible

        self.step(A.Command.EAT)
        self._last_eat_turn = bl.time
        return True

    def eat_ground(self):
        """P3b: Eat corpse/food from ground if standing on one."""
        bl = self.blstats
        if bl is None or bl.hunger < HUNGRY:
            return False
        if bl.time - self._last_eat_turn < 3:
            return False

        msg = self.initial_message.lower() if self.initial_message else ''
        if 'you see here' not in msg:
            return False

        if 'corpse' in msg:
            parts = msg.split('you see here ')
            if len(parts) > 1:
                corpse_desc = parts[1].split(' corpse')[0]
                for article in ['a ', 'an ', 'the ']:
                    if corpse_desc.startswith(article):
                        corpse_desc = corpse_desc[len(article):]
                        break
                if not self.food.is_corpse_safe(corpse_desc, self.resistances):
                    return False
                if corpse_desc in _EXTRA_UNSAFE:
                    return False
                if corpse_desc == 'floating eye':
                    return False
        elif not any(kw in msg for kw in ['food', 'ration', 'lembas', 'wafer', 'cram']):
            return False

        self._last_eat_turn = bl.time
        self.step(A.Command.EAT)
        return True

    def eat(self):
        """P3c: Eat from inventory when hungry."""
        bl = self.blstats
        if bl is None or bl.hunger < HUNGRY:
            return False
        if bl.time - self._last_eat_turn < 5:
            return False

        food_letter = self._find_best_food_letter()
        if food_letter is None:
            return False

        self._last_eat_turn = bl.time
        self._two_step_eat(food_letter)
        return True

    def pickup_useful(self):
        """P4: Pick up useful items from ground."""
        msg = self.initial_message.lower() if self.initial_message else ''
        if 'you see here' not in msg:
            return False
        if 'cursed' in msg:
            return False

        # Carrying capacity check
        if self.blstats.carrying_capacity >= 2:  # stressed or worse
            return False

        food_kw = ['food ration', 'cram ration', 'lembas wafer', 'k-ration', 'c-ration',
                    'tripe ration', 'tin', 'lizard corpse', 'lichen corpse']
        armor_kw = ['mail', 'armor', 'helm', 'cloak', 'gloves', 'gauntlets',
                     'boots', 'shoes', 'jacket', 'shield']
        weapon_kw = ['long sword', 'katana', 'silver saber', 'broadsword', 'scimitar',
                      'battle-axe', 'morning star', 'war hammer', 'dagger', 'dart']
        potion_kw = ['potion']
        wand_kw = ['wand']
        scroll_kw = ['scroll']
        gold_kw = ['gold piece']
        tool_kw = ['key', 'lamp', 'lantern', 'pick-axe', 'unicorn horn']

        is_useful = any(
            any(w in msg for w in kw_list)
            for kw_list in [food_kw, armor_kw, weapon_kw, potion_kw, wand_kw,
                           scroll_kw, gold_kw, tool_kw]
        )
        # Skip junk
        junk = ['rock', 'statue', 'boulder', 'chain', 'iron ball', 'loadstone',
                'worthless', 'gray stone']
        if any(j in msg for j in junk):
            is_useful = False

        if not is_useful:
            return False

        pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
        if pickup_idx is None:
            return False
        if self._env_step(pickup_idx):
            self._parse_blstats()
            raise AgentFinished()
        self._update_game_state()
        return True

    def auto_equip(self):
        """P5: Equip best available weapons and armor."""
        self._parse_inventory()

        # Wield better weapon
        weapon_letter = self.equip.find_best_weapon(self.inventory)
        if weapon_letter:
            self._two_step_wield(weapon_letter)
            self._parse_inventory()

        # Wear armor for empty slots
        armor_letter = self.equip.find_best_armor(self.inventory)
        if armor_letter:
            buc = self._buc_status.get(armor_letter)
            if buc == 'cursed':
                return False  # don't wear cursed
            self._two_step_wear(armor_letter)
            self._parse_inventory()

        return False

    def dip_excalibur(self):
        """P6: Dip long sword in fountain for Excalibur."""
        bl = self.blstats
        if bl is None or self.has_excalibur:
            return False
        if bl.xl < 5:
            return False
        if self.alignment != 1:  # must be lawful
            return False

        py, px = bl.y, bl.x
        on_fountain = (py, px) in self.lvl.fountains
        if not on_fountain:
            g_here = int(self.glyphs[py, px])
            obj_here = int(self.lvl.objects[py, px]) if self.lvl.objects[py, px] != -1 else -1
            on_fountain = _cmap(g_here) in _FOUNTAIN or _cmap(obj_here) in _FOUNTAIN

        if not on_fountain:
            return False

        sword_letter = self._find_long_sword_letter()
        if sword_letter is None:
            return False

        # DIP command then sword letter
        dip_idx = self._val2idx.get(int(A.Command.DIP))
        if dip_idx is None:
            return False
        if self._env_step(dip_idx):
            self._parse_blstats(); raise AgentFinished()
        sword_idx = self._val2idx.get(ord(sword_letter))
        if sword_idx is not None:
            if self._env_step(sword_idx):
                self._parse_blstats(); raise AgentFinished()
            # Handle "Dip into fountain?" confirmation
            for _ in range(10):
                misc = self.obs.get('misc', [0, 0, 0])
                if misc[0]:
                    if self._env_step(self._val2idx.get(ord('y'), 0)):
                        self._parse_blstats(); raise AgentFinished()
                    continue
                if misc[2] or '--More--' in bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace'):
                    if self._env_step(self._val2idx.get(32, 0)):
                        self._parse_blstats(); raise AgentFinished()
                    continue
                break
        self._update_game_state()

        # Check result
        self._parse_inventory()
        if any('Excalibur' in s for s in self.inventory.values()):
            self.has_excalibur = True

        # Remove fountain if dried up
        g = int(self.glyphs[py, px]) if self.glyphs is not None else 0
        if _cmap(g) not in _FOUNTAIN:
            self.lvl.fountains.discard((py, px))

        return True

    def seek_fountain(self):
        """P6b: Navigate to fountain for Excalibur if conditions met."""
        bl = self.blstats
        if bl is None or self.has_excalibur:
            return False
        if bl.xl < 5 or self.alignment != 1:
            return False
        if self._find_long_sword_letter() is None:
            return False

        py, px = bl.y, bl.x
        if self.lvl.fountains:
            dis = self.bfs(allow_hostiles=True)
            best_f, best_d = None, 999
            for fy, fx in self.lvl.fountains:
                d = dis[fy, fx]
                if d != -1 and d < best_d:
                    best_d = d
                    best_f = (fy, fx)
            if best_f and best_d > 0:
                return self.step_toward(best_f[0], best_f[1], dis)
        return False

    def sokoban(self):
        """P7: Solve Sokoban puzzles."""
        bl = self.blstats
        if bl is None or bl.dungeon_number != DUNGEON_SOKOBAN:
            return False
        return self._solve_sokoban()

    def explore(self):
        """P8: Explore the dungeon. Open doors, find stairs, search."""
        py, px = self.blstats.y, self.blstats.x
        lvl = self.current_level()

        # In Sokoban: go up after solving
        if self.blstats.dungeon_number == DUNGEON_SOKOBAN:
            if lvl.soko_solution is not None and lvl.soko_step >= len(lvl.soko_solution):
                if lvl.stairs_up:
                    dis = self.bfs(allow_hostiles=True)
                    for uy, ux in lvl.stairs_up:
                        if (py, px) == (uy, ux):
                            self.step(A.MiscDirection.UP)
                            return
                        if dis[uy, ux] != -1:
                            self.step_toward(uy, ux, dis)
                            return

        # On downstairs: check descent conditions
        if self._on_stairs_down():
            if self._should_descend():
                # Wait for pet at stairs (AutoAscend pet management)
                if self._pet_alive and self._pet_pos is not None:
                    pet_dist = max(abs(self._pet_pos[0] - py), abs(self._pet_pos[1] - px))
                    if pet_dist > 2 and lvl.turns_spent < 50:
                        self.step(A.Command.SEARCH)  # wait for pet
                        return
                self.step(A.MiscDirection.DOWN)
                return

        # Excalibur: dip if on fountain
        if not self.has_excalibur and self.blstats.xl >= 5 and self.alignment == 1:
            if (py, px) in lvl.fountains:
                self.dip_excalibur()
                return

        dis = self.bfs()

        # 1. Adjacent closed doors: open or kick
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = py + dy, px + dx
            if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                g = int(self.glyphs[nr, nc])
                if _cmap(g) in _CLOSED_DOOR and lvl.door_attempts[nr, nc] < 10:
                    lvl.door_attempts[nr, nc] += 1
                    if lvl.door_attempts[nr, nc] <= 2:
                        self._move_dir(dy, dx)
                        msg_low = self.message.lower()
                        if 'locked' not in msg_low and _cmap(int(self.glyphs[nr, nc])) not in _CLOSED_DOOR:
                            return
                    self._kick_dir(dy, dx)
                    return

        # 2. Navigate to nearest closed door
        best_door, best_dd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if _cmap(int(self.glyphs[r, c])) in _CLOSED_DOOR and lvl.door_attempts[r, c] < 5:
                    d = dis[r, c]
                    if d != -1 and d < best_dd:
                        best_dd = d
                        best_door = (r, c)
        if best_door and best_dd > 1:
            if self.step_toward(best_door[0], best_door[1], dis):
                return

        # 3. Force descent check
        descent_timer = max(40, 200 - self.blstats.depth * 30)
        xl_ready = self._xl_ready()
        force_descend = lvl.turns_spent > descent_timer and xl_ready

        if force_descend and lvl.stairs_down and self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self.bfs(allow_hostiles=True)
            best_s, best_sd = None, 999
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

        # 4. Frontier exploration (AutoAscend-style priority)
        best_f, best_fp = None, float('-inf')
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or not lvl.walkable[r, c]:
                    continue

                # Check for adjacent unseen tiles
                has_unseen = False
                adj_stone = 0
                adj_wall = 0
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            if not lvl.seen[nr, nc]:
                                has_unseen = True
                            obj_val = int(lvl.objects[nr, nc]) if lvl.objects[nr, nc] != -1 else int(self.glyphs[nr, nc])
                            cm2 = _cmap(obj_val)
                            if cm2 == SS_STONE:
                                adj_stone += 1
                            elif cm2 in _WALL:
                                adj_wall += 1

                if not has_unseen:
                    continue

                # Priority: door+stone bonus, dead-end bonus, distance penalty
                p = 0.0
                obj_here = int(lvl.objects[r, c]) if lvl.objects[r, c] != -1 else -1
                if _cmap(obj_here) in _DOOR and adj_stone >= 3:
                    p += 250

                # Dead end: <= 1 walkable cardinal neighbor
                cardinal_w = sum(1 for dy3, dx3 in [(-1,0),(1,0),(0,-1),(0,1)]
                                if 0 <= r+dy3 < MAP_H and 0 <= c+dx3 < MAP_W
                                and lvl.walkable[r+dy3, c+dx3])
                if cardinal_w <= 1:
                    p += 250

                p += adj_stone * 10 + adj_wall * 5
                p -= d * 2  # distance penalty
                sc = lvl.search_count[r, c]
                p -= sc * sc * 2  # quadratic re-search penalty

                if p > best_fp:
                    best_fp = p
                    best_f = (r, c)

        if best_f:
            if self.step_toward(best_f[0], best_f[1], dis):
                return

        # 5. No frontier: navigate to stairs
        if lvl.stairs_down and self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self.bfs(allow_hostiles=True)
            best_s, best_sd = None, 999
            for sy, sx in lvl.stairs_down:
                d = fight_dis[sy, sx]
                if d != -1 and d < best_sd:
                    best_sd = d
                    best_s = (sy, sx)
            if best_s:
                if best_sd == 0:
                    if self._should_descend():
                        self.step(A.MiscDirection.DOWN)
                        return
                if self.step_toward(best_s[0], best_s[1], fight_dis):
                    return

        # 6. Search (AutoAscend to_search_func)
        best_s, best_sp = None, float('-inf')
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
                            obj_val = int(lvl.objects[nr, nc]) if lvl.objects[nr, nc] != -1 else int(self.glyphs[nr, nc])
                            cm2 = _cmap(obj_val)
                            if cm2 == SS_STONE:
                                stones += 1
                            elif cm2 in _WALL:
                                walls += 1

                if stones == 0 and walls == 0:
                    continue

                sc = lvl.search_count[r, c]
                p = -1.0 - sc * sc * 2
                obj_here = int(lvl.objects[r, c]) if lvl.objects[r, c] != -1 else -1
                if _cmap(obj_here) in _DOOR and stones >= 3:
                    p += 250
                cardinal_w = sum(1 for dy3, dx3 in [(-1,0),(1,0),(0,-1),(0,1)]
                                if 0 <= r+dy3 < MAP_H and 0 <= c+dx3 < MAP_W
                                and lvl.walkable[r+dy3, c+dx3])
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
        search_rounds = min(5, max(1, 12 - lvl.search_count[py, px]))
        for _ in range(search_rounds):
            lvl.search_count[py, px] += 1
            self.step(A.Command.SEARCH)
            # Check if new passage appeared
            for dy2 in (-1, 0, 1):
                for dx2 in (-1, 0, 1):
                    if dy2 == 0 and dx2 == 0:
                        continue
                    nr, nc = py + dy2, px + dx2
                    if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                        g = int(self.glyphs[nr, nc])
                        cm = _cmap(g)
                        if cm in _CLOSED_DOOR or (cm in _WALKABLE and not lvl.seen[nr, nc]):
                            return

    def rest(self):
        """P9: Rest to recover HP when safe."""
        bl = self.blstats
        if bl is None:
            return False
        if bl.hp >= bl.max_hp * 0.6:
            return False
        # Don't rest with adjacent monsters
        mons = self.get_hostile_monsters()
        if any(d <= 2 for d, _, _, _, _ in mons):
            return False
        # Don't rest if hungry
        if bl.hunger >= HUNGRY:
            return False
        # Rest up to 10 turns
        rested = False
        for _ in range(10):
            if self.blstats.hp >= self.blstats.max_hp * 0.7:
                break
            mons = self.get_hostile_monsters()
            if any(d <= 2 for d, _, _, _, _ in mons):
                break
            self.step(A.Command.SEARCH)
            rested = True
        return rested

    # ================================================================
    # Descent strategy
    # ================================================================

    def _on_stairs_down(self):
        """Check if player is on downstairs."""
        py, px = self.blstats.y, self.blstats.x
        lvl = self.current_level()
        if (py, px) in lvl.stairs_down:
            return True
        g_here = int(self.glyphs[py, px]) if self.glyphs is not None else 0
        if _cmap(g_here) in _STAIRS_DOWN:
            return True
        obj_here = int(lvl.objects[py, px]) if lvl.objects[py, px] != -1 else -1
        if obj_here != -1 and _cmap(obj_here) in _STAIRS_DOWN:
            return True
        return False

    def _xl_ready(self):
        """Check XL gate for current milestone."""
        bl = self.blstats
        if self.milestone == MILESTONE_FARM_DL1:
            return bl.xl >= 5
        elif self.milestone in (MILESTONE_FIND_EXCALIBUR, MILESTONE_PUSH_DEEP):
            return bl.xl >= max(bl.depth, 2)
        return True

    def _should_descend(self):
        """Full descent decision."""
        bl = self.blstats
        if bl.hp < bl.max_hp * 0.5:
            return False

        lvl = self.current_level()
        descent_timer = max(40, 200 - bl.depth * 30)
        time_ok = lvl.turns_spent > descent_timer

        if self.milestone == MILESTONE_FARM_DL1:
            return bl.xl >= 5 and time_ok
        elif self.milestone == MILESTONE_FIND_EXCALIBUR:
            return bl.xl >= max(bl.depth, 2) and time_ok
        else:
            return bl.xl >= max(bl.depth - 2, 2) and time_ok

    # ================================================================
    # Sokoban
    # ================================================================

    def _solve_sokoban(self):
        """Execute Sokoban puzzle solution."""
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
            if int(self.glyphs[game_by, game_bx]) != BOULDER_GLYPH:
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

    # ================================================================
    # Multi-step action helpers
    # ================================================================

    def _two_step_eat(self, letter):
        """EAT command then food letter via raw _env_step.

        Uses raw steps to bypass the prompt handler, which would
        try to pick its own food letter. This gives us control
        over exactly which item to eat.
        """
        eat_idx = self._val2idx.get(int(A.Command.EAT))
        if eat_idx is None:
            return
        if self._env_step(eat_idx):
            self._parse_blstats(); raise AgentFinished()
        # Handle prompts until we get the getlin for food selection
        for _ in range(20):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace')
            if misc[0]:  # yn prompt (e.g., "eat it?")
                if self._env_step(self._val2idx.get(ord('y'), 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            if misc[1]:  # getlin prompt
                l_idx = self._val2idx.get(ord(letter))
                if l_idx is not None:
                    if self._env_step(l_idx):
                        self._parse_blstats(); raise AgentFinished()
                else:
                    if self._env_step(self._val2idx.get(27, 0)):
                        self._parse_blstats(); raise AgentFinished()
                break
            if misc[2] or '--More--' in msg:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            break
        self._update_game_state()

    def _two_step_wield(self, letter):
        """WIELD command then weapon letter via raw _env_step."""
        wield_idx = self._val2idx.get(int(A.Command.WIELD))
        if wield_idx is None:
            return
        if self._env_step(wield_idx):
            self._parse_blstats(); raise AgentFinished()
        for _ in range(10):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace')
            if misc[1]:
                l_idx = self._val2idx.get(ord(letter))
                if l_idx is not None:
                    if self._env_step(l_idx):
                        self._parse_blstats(); raise AgentFinished()
                else:
                    if self._env_step(self._val2idx.get(27, 0)):
                        self._parse_blstats(); raise AgentFinished()
                break
            if misc[0]:
                if self._env_step(self._val2idx.get(ord('y'), 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            if misc[2] or '--More--' in msg:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            break
        self._update_game_state()

    def _two_step_wear(self, letter):
        """WEAR command then armor letter via raw _env_step."""
        wear_idx = self._val2idx.get(int(A.Command.WEAR))
        if wear_idx is None:
            return
        if self._env_step(wear_idx):
            self._parse_blstats(); raise AgentFinished()
        for _ in range(10):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace')
            if misc[1]:
                l_idx = self._val2idx.get(ord(letter))
                if l_idx is not None:
                    if self._env_step(l_idx):
                        self._parse_blstats(); raise AgentFinished()
                else:
                    if self._env_step(self._val2idx.get(27, 0)):
                        self._parse_blstats(); raise AgentFinished()
                break
            if misc[0]:
                if self._env_step(self._val2idx.get(ord('y'), 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            if misc[2] or '--More--' in msg:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            break
        self._update_game_state()

    def _two_step_quaff(self, letter):
        """QUAFF command then potion letter via raw _env_step."""
        quaff_idx = self._val2idx.get(int(A.Command.QUAFF))
        if quaff_idx is None:
            return
        if self._env_step(quaff_idx):
            self._parse_blstats(); raise AgentFinished()
        for _ in range(10):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace')
            if misc[1]:
                l_idx = self._val2idx.get(ord(letter))
                if l_idx is not None:
                    if self._env_step(l_idx):
                        self._parse_blstats(); raise AgentFinished()
                else:
                    if self._env_step(self._val2idx.get(27, 0)):
                        self._parse_blstats(); raise AgentFinished()
                break
            if misc[0]:
                if self._env_step(self._val2idx.get(ord('y'), 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            if misc[2] or '--More--' in msg:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats(); raise AgentFinished()
                continue
            break
        self._update_game_state()

    def _three_step_zap(self, wand_letter, dy, dx):
        """ZAP command, wand letter, direction."""
        zap_idx = self._val2idx.get(int(A.Command.ZAP))
        if zap_idx is None:
            return
        if self._env_step(zap_idx):
            self._parse_blstats(); raise AgentFinished()
        w_idx = self._val2idx.get(ord(wand_letter))
        if w_idx is not None:
            if self._env_step(w_idx):
                self._parse_blstats(); raise AgentFinished()
            dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W',
                    (-1,1):'NE',(1,1):'SE',(1,-1):'SW',(-1,-1):'NW'}
            dname = dmap.get((dy, dx))
            if dname and dname in self._name2idx:
                if self._env_step(self._name2idx[dname]):
                    self._parse_blstats(); raise AgentFinished()
        self._update_game_state()

    def _three_step_throw(self, proj_letter, dy, dx):
        """THROW command, projectile letter, direction."""
        throw_idx = self._val2idx.get(int(A.Command.THROW))
        if throw_idx is None:
            return
        if self._env_step(throw_idx):
            self._parse_blstats(); raise AgentFinished()
        p_idx = self._val2idx.get(ord(proj_letter))
        if p_idx is not None:
            if self._env_step(p_idx):
                self._parse_blstats(); raise AgentFinished()
            dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W',
                    (-1,1):'NE',(1,1):'SE',(1,-1):'SW',(-1,-1):'NW'}
            dname = dmap.get((dy, dx))
            if dname and dname in self._name2idx:
                if self._env_step(self._name2idx[dname]):
                    self._parse_blstats(); raise AgentFinished()
        self._update_game_state()

    def _engrave_elbereth(self):
        """Engrave 'Elbereth' in the dust using fingers."""
        engrave_idx = self._val2idx.get(int(A.Command.ENGRAVE))
        if engrave_idx is None:
            return
        if self._env_step(engrave_idx):
            self._parse_blstats(); raise AgentFinished()

        # Handle "add to current engraving?" and "What do you want to write with?"
        for _ in range(5):
            misc = self.obs.get('misc', [0, 0, 0])
            msg = bytes(self.obs.get('message', b'')).decode('latin-1', errors='replace').lower()
            if misc[0]:
                # yn prompt
                if 'add to' in msg:
                    resp = self._val2idx.get(ord('n'))
                else:
                    resp = self._val2idx.get(ord('y'))
                if resp and self._env_step(resp):
                    self._parse_blstats(); raise AgentFinished()
                continue
            break

        # '-' for fingers
        dash_idx = self._val2idx.get(ord('-'))
        if dash_idx is not None:
            if self._env_step(dash_idx):
                self._parse_blstats(); raise AgentFinished()

        # Type E,l,b,e,r,e,t,h
        for ch in 'Elbereth':
            ch_idx = self._val2idx.get(ord(ch))
            if ch_idx is not None:
                if self._env_step(ch_idx):
                    self._parse_blstats(); raise AgentFinished()

        # Enter to finish
        cr_idx = self._val2idx.get(13)
        if cr_idx is not None:
            if self._env_step(cr_idx):
                self._parse_blstats(); raise AgentFinished()

        self._update_game_state()

    # ================================================================
    # Prayer helpers
    # ================================================================

    def _can_pray(self):
        """Check if prayer is safe."""
        bl = self.blstats
        if bl is None:
            return False
        # Prayer timeout: 300 turns early game, reduce with difficulty
        timeout = 300
        if bl.time < 300:
            timeout = 300  # initial timeout from game start
        turns_since = bl.time - self._last_prayer_turn
        return turns_since >= timeout

    def _do_pray(self):
        """Execute prayer."""
        self._last_prayer_turn = self.blstats.time
        self.step(A.Command.PRAY)

    # ================================================================
    # Inventory search helpers
    # ================================================================

    def _find_best_food_letter(self):
        """Find best food item to eat from inventory."""
        best_letter = None
        best_pri = -1
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc != FOOD_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue
            # Don't eat corpses from inventory (risky)
            if 'corpse' in lower:
                # Exception: lizard corpse (always safe, cures stoning)
                if 'lizard' in lower:
                    return letter
                continue

            # Priority: food rations > cram > lembas > tripe > tins > other
            pri = 0
            if 'food ration' in lower:
                pri = 10
            elif 'cram ration' in lower:
                pri = 9
            elif 'lembas wafer' in lower:
                pri = 8
            elif 'k-ration' in lower or 'c-ration' in lower:
                pri = 7
            elif 'tripe ration' in lower:
                pri = 3
            elif 'tin' in lower:
                pri = 2
            else:
                pri = 1

            if pri > best_pri:
                best_pri = pri
                best_letter = letter
        return best_letter

    def _find_best_wield_letter(self):
        """Find best weapon to wield."""
        return self.equip.find_best_weapon(self.inventory)

    def _find_best_wear_letter(self):
        """Find best armor to wear."""
        letter = self.equip.find_best_armor(self.inventory)
        if letter and self._buc_status.get(letter) == 'cursed':
            return None
        return letter

    def _find_long_sword_letter(self):
        """Find long sword in inventory for Excalibur."""
        for letter, item in self.inventory.items():
            if 'long sword' in item.lower() and 'cursed' not in item.lower():
                return letter
        return None

    def _find_projectile_letter(self):
        """Find a projectile to throw."""
        if self._projectile_letters:
            return next(iter(self._projectile_letters))
        return None

    def _find_best_wand_letter(self):
        """Find best wand to zap."""
        if self._wand_letters:
            # Prefer wands with known identity
            for letter, item in self._wand_letters.items():
                lower = item.lower()
                for good in ['death', 'fire', 'lightning', 'cold', 'magic missile', 'sleep', 'striking']:
                    if good in lower:
                        return letter
            return next(iter(self._wand_letters))
        return None

    def _find_healing_potion_letter(self):
        """Find a healing potion in inventory."""
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc != POTION_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue
            for h in ['healing', 'extra healing', 'full healing']:
                if h in lower:
                    return letter
        # Fallback: any non-cursed potion (might be healing)
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc != POTION_CLASS:
                continue
            lower = item.lower()
            if 'cursed' in lower:
                continue
            return letter
        return None

    def _find_droppable_letter(self):
        """Find least valuable item to drop when encumbered."""
        keep = {'(weapon in hand)', '(wielded)', '(being worn)',
                'food ration', 'lizard corpse'}
        # Drop corpses, rocks, gems first
        for letter, item in self.inventory.items():
            lower = item.lower()
            if any(k in lower for k in keep):
                continue
            if 'corpse' in lower or 'rock' in lower or 'stone' in lower:
                return letter
        # Drop any non-essential
        for letter, item in self.inventory.items():
            lower = item.lower()
            if any(k in lower for k in keep):
                continue
            return letter
        return None

    # ================================================================
    # Altar use
    # ================================================================

    def use_altar(self):
        """Drop items on altar for BUC identification."""
        bl = self.blstats
        if bl is None:
            return False
        py, px = bl.y, bl.x
        lvl = self.current_level()
        if (py, px) not in lvl.altars:
            return False

        # Drop unidentified items on altar to learn BUC
        for letter, item in self.inventory.items():
            lower = item.lower()
            if '(weapon in hand)' in lower or '(wielded)' in lower or '(being worn)' in lower:
                continue
            if letter in self._buc_status:
                continue  # already known
            if 'blessed' in lower or 'uncursed' in lower or 'cursed' in lower:
                continue  # already identified
            oc = self.inv_oclasses.get(letter, -1)
            if oc in (WEAPON_CLASS, ARMOR_CLASS, POTION_CLASS, SCROLL_CLASS,
                      RING_CLASS, WAND_CLASS, AMULET_CLASS):
                # Drop it
                drop_idx = self._val2idx.get(int(A.Command.DROP))
                if drop_idx is not None:
                    if self._env_step(drop_idx):
                        self._parse_blstats(); raise AgentFinished()
                    l_idx = self._val2idx.get(ord(letter))
                    if l_idx is not None:
                        if self._env_step(l_idx):
                            self._parse_blstats(); raise AgentFinished()
                    self._update_game_state()
                    # Pick it back up to read BUC
                    pickup_idx = self._val2idx.get(int(A.Command.PICKUP))
                    if pickup_idx is not None:
                        if self._env_step(pickup_idx):
                            self._parse_blstats(); raise AgentFinished()
                        self._update_game_state()
                    return True
        return False

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
                    # Stall detection
                    cur_turn = self.blstats.time if self.blstats else 0
                    if cur_turn == stall_turn:
                        stall_count += 1
                        if stall_count > 5:
                            self.step(A.Command.SEARCH)
                            stall_count = 0
                            continue
                    else:
                        stall_turn = cur_turn
                        stall_count = 0

                    # Increment Elbereth age
                    self._elbereth_turns += 1

                    # Priority cascade (AutoAscend-style)
                    # P0: Emergency (stoning, critical HP, starvation)
                    if self.emergency():
                        continue

                    # P1: Proactive prayer
                    if self.pray_strategy():
                        continue

                    # P2: Fight (continuous combat loop)
                    if self.fight():
                        continue

                    # P3a: Eat corpse after kill
                    if self.eat_corpse_after_kill():
                        continue

                    # P3b: Eat from ground
                    if self.eat_ground():
                        continue

                    # P3c: Eat from inventory
                    if self.eat():
                        continue

                    # P4: Pickup useful items
                    if self.pickup_useful():
                        continue

                    # P5: Auto-equip (weapons + armor)
                    self.auto_equip()

                    # P6: Excalibur (dip or seek fountain)
                    if self.dip_excalibur():
                        continue
                    if self.seek_fountain():
                        continue

                    # P6b: Altar BUC identification
                    if self.use_altar():
                        continue

                    # P7: Sokoban
                    if self.sokoban():
                        continue

                    # P8: Explore (doors, frontier, stairs, search)
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
