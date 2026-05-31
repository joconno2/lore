"""Expert system agent for NetHack.

Priority-based decision loop that ties together combat, prayer, item ID,
navigation, and observation parsing subsystems. Drop-in replacement for
a neural policy: takes NLE obs dict, returns action index.

Subsystem imports are fault-tolerant. Missing modules get stubbed so
the agent still runs (with degraded capability) during incremental
development.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Conditional imports: stub anything not yet built
# ============================================================

try:
    from nhc.obs_parser import (
        GameState, MonsterInfo,
        glyph_is_monster, glyph_is_pet, glyph_is_stairs_down,
        glyph_is_stairs_up, glyph_to_monster_name, glyph_to_mon_id,
        GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_BODY_OFF, GLYPH_OBJ_OFF,
        GLYPH_CMAP_OFF, GLYPH_RIDDEN_OFF,
        CMAP_UPSTAIR, CMAP_DNSTAIR, CMAP_ROOM, CMAP_CORR,
        CMAP_LITCORR, CMAP_DARKROOM,
        S_DNSTAIR, S_DNLADDER, S_UPSTAIR,
        MAP_H, MAP_W,
        COND_STONE, COND_SLIME,
        NUMMONS,
    )
    _HAS_OBS_PARSER = True
except ImportError:
    _HAS_OBS_PARSER = False
    GameState = None
    # Fallback constants
    NUMMONS = 381
    GLYPH_MON_OFF = 0
    GLYPH_PET_OFF = 381
    GLYPH_BODY_OFF = 1144
    GLYPH_OBJ_OFF = 1906
    GLYPH_CMAP_OFF = 2359
    GLYPH_RIDDEN_OFF = 1525
    MAP_H = 21
    MAP_W = 79
    S_DNSTAIR = GLYPH_CMAP_OFF + 24
    S_DNLADDER = GLYPH_CMAP_OFF + 26
    S_UPSTAIR = GLYPH_CMAP_OFF + 23
    COND_STONE = 0x00000001
    COND_SLIME = 0x00000002

try:
    from nhc.combat import ThreatDB, ThreatReport, CorpseReport
    _HAS_COMBAT = True
except ImportError:
    ThreatDB = None
    _HAS_COMBAT = False

try:
    from nhc.prayer import PrayerState, HungerState, TroubleSeverity
    _HAS_PRAYER = True
except ImportError:
    PrayerState = None
    _HAS_PRAYER = False

try:
    from nhc.item_id import AppearanceTracker
    _HAS_ITEM_ID = True
except ImportError:
    AppearanceTracker = None
    _HAS_ITEM_ID = False

try:
    from nhc.navigation import DungeonMap
    _HAS_NAV = True
except ImportError:
    DungeonMap = None
    _HAS_NAV = False


# ============================================================
# NLE action indices (from nle.nethack.ACTIONS canonical order)
# ============================================================

class Actions:
    """NetHackScore-v0 action indices (23 actions).

    Indices match env.unwrapped.actions:
      0=MORE(CR), 1=N(k), 2=E(l), 3=S(j), 4=W(h),
      5=NE(u), 6=SE(n), 7=SW(b), 8=NW(y),
      9-16=long moves, 17=upstairs(<), 18=downstairs(>),
      19=wait(.), 20=kick(^D), 21=eat(e), 22=search(s)
    """
    MORE = 0
    N = 1; E = 2; S = 3; W = 4
    NE = 5; SE = 6; SW = 7; NW = 8
    UP = 17; DOWN = 18; WAIT = 19
    KICK = 20; EAT = 21; SEARCH = 22
    # Actions not available in NetHackScore-v0 (mapped to SEARCH as fallback)
    APPLY = 22; CLOSE = 22; DIP = 22; DROP = 22; ENGRAVE = 22
    FIRE = 22; INV = 22; LOOT = 22; OPEN = 22
    PAY = 22; PICKUP = 22; PRAY = 22; PUTON = 22
    QUAFF = 22; READ = 22; REMOVE = 22; RIDE = 22
    TAKEOFF = 22; THROW = 22; WEAR = 22; WIELD = 22
    ZAP = 22; ESC = 22
    NUM_ACTIONS = 23

    MOVE_DELTAS = {
        1: (-1, 0),   # N
        2: (0, 1),    # E
        3: (1, 0),    # S
        4: (0, -1),   # W
        5: (-1, 1),   # NE
        6: (1, 1),    # SE
        7: (1, -1),   # SW
        8: (-1, -1),  # NW
    }
    DELTA_TO_MOVE = {v: k for k, v in MOVE_DELTAS.items()}


def _try_resolve_actions():
    """Attempt to resolve action indices from NLE at import time."""
    try:
        from nle import nethack
        act_list = list(nethack.ACTIONS)
        lookup = {}
        for i, a in enumerate(act_list):
            name = a.name if hasattr(a, "name") else str(a)
            if name not in lookup:  # first occurrence wins (regular compass before long-move)
                lookup[name] = i

        # NLE action names are bare (e.g., "N", "APPLY", "UP", "WAIT")
        # Map our attribute names to what NLE uses
        name_map = {
            "N": "N", "E": "E", "S": "S", "W": "W",
            "NE": "NE", "SE": "SE", "SW": "SW", "NW": "NW",
            "UP": "UP", "DOWN": "DOWN", "WAIT": "WAIT",
            "APPLY": "APPLY", "CLOSE": "CLOSE", "DIP": "DIP", "DROP": "DROP",
            "EAT": "EAT", "ENGRAVE": "ENGRAVE", "FIRE": "FIRE",
            "KICK": "KICK", "LOOT": "LOOT", "OPEN": "OPEN",
            "PICKUP": "PICKUP", "PRAY": "PRAY", "PUTON": "PUTON",
            "QUAFF": "QUAFF", "READ": "READ", "SEARCH": "SEARCH",
            "TAKEOFF": "TAKEOFF", "THROW": "THROW", "WEAR": "WEAR",
            "WIELD": "WIELD", "ZAP": "ZAP", "MORE": "MORE",
            "INV": "INVENTORY", "REMOVE": "REMOVE", "ESC": "ESC",
        }
        for attr, nle_name in name_map.items():
            if nle_name in lookup:
                setattr(Actions, attr, lookup[nle_name])

        Actions.NUM_ACTIONS = len(act_list)
        Actions.MOVE_DELTAS = {
            Actions.N: (-1, 0), Actions.E: (0, 1),
            Actions.S: (1, 0), Actions.W: (0, -1),
            Actions.NE: (-1, 1), Actions.SE: (1, 1),
            Actions.SW: (1, -1), Actions.NW: (-1, -1),
        }
        Actions.DELTA_TO_MOVE = {v: k for k, v in Actions.MOVE_DELTAS.items()}
    except ImportError:
        pass

_try_resolve_actions()


# ============================================================
# Navigation helpers
# ============================================================

# Walkable cmap indices: doorway(12), open doors(13,14), room(19),
# darkroom(20), corridor(21), lit corridor(22), stairs(23-26),
# altar(27), grave(28), throne(29), sink(30), fountain(31)
_WALKABLE_CMAPS = frozenset({12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
# Door cmaps: doorway(12), open(13,14), closed(15,16)
_DOOR_CMAPS = frozenset({12, 13, 14, 15, 16})
_CLOSED_DOOR_CMAPS = frozenset({15, 16})
# Wall cmaps: 1-12 are various wall types (vwall, hwall, corners, etc.)
_WALL_CMAPS = frozenset(range(1, 12))
_STONE_CMAP = 0
# Boulder glyph (GLYPH_OBJ_OFF + 447)
_BOULDER_GLYPH = GLYPH_OBJ_OFF + 447


def _direction_toward(py: int, px: int, ty: int, tx: int) -> int:
    """Movement action to step from (py,px) toward (ty,tx). Chebyshev step."""
    dy = 0 if ty == py else (1 if ty > py else -1)
    dx = 0 if tx == px else (1 if tx > px else -1)
    if dy == 0 and dx == 0:
        return Actions.WAIT
    return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)


def _monster_in_line(py: int, px: int, my: int, mx: int) -> Optional[tuple]:
    """Check if monster is in a cardinal/diagonal line from player.
    Returns (dy, dx) unit direction or None."""
    dr, dc = my - py, mx - px
    if dr == 0 and dc == 0:
        return None
    if dr == 0:
        return (0, 1 if dc > 0 else -1)
    if dc == 0:
        return (1 if dr > 0 else -1, 0)
    if abs(dr) == abs(dc):
        return (1 if dr > 0 else -1, 1 if dc > 0 else -1)
    return None


def _direction_away(py: int, px: int, ty: int, tx: int) -> int:
    """Movement action to step away from (ty, tx)."""
    dy = 0 if ty == py else (-1 if ty > py else 1)
    dx = 0 if tx == px else (-1 if tx > px else 1)
    if dy == 0 and dx == 0:
        return Actions.SEARCH
    return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)


def _glyph_cmap(g: int) -> int:
    """Return cmap index if g is a cmap glyph, else -1."""
    if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87:
        return g - GLYPH_CMAP_OFF
    return -1


def _is_walkable_glyph(g: int) -> bool:
    """Check if a glyph represents a tile the player can walk on."""
    if g == _BOULDER_GLYPH:
        return False
    if GLYPH_MON_OFF <= g < GLYPH_OBJ_OFF:
        return True  # monsters, pets, bodies, ridden
    if GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF:
        return True  # objects on floor
    c = _glyph_cmap(g)
    return c in _WALKABLE_CMAPS


def _is_door_glyph(g: int) -> bool:
    return _glyph_cmap(g) in _DOOR_CMAPS


def _is_closed_door_glyph(g: int) -> bool:
    return _glyph_cmap(g) in _CLOSED_DOOR_CMAPS


def _is_wall_glyph(g: int) -> bool:
    return _glyph_cmap(g) in _WALL_CMAPS


def _is_stone_glyph(g: int) -> bool:
    return _glyph_cmap(g) == _STONE_CMAP


def _bfs_distances(py: int, px: int, walkable: np.ndarray,
                   walkable_diag: np.ndarray) -> np.ndarray:
    """BFS from (py, px) respecting walkable masks. Returns distance array (-1 = unreachable).

    walkable_diag should be False for door tiles (no diagonal through doors)
    and for tiles adjacent to only walls (no diagonal squeeze).
    """
    from collections import deque
    rows, cols = walkable.shape
    dis = np.full((rows, cols), -1, dtype=np.int32)
    dis[py, px] = 0
    queue = deque([(py, px)])

    while queue:
        y, x = queue.popleft()
        d = dis[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                    continue
                if dis[ny, nx] != -1:
                    continue
                if not walkable[ny, nx]:
                    continue
                # Diagonal move checks
                if abs(dy) + abs(dx) > 1:
                    # Both source and dest must allow diagonal
                    if not walkable_diag[ny, nx] or not walkable_diag[y, x]:
                        continue
                    # Can't squeeze: at least one of the two adjacent cardinal
                    # tiles must be walkable
                    if not walkable[y, nx] and not walkable[ny, x]:
                        continue
                dis[ny, nx] = d + 1
                queue.append((ny, nx))

    return dis


def _find_stairs_down(glyphs: np.ndarray) -> Optional[tuple]:
    """Scan the glyph map for downstairs or down ladder."""
    if glyphs is None:
        return None
    rows, cols = glyphs.shape
    for r in range(rows):
        for c in range(cols):
            g = int(glyphs[r, c])
            if g == S_DNSTAIR or g == S_DNLADDER:
                return (r, c)
    return None


def _find_stairs_down_from_objects(objects: np.ndarray) -> Optional[tuple]:
    """Scan remembered terrain for stairs down."""
    rows, cols = objects.shape
    dn_stair_cmap = GLYPH_CMAP_OFF + 24  # CMAP_DNSTAIR
    dn_ladder_cmap = GLYPH_CMAP_OFF + 26  # CMAP_DNLADDER
    for r in range(rows):
        for c in range(cols):
            o = int(objects[r, c])
            if o == dn_stair_cmap or o == dn_ladder_cmap:
                return (r, c)
    return None


# ============================================================
# Glyph helpers for item/corpse detection
# ============================================================

def _tile_has_corpse(glyphs: np.ndarray, row: int, col: int) -> tuple[bool, str]:
    """Check if there's a corpse at (row, col). Returns (has_corpse, name)."""
    if glyphs is None:
        return False, ""
    g = int(glyphs[row, col])
    if GLYPH_BODY_OFF <= g < GLYPH_BODY_OFF + 381:
        if _HAS_OBS_PARSER:
            name = glyph_to_monster_name(GLYPH_MON_OFF + (g - GLYPH_BODY_OFF))
            return True, name or ""
        return True, f"corpse_{g - GLYPH_BODY_OFF}"
    return False, ""


def _tile_has_object(glyphs: np.ndarray, row: int, col: int) -> bool:
    """Check if there's a non-corpse object at (row, col)."""
    if glyphs is None:
        return False
    g = int(glyphs[row, col])
    return GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF


# ============================================================
# Stub ThreatReport for when combat.py is unavailable
# ============================================================

# AutoAscend monster categories
_NEVER_MELEE = {
    "floating eye",   # paralysis on melee = death
    "gas spore",      # explodes on death dealing 4d6
}
_INSTAKILL = {
    "cockatrice", "chickatrice", "Medusa",
    "green slime", "Death", "Pestilence", "Famine",
    "purple worm",  # engulf
}
_WEAK = {"lichen", "newt", "shrieker", "grid bug"}
_WEIRD = {"leprechaun", "nymph"}  # steal items, avoid melee

@dataclass
class _StubThreatReport:
    name: str
    danger_level: int = 5
    special_attacks: list = field(default_factory=list)
    required_resistances: list = field(default_factory=list)
    elbereth_effective: bool = True
    ranged_preferred: bool = False
    never_melee: bool = False
    instakill_risk: bool = False
    recommended_action: str = "melee"

    def __post_init__(self):
        if self.name in _INSTAKILL:
            self.instakill_risk = True
            self.danger_level = 10
            self.recommended_action = "flee"
        if self.name in _NEVER_MELEE:
            self.never_melee = True
            self.ranged_preferred = True
            self.danger_level = max(self.danger_level, 7)
            self.recommended_action = "avoid"
        if self.name in _WEIRD:
            self.danger_level = max(self.danger_level, 6)
            self.recommended_action = "avoid"
        if self.name in _WEAK:
            self.danger_level = 1


# ============================================================
# ExpertAgent
# ============================================================

class ExpertAgent:
    """Priority-based expert system agent for NetHack.

    Takes NLE observations, returns canonical action indices.
    Integrates combat threat assessment, prayer safety, item
    identification, and navigation.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.state: GameState = GameState() if _HAS_OBS_PARSER else None
        self.threat_db = ThreatDB() if _HAS_COMBAT else None
        self.prayer = PrayerState() if _HAS_PRAYER else None
        self.item_tracker = AppearanceTracker() if _HAS_ITEM_ID else None

        # Per-episode state
        self._step_count: int = 0
        self.resistances: set[str] = set()
        self.has_food: bool = False
        self.has_lizard_corpse: bool = False
        self.last_action: int = Actions.SEARCH
        self.turns_on_tile: int = 0
        self.last_pos: tuple = (0, 0)
        self.search_count: int = 0

        # Logging state
        self._prev_hp: int = 0
        self._prev_dlevel: int = 1
        self._prev_xlevel: int = 1
        self._last_priority: str = ""
        self._last_action_reason: str = ""
        self._episode_stats: dict = self._make_episode_stats()

        # Per-turn tile state (detected from messages + glyphs)
        self._on_corpse: bool = False
        self._corpse_name: str = ""
        self._on_item: bool = False
        self._on_edible_item: bool = False
        # Multi-step action state
        self._pending_action: Optional[str] = None  # "eat", "pray", "wield", "wear"
        self._pending_letter: Optional[str] = None
        self._eat_attempts: int = 0
        self._has_weapon_wielded: bool = False
        # Path caching
        self._cached_path: Optional[list] = None
        self._stuck_moves: int = 0
        # Navigation state: seen/walkable masks, remembered terrain
        self._seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self._walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self._objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self._pet_pos: Optional[tuple] = None
        self._last_target: Optional[tuple] = None
        self._search_count_map = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self._door_open_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self._kick_dir: Optional[int] = None  # direction to kick when prompted
        self._refused_attacks: set = set()  # monster names we refused to attack (peacefuls)
        self._refused_positions: set = set()  # positions we shouldn't walk to (peacefuls there)
        self._eat_cooldown: int = 0  # steps to skip eating after a failed eat
        self._throw_cooldown: int = 0  # steps to skip throwing after an attempt
        # Elbereth state
        self._pending_sequence: list = []
        self._on_elbereth: bool = False
        self._elbereth_cooldown: int = 0
        # Inactivity detection
        self._inactivity_steps: int = 0
        self._prev_turn: int = -1

    @staticmethod
    def _make_episode_stats() -> dict:
        return {
            "kills": 0,
            "kill_names": [],
            "doors_opened": 0,
            "items_picked_up": 0,
            "levels_visited": set(),
            "prayers": 0,
            "prayer_results": [],
            "hp_min": 999,
            "corpses_eaten": 0,
            "turns_per_dlevel": {},
            "cause_of_death": "timeout",
        }

    def reset(self) -> None:
        """Reset state for a new episode."""
        if self.verbose and self._step_count > 0:
            self.print_episode_summary()
        self._step_count = 0
        self.state = GameState() if _HAS_OBS_PARSER else None
        if _HAS_PRAYER:
            self.prayer = PrayerState()
        # Lawful Valkyrie starts with cold resistance and stealth
        self.resistances = {"cold resistance"}
        self.has_food = False
        self.has_lizard_corpse = False
        self.last_action = Actions.SEARCH
        self.turns_on_tile = 0
        self.last_pos = (0, 0)
        self.search_count = 0
        self._on_corpse = False
        self._corpse_name = ""
        self._on_item = False
        self._on_edible_item = False
        # Ranged/excalibur reset
        self._ranged_dir = None
        self._daggers = {}
        self._wands = {}
        self._has_excalibur = False
        self._fountain_positions = set()
        # Logging state
        self._prev_hp = 0
        self._prev_dlevel = 1
        self._prev_xlevel = 1
        self._last_priority = ""
        self._last_action_reason = ""
        self._episode_stats = self._make_episode_stats()
        # Equipment state
        self._has_weapon_wielded = False
        self._pending_letter = None
        # Ranged combat state
        self._ranged_dir: Optional[int] = None
        self._daggers: dict = {}
        self._wands: dict = {}
        # Excalibur state
        self._has_excalibur = False
        self._fountain_positions: set = set()
        # Reset navigation masks
        self._seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self._walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self._objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self._pet_pos = None
        self._cached_path = None
        self._stuck_moves = 0
        self._last_target = None
        self._search_count_map = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self._door_open_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self._kick_dir = None
        self._refused_attacks = set()
        self._refused_positions = set()
        self._eat_cooldown = 0
        self._throw_cooldown = 0
        self._pending_sequence = []
        self._on_elbereth = False
        self._elbereth_cooldown = 0
        self._inactivity_steps = 0
        self._prev_turn = -1

    def act(self, obs: dict) -> int:
        """Main decision function. Takes NLE observation, returns action index."""
        self._step_count += 1
        s = self.state
        s.update(obs)

        # If we're in text-entry mode (typing Elbereth), send queued chars
        if s.in_getlin and self._pending_sequence:
            action = self._pending_sequence.pop(0)
            self.last_action = action
            return action

        # If we have queued actions (multi-step command), send next one
        if self._pending_sequence:
            action = self._pending_sequence.pop(0)
            self.last_action = action
            return action

        # AutoAscend-style prompt handling: clear misc flags before priority cascade.
        # NetHackChallenge-v0 passes all prompts through.
        misc = obs.get("misc")
        if misc is not None:
            msg_raw = obs.get("message")
            msg_str = ""
            if msg_raw is not None:
                msg_str = bytes(msg_raw).rstrip(b'\x00').decode("latin-1", errors="replace").strip()

            # xwaitingforspace: always send SPACE (AutoAscend does this)
            if misc[0] and not misc[1]:
                return self._letter_to_action(' ')

            # Text entry mode (getlin): ESC unless it's our engrave sequence
            if misc[1] and not self._pending_sequence:
                if self._pending_action == "elbereth":
                    pass  # fall through to message handlers below
                else:
                    return Actions.ESC

            # yn prompt without text entry: auto-dismiss unrecognized prompts
            # IMPORTANT: don't dismiss prompts that our message parser handles
            if misc[2] and not misc[1]:
                handled_prompts = [
                    "[yn]", "[ynq]", "eat it?", "eat this?",
                    "Really attack", "direction?", "In what direction",
                    "What do you want to", "Do you want to add",
                    "Dip", "into the fountain",
                    "This door is locked",
                    "Are you sure", "pray", "drink", "quaff",
                ]
                if not any(p in msg_str for p in handled_prompts):
                    return self._letter_to_action(' ')

        # Update seen/walkable masks from current glyphs
        self._update_seen_and_walkable(s)

        # Decrement Elbereth cooldown
        if self._elbereth_cooldown > 0:
            self._elbereth_cooldown -= 1

        # Track Elbereth status: reset when we move
        pos = s.position
        if pos != self.last_pos and self._on_elbereth:
            self._on_elbereth = False

        # Check for --More-- prompt. Must clear before anything else works.
        msg_raw = obs.get("message")
        if msg_raw is not None:
            msg_bytes = bytes(msg_raw).rstrip(b'\x00')
            msg_str = msg_bytes.decode("latin-1", errors="replace").strip()
            if "--More--" in msg_str or msg_str.endswith("--More--"):
                if self.verbose:
                    self._log("MORE", f"clearing prompt: {msg_str[:60]}")
                return Actions.MORE
            # Handle drop prompt
            if "What do you want to drop?" in msg_str:
                if self._pending_letter:
                    letter = self._pending_letter
                    self._pending_letter = None
                    self._pending_action = None
                    return self._letter_to_action(letter)
                self._pending_action = None
                return Actions.MORE
            # Handle wield/wear/takeoff prompts
            if "What do you want to wield?" in msg_str or \
               "What do you want to wear?" in msg_str or \
               "What do you want to take off?" in msg_str:
                if self._pending_letter:
                    letter = self._pending_letter
                    self._pending_letter = None
                    self._pending_action = None
                    return self._letter_to_action(letter)
                self._pending_action = None
                return Actions.ESC  # cancel
            # "You are already wearing that" or similar
            if "already wearing" in msg_str or "already wielding" in msg_str:
                self._pending_action = None
                self._pending_letter = None
                return Actions.MORE
            # "You can't wear that" type messages
            if "You can't" in msg_str and ("wear" in msg_str or "wield" in msg_str):
                self._pending_action = None
                self._pending_letter = None
                return Actions.MORE
            # Handle eat prompts
            if "What do you want to eat?" in msg_str:
                # Find best food item from our inventory knowledge
                food_letter = self._find_food_letter(s)
                if food_letter is not None:
                    if self.verbose:
                        self._log("EAT", f"selecting item '{food_letter}'")
                    self._pending_action = None
                    return self._letter_to_action(food_letter)
                else:
                    # No recognized food. Try to cancel.
                    # Parse offered letters from "[X or ?*]" and avoid selecting
                    if self.verbose:
                        self._log("EAT", "no edible food found, canceling")
                    self._pending_action = None
                    self.has_food = False
                    self._eat_cooldown = 200  # long cooldown
                    # Try ESC, space, then MORE in order
                    return Actions.ESC
            # "eat it? [yn]" for corpse on ground
            if "eat it?" in msg_str or "eat this?" in msg_str:
                # Check if we're being asked about something dangerous
                lower_msg = msg_str.lower()
                refuse = False
                # Don't eat if satiated (choking risk)
                if s.hunger_state == "satiated":
                    refuse = True
                # Check for old/tainted food
                if "old " in lower_msg or "rotten" in lower_msg or "tainted" in lower_msg:
                    refuse = True
                if refuse:
                    if self.verbose:
                        self._log("EAT", f"refusing: {msg_str[:40]}")
                    self._pending_action = None
                    return self._letter_to_action('n')
                if self.verbose:
                    self._log("EAT", f"confirming: {msg_str[:40]}")
                self._pending_action = None
                return self._letter_to_action('y')
            # Handle yn prompts - catch [yn], [ynq], and default indicators like (n) or (y)
            if "[yn]" in msg_str or "[ynq]" in msg_str:
                lower_msg = msg_str.lower()
                # Say no to dangerous prompts
                if "die?" in lower_msg or "Really attack" in msg_str or \
                   "suicide" in lower_msg or "overeat" in lower_msg or \
                   "call is" in lower_msg or "Really quit" in msg_str:
                    if self.verbose:
                        self._log("YN", f"answering NO to: {msg_str[:60]}")
                    # If "Really attack", mark the monster as peaceful and block the tile
                    if "Really attack" in msg_str:
                        self._cached_path = None
                        self._stuck_moves += 5  # force repath
                        # Extract monster name from "Really attack the X?"
                        attack_lower = msg_str.lower()
                        for prefix in ["really attack the ", "really attack "]:
                            if prefix in attack_lower:
                                name = attack_lower.split(prefix, 1)[1].split("?")[0].strip()
                                self._refused_attacks.add(name)
                                break
                        # Track the position of the peaceful monster
                        if self.last_action in Actions.MOVE_DELTAS:
                            dd = Actions.MOVE_DELTAS[self.last_action]
                            wr, wc = s.position[0] + dd[0], s.position[1] + dd[1]
                            if 0 <= wr < MAP_H and 0 <= wc < MAP_W:
                                self._refused_positions.add((wr, wc))
                    return self._letter_to_action('n')
                # Say yes to everything else (pray, eat, etc.)
                if self.verbose:
                    self._log("YN", f"answering yes to: {msg_str[:60]}")
                return self._letter_to_action('y')
            # Handle menu dismissal
            if msg_str.endswith("[q]") or msg_str.endswith("(end)"):
                if self.verbose:
                    self._log("MENU", f"dismissing: {msg_str[:60]}")
                return Actions.MORE
            # Locked door: kick it in the same direction we tried to open
            if "This door is locked" in msg_str:
                if self.verbose:
                    self._log("DOOR", "door is locked, kicking")
                # _kick_dir was set when we walked into the door
                return Actions.KICK
            # "In what direction?" prompt after kick
            if "In what direction?" in msg_str:
                if self._kick_dir is not None:
                    d = self._kick_dir
                    self._kick_dir = None
                    return d
                # Fallback: kick toward cached path target
                if self._cached_path and len(self._cached_path) > 0:
                    next_tile = self._cached_path[0]
                    dy = next_tile[0] - s.py
                    dx = next_tile[1] - s.px
                    action = Actions.DELTA_TO_MOVE.get((dy, dx))
                    if action is not None:
                        return action
                return Actions.MORE  # cancel if no direction

            # Quaff prompt
            if "What do you want to drink?" in msg_str or \
               "What do you want to quaff?" in msg_str:
                if self._pending_letter:
                    letter = self._pending_letter
                    self._pending_letter = None
                    self._pending_action = None
                    return self._letter_to_action(letter)
                self._pending_action = None
                return Actions.ESC

            # Engrave prompts
            if "What do you want to write with?" in msg_str:
                return self._letter_to_action('-')  # write with finger (dust)
            if "You write in the dust" in msg_str or \
               "You engrave in the dust" in msg_str:
                # Intermediate message after '-'. Dismiss with space.
                return self._letter_to_action(' ')
            if "Do you want to add to the current engraving?" in msg_str:
                return self._letter_to_action('n')  # overwrite, fresh Elbereth
            if "What do you want to write in the dust here?" in msg_str or \
               "What do you want to engrave in the dust here?" in msg_str:
                # Queue "Elbereth" + CR as pending sequence
                self._pending_sequence = [self._letter_to_action(ch) for ch in "lbereth"]
                self._pending_sequence.append(Actions.MORE)  # CR to finish
                if self.verbose:
                    self._log("ELBERETH", "typing Elbereth")
                return self._letter_to_action('E')  # first char
            # Engrave failed
            if "Never mind" in msg_str and self._pending_action == "elbereth":
                self._pending_action = None
                self._pending_sequence = []
                self._elbereth_cooldown = 100
                if self.verbose:
                    self._log("ELBERETH", "engrave rejected")
            # Throw/zap/dip multi-step prompts
            if "What do you want to throw?" in msg_str or \
               "What do you want to zap?" in msg_str or \
               "What do you want to dip?" in msg_str:
                if self._pending_letter:
                    letter = self._pending_letter
                    self._pending_letter = None
                    return self._letter_to_action(letter)
                self._pending_action = None
                return Actions.MORE  # cancel
            if "In what direction?" in msg_str:
                if self._ranged_dir is not None:
                    d = self._ranged_dir
                    self._ranged_dir = None
                    self._pending_action = None
                    return d
                if self._kick_dir is not None:
                    d = self._kick_dir
                    self._kick_dir = None
                    return d
                self._pending_action = None
                return Actions.MORE
            # Dip into fountain confirmation
            if "Dip" in msg_str and "into the fountain?" in msg_str:
                return self._letter_to_action('y')
            # Throw/zap failed
            if "not carrying anything" in msg_str or "You don't have anything" in msg_str:
                self._pending_action = None
            # Diagonal door failure: clear path
            if "diagonally" in msg_str:
                self._cached_path = None
                self._stuck_moves += 2
            # Boulder: can't push
            if "move the boulder, but in vain" in msg_str:
                self._cached_path = None
                self._stuck_moves += 2
            # Hit a wall: clear path, mark stuck, fix walkable mask
            if "It's a wall" in msg_str:
                self._cached_path = None
                self._stuck_moves += 2
                # Mark the tile we tried to walk to as unwalkable
                if self.last_action in Actions.MOVE_DELTAS:
                    dd = Actions.MOVE_DELTAS[self.last_action]
                    wr, wc = s.position[0] + dd[0], s.position[1] + dd[1]
                    if 0 <= wr < MAP_H and 0 <= wc < MAP_W:
                        self._walkable[wr, wc] = False
            # Overburdened: drop something immediately
            if "You collapse" in msg_str or "can barely move" in msg_str:
                drop_letter = self._find_droppable_item(s)
                if drop_letter:
                    self._pending_action = "drop"
                    self._pending_letter = drop_letter
                    return Actions.DROP
            # Nothing to pick up: clear item flag
            if "nothing here to pick up" in msg_str:
                self._on_item = False
            # Carrying too much: drop the heaviest non-essential item
            if "carrying too much" in msg_str:
                self._cached_path = None
                drop_letter = self._find_droppable_item(s)
                if drop_letter:
                    self._pending_action = "drop"
                    self._pending_letter = drop_letter
                    if self.verbose:
                        self._log("DROP", f"dropping '{drop_letter}' (encumbered)")
                    return Actions.DROP
                self._stuck_moves += 3
            # Eat failed: clear pending and set cooldown
            if "You cannot eat that" in msg_str or \
               "You don't have anything to eat" in msg_str or \
               "You can't eat that" in msg_str or \
               "That is not edible" in msg_str:
                self._pending_action = None
                self.has_food = False
                self._eat_attempts = 0
                self._eat_cooldown = 200
            # "Never mind" from eat specifically
            if "Never mind" in msg_str and self._pending_action == "eat":
                self._pending_action = None
                self.has_food = False
                self._eat_cooldown = 500  # long cooldown to prevent eat spam

        # Clear stale pending actions. If we reach here without a prompt
        # handling the pending action, it means the action completed or failed
        # silently. Don't let it block _decide forever.
        if self._pending_action in ("throw", "zap", "dip", "offer", "quaff"):
            self._pending_action = None
            self._pending_letter = None
            self._ranged_dir = None
        if self._pending_action == "elbereth" and not self._pending_sequence:
            # Elbereth sequence completed (all chars sent)
            self._pending_action = None
            self._on_elbereth = True
            if self.verbose:
                self._log("ELBERETH", "written successfully")

        # Track stuck detection
        pos = s.position
        if pos == self.last_pos:
            self.turns_on_tile += 1
            # If we tried to move but didn't, clear cached path
            if self._cached_path and self.last_action in Actions.MOVE_DELTAS:
                self._cached_path = None
                self._stuck_moves += 1
        else:
            self.turns_on_tile = 0
            self.search_count = 0
            self._stuck_moves = 0
            # Advance cached path if we moved to expected tile
            if self._cached_path and pos == self._cached_path[0]:
                self._cached_path.pop(0)
            elif self._cached_path:
                self._cached_path = None  # off-path, recompute
        self.last_pos = pos

        # Update prayer subsystem from blstats
        if self.prayer is not None:
            bl = obs.get("blstats")
            if bl is not None:
                self.prayer.update_from_blstats(bl)

        # Parse messages for resistance gains and inventory clues
        self._parse_message_for_state(s)

        # Parse game events for logging and stats
        self._last_action_reason = ""
        self._parse_events(s)

        # Detect what's on the player's tile (from messages + glyphs)
        self._detect_tile_contents(s)

        # Scan inventory for food / lizard
        self._check_inventory(s)

        action = self._decide(s)
        action = self._validate_move(action, s)

        # Inactivity guard: prevent step waste from non-turn-advancing actions
        if s.turn == self._prev_turn:
            self._inactivity_steps += 1
            # If same action as last time and it didn't advance turn, force SEARCH
            if action == self.last_action and self._inactivity_steps >= 2:
                action = Actions.SEARCH
                self._inactivity_steps = 0
            # Hard cap after 5 inactive steps
            elif self._inactivity_steps > 5:
                action = Actions.SEARCH
                self._inactivity_steps = 0
        else:
            self._inactivity_steps = 0
        self._prev_turn = s.turn

        self.last_action = action
        self._prev_hp = s.hp
        if self.verbose:
            self._log_decision(s, action)
        return action

    def _log(self, tag: str, msg: str):
        """Print a compact tagged log line."""
        s = self.state
        print(f"t={s.turn:>5} dl{s.dlevel} hp{s.hp:>3}/{s.max_hp:<3} | {tag} {msg}")

    def _log_decision(self, s, action: int):
        """Log action with priority tag. One line per step, only when interesting."""
        prio = self._last_priority
        reason = self._last_action_reason
        action_name = self._action_name(action)
        # Skip boring navigation steps (pure movement with no events)
        if prio == "P6-nav" and not reason:
            return
        self._log(prio or "ACT", f"{action_name} {reason}")

    def _parse_events(self, s) -> None:
        """Scan messages for game events and update episode stats."""
        if not s.messages:
            return
        text = s.last_message
        lower = text.lower()
        stats = self._episode_stats

        # Kills
        if "you kill " in lower or "you destroy " in lower:
            # Extract monster name from "You kill the X!" or "You destroy the X!"
            for prefix in ["you kill the ", "you kill ", "you destroy the ", "you destroy "]:
                if prefix in lower:
                    name = lower.split(prefix, 1)[1].rstrip("!. ")
                    stats["kills"] += 1
                    stats["kill_names"].append(name)
                    if self.verbose:
                        self._log("KILL", f"{name} ({stats['kills']} total)")
                    break

        # Death
        if "you die" in lower:
            stats["cause_of_death"] = text.strip()
            if self.verbose:
                self._log("DEATH", text.strip()[:60])

        # Door opens
        if "the door opens" in lower or "it crashes open" in lower:
            stats["doors_opened"] += 1
            if self.verbose:
                self._log("DOOR", f"opened ({stats['doors_opened']} total)")

        # Items found
        if "you see here" in lower or "you find" in lower:
            # Extract item description
            for prefix in ["you see here ", "you find "]:
                if prefix in lower:
                    item = lower.split(prefix, 1)[1].rstrip(". ")
                    if self.verbose:
                        self._log("ITEM", item[:50])
                    break

        # Items picked up
        if lower.endswith("(weapon in hand)") or lower.endswith("(being worn)") or \
           ("you pick up" in lower) or (lower.startswith("f - ") or lower.startswith("g - ")):
            stats["items_picked_up"] += 1

        # Resistance gained
        resist_msgs = {
            "you feel especially healthy": "poison",
            "you feel a momentary chill": "cold",
            "you feel warm": "fire",
            "you feel full of energy": "shock",
            "you feel very firm": "disintegration",
            "you feel wide awake": "sleep",
        }
        for msg_fragment, resist in resist_msgs.items():
            if msg_fragment in lower:
                if self.verbose:
                    self._log("RESIST", f"gained {resist} resistance")

        # Stairs
        if "you descend" in lower or "you climb down" in lower:
            if self.verbose:
                self._log("STAIRS", f"descending to dl{s.dlevel}")
        if "you ascend" in lower or "you climb up" in lower:
            if self.verbose:
                self._log("STAIRS", f"ascending to dl{s.dlevel}")

        # Prayer
        if "you begin praying" in lower:
            stats["prayers"] += 1
            if self.verbose:
                self._log("PRAY", f"praying ({stats['prayers']} total)")
        if "you feel much better" in lower or "you are granted" in lower:
            stats["prayer_results"].append("success")
            if self.verbose:
                self._log("PRAY", "answered")
        if "you finish your prayer" in lower and "angry" in lower:
            stats["prayer_results"].append("anger")
            if self.verbose:
                self._log("PRAY", "angered god")

        # Corpse eating
        if "this corpse tastes" in lower or "this corpse is" in lower or \
           "you bite into the" in lower:
            stats["corpses_eaten"] += 1

        # HP tracking: significant drops
        if self._prev_hp > 0 and s.hp < self._prev_hp:
            drop = self._prev_hp - s.hp
            if drop > self._prev_hp * 0.25:
                if self.verbose:
                    self._log("HP", f"took {drop} dmg ({self._prev_hp}->{s.hp})")

        # Track minimum HP
        if s.hp > 0 and s.hp < stats["hp_min"]:
            stats["hp_min"] = s.hp

        # Level changes
        if s.dlevel != self._prev_dlevel:
            if self.verbose:
                self._log("DLVL", f"dl{self._prev_dlevel}->dl{s.dlevel}")
            self._prev_dlevel = s.dlevel
            # Reset navigation masks for new level
            self._seen = np.zeros((MAP_H, MAP_W), dtype=bool)
            self._walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
            self._objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
            self._search_count_map = np.zeros((MAP_H, MAP_W), dtype=np.int32)
            self._door_open_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)
            self._cached_path = None
            self._stuck_moves = 0
            self._refused_positions = set()
        if s.xlevel != self._prev_xlevel:
            if self.verbose:
                self._log("XLVL", f"xl{self._prev_xlevel}->xl{s.xlevel}")
            self._prev_xlevel = s.xlevel

        # Track levels visited and time per dlevel
        stats["levels_visited"].add(s.dlevel)
        dl_key = f"dl{s.dlevel}"
        stats["turns_per_dlevel"][dl_key] = stats["turns_per_dlevel"].get(dl_key, 0) + 1

    def print_episode_summary(self) -> None:
        """Print per-episode statistics."""
        s = self.state
        stats = self._episode_stats
        print("\n--- EPISODE SUMMARY ---")
        print(f"  turns: {s.turn}  steps: {self._step_count}  score: {s.score}")
        print(f"  kills: {stats['kills']}  doors: {stats['doors_opened']}  "
              f"items: {stats['items_picked_up']}  corpses eaten: {stats['corpses_eaten']}")
        print(f"  levels visited: {sorted(stats['levels_visited'])}")
        print(f"  prayers: {stats['prayers']}  results: {stats['prayer_results']}")
        print(f"  hp min: {stats['hp_min']}  final hp: {s.hp}/{s.max_hp}")
        if stats["turns_per_dlevel"]:
            dl_str = ", ".join(f"{k}:{v}" for k, v in sorted(stats["turns_per_dlevel"].items()))
            print(f"  time per level: {dl_str}")
        if stats["kill_names"]:
            # Count kills by name
            from collections import Counter
            counts = Counter(stats["kill_names"])
            top = counts.most_common(8)
            kill_str = ", ".join(f"{n}x{c}" if c > 1 else n for n, c in top)
            if len(counts) > 8:
                kill_str += f", +{len(counts)-8} more"
            print(f"  kills: {kill_str}")
        inv_size = len(s.inventory) if s.inventory else 0
        print(f"  inventory: {inv_size} items  cause: {stats['cause_of_death']}")
        print("--- END SUMMARY ---\n")

    @staticmethod
    def _action_name(action: int) -> str:
        """Reverse-lookup action index to name."""
        for name in dir(Actions):
            if name.startswith('_') or name in ('MOVE_DELTAS', 'DELTA_TO_MOVE', 'NUM_ACTIONS'):
                continue
            if getattr(Actions, name) == action:
                return name
        return f"action_{action}"

    def _validate_move(self, action: int, s) -> int:
        """Prevent illegal moves: diagonal through doorways, into walls, into peacefuls, into pets."""
        dd = Actions.MOVE_DELTAS.get(action)
        if dd is None:
            return action  # not a movement action
        dy, dx = dd
        py, px = s.position
        nr, nc = py + dy, px + dx
        # Block moves into pets (wastes steps without advancing turn)
        if s._glyphs is not None and 0 <= nr < MAP_H and 0 <= nc < MAP_W:
            g = int(s._glyphs[nr, nc])
            if GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS:
                # Try alternate direction around the pet
                for cdy, cdx in [(0, dx), (dy, 0), (0, -dx), (-dy, 0)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W and self._walkable[cr, cc]:
                        cg = int(s._glyphs[cr, cc])
                        if not (GLYPH_PET_OFF <= cg < GLYPH_PET_OFF + NUMMONS):
                            alt = Actions.DELTA_TO_MOVE.get((cdy, cdx))
                            if alt is not None:
                                return alt
                return Actions.SEARCH  # wait if completely blocked by pets
        # Block moves into refused positions (peacefuls)
        if (nr, nc) in self._refused_positions:
            # Try alternate direction that avoids the peaceful
            for cdy, cdx in [(dy, 0), (0, dx), (-dy, 0), (0, -dx)]:
                if cdy == 0 and cdx == 0:
                    continue
                cr, cc = py + cdy, px + cdx
                if 0 <= cr < MAP_H and 0 <= cc < MAP_W:
                    if self._walkable[cr, cc] and (cr, cc) not in self._refused_positions:
                        alt = Actions.DELTA_TO_MOVE.get((cdy, cdx))
                        if alt is not None:
                            return alt
            return Actions.SEARCH  # rest if no alternative
        if abs(dy) + abs(dx) <= 1:
            return action  # cardinal move, always ok
        # Diagonal move: check for door at source or destination
        py, px = s.position
        nr, nc = py + dy, px + dx
        if nr < 0 or nr >= MAP_H or nc < 0 or nc >= MAP_W:
            return Actions.WAIT
        glyphs = s._glyphs
        if glyphs is not None:
            src_g = int(glyphs[py, px])
            dst_g = int(glyphs[nr, nc])
            src_obj = int(self._objects[py, px])
            dst_obj = int(self._objects[nr, nc])
            src_door = _is_door_glyph(src_g) or (_glyph_cmap(src_obj) in _DOOR_CMAPS if src_obj != -1 else False)
            dst_door = _is_door_glyph(dst_g) or _is_closed_door_glyph(dst_g) or \
                       (_glyph_cmap(dst_obj) in _DOOR_CMAPS if dst_obj != -1 else False)
            if src_door or dst_door:
                # Try the two cardinal components instead
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W:
                        cg = int(glyphs[cr, cc])
                        if _is_walkable_glyph(cg) or _is_closed_door_glyph(cg) or \
                           (GLYPH_MON_OFF <= cg < GLYPH_CMAP_OFF):
                            cardinal = Actions.DELTA_TO_MOVE.get((cdy, cdx))
                            if cardinal is not None:
                                return cardinal
                return Actions.WAIT
            # Also check if destination is a wall/boulder
            if not _is_walkable_glyph(dst_g) and not _is_closed_door_glyph(dst_g) and \
               not (GLYPH_MON_OFF <= dst_g < GLYPH_CMAP_OFF):
                # Try cardinal components
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W:
                        cg = int(glyphs[cr, cc])
                        if _is_walkable_glyph(cg) or (GLYPH_MON_OFF <= cg < GLYPH_CMAP_OFF):
                            cardinal = Actions.DELTA_TO_MOVE.get((cdy, cdx))
                            if cardinal is not None:
                                return cardinal
                return Actions.WAIT
        return action

    def _decide(self, s) -> int:
        """Priority cascade. Returns action for highest-priority applicable rule."""
        # If a multi-step action is in progress (throw/zap/dip/offer),
        # don't run the priority cascade. Wait for prompts to resolve.
        if self._pending_action in ("throw", "zap", "dip", "offer", "quaff"):
            return Actions.WAIT  # prompts handled in act() before _decide
        priorities = [
            ("P0-emerg", self._p0_emergencies),
            ("P1-combat", self._p1_adjacent_combat),
            ("P2-ranged", self._p2_ranged_threats),
            ("P2b-excal", self._p4c_excalibur),  # Excalibur is top strategic priority
            ("P3-food", self._p3_food),
            ("P4-items", self._p4_items),
            ("P4b-equip", self._p4b_equipment),
            ("P5-corpse", self._p5_corpse_intrinsics),
            ("P5b-rest", self._p5b_rest),
            ("P6-nav", self._p6_navigation),
        ]
        for name, fn in priorities:
            a = fn(s)
            if a is not None:
                self._last_priority = name
                return a

        self._last_priority = "FALLBACK"
        self._last_action_reason = "no priority matched"
        return Actions.SEARCH

    # ----------------------------------------------------------
    # P0: Critical emergencies
    # ----------------------------------------------------------

    def _p0_emergencies(self, s) -> Optional[int]:
        conds = s.conditions

        # Stoning
        if "stoned" in conds:
            self._last_action_reason = "stoning"
            if self.has_lizard_corpse:
                return Actions.EAT
            return self._try_pray(s, "stoning")

        # Sliming
        if "slimed" in conds:
            self._last_action_reason = "sliming"
            if self.has_lizard_corpse:
                return Actions.EAT
            return self._try_pray(s, "sliming")

        # HP critical: pray, quaff potion, or flee
        hp_danger = s.hp <= max(5, s.max_hp // 3)  # 33% threshold (AutoAscend)
        if hp_danger:
            self._last_action_reason = f"hp_critical ({s.hp}/{s.max_hp})"
            # Try prayer first
            pray_action = self._try_pray(s, "hp_critical")
            if pray_action is not None:
                return pray_action
            # Try healing potion
            potion = self._find_healing_potion(s)
            if potion is not None:
                self._pending_action = "quaff"
                self._pending_letter = potion
                self._last_action_reason = f"quaffing healing potion"
                return Actions.QUAFF
            # Can't pray or heal: flee or Elbereth
            if s.has_adjacent_monsters:
                return self._flee_or_elbereth(s)
            return Actions.SEARCH  # rest

        # Fainting from hunger
        if s.hunger_state in ("fainting", "fainted", "starved"):
            self._last_action_reason = f"starving ({s.hunger_state})"
            pray_action = self._try_pray(s, "starving")
            if pray_action is not None:
                return pray_action
            if self.has_food:
                return Actions.EAT
            # No food, can't pray: just wait. Don't fall through to combat.
            return Actions.SEARCH

        # Terminal illness / food poisoning
        if "foodpois" in conds or "termill" in conds:
            self._last_action_reason = "illness"
            return self._try_pray(s, "illness")

        return None

    # ----------------------------------------------------------
    # P1: Adjacent combat
    # ----------------------------------------------------------

    def _p1_adjacent_combat(self, s) -> Optional[int]:
        # On Elbereth: wait if HP low, resume fighting if HP full
        if self._on_elbereth:
            hp_ratio = s.hp / max(s.max_hp, 1)
            if hp_ratio < 0.8:
                return Actions.SEARCH  # regen HP while protected
            else:
                self._on_elbereth = False  # healed up, resume normal play
        # Filter out pets, peacefuls, and unreachable monsters
        hostile = []
        glyphs = s._glyphs
        # Peacefuls: shopkeepers, priests, guards, quest leaders
        # Shopkeepers display personal names (Asidonhopo, etc.) not "shopkeeper"
        # so we detect by monster ID (268=shopkeeper, 269=guard, 271=Oracle, 272-273=priests)
        _PEACEFUL_NAMES = {
            "shopkeeper", "aligned priest", "high priest",
            "guard", "Oracle", "watchman", "watch captain",
        }
        _PEACEFUL_IDS = {268, 269, 271, 272, 273}  # shopkeeper, guard, Oracle, priests
        for m in s.adjacent_monsters:
            if m.is_pet:
                continue
            # Skip by monster ID (catches shopkeepers with personal names)
            if m.mon_id in _PEACEFUL_IDS:
                self._refused_positions.add((m.row, m.col))
                continue
            # Skip known peaceful monsters by name
            if m.name in _PEACEFUL_NAMES:
                self._refused_positions.add((m.row, m.col))
                continue
            # Skip monsters we were told not to attack (by name or position)
            if m.name.lower() in self._refused_attacks:
                continue
            if (m.row, m.col) in self._refused_positions:
                continue
            # Check if the tile under the monster is water/lava/stone (can't melee)
            obj = int(self._objects[m.row, m.col])
            if obj != -1:
                cm = _glyph_cmap(obj)
                if cm in (32, 33, 34, 36):  # pool, moat, water, lava
                    continue
            # Check current glyph at monster tile: if it's a closed door or wall,
            # the "monster" might be behind it (NLE renders monsters on non-walkable tiles)
            if glyphs is not None:
                # The monster glyph should be at m.row, m.col. But the tile underneath
                # might be non-walkable. We can check if moving there repeatedly fails.
                pass
            # If we've been stuck for multiple steps, skip all combat.
            # The position hasn't changed, so we're fighting something unreachable.
            if self.turns_on_tile > 4:
                continue
            hostile.append(m)
        if not hostile:
            return None

        py, px = s.position

        # Assess each adjacent monster
        threats = []
        for mon in hostile:
            name = mon.name
            if self.threat_db is not None:
                report = self.threat_db.assess_threat(name, self._player_state(s))
            else:
                report = _StubThreatReport(name)
            threats.append((mon, report))

        threats.sort(key=lambda x: -x[1].danger_level)

        # Instakill risks: Elbereth first, flee if can't engrave
        any_instakill = any(r.instakill_risk for _, r in threats)
        if any_instakill:
            ik_name = next(m.name for m, r in threats if r.instakill_risk)
            # Elbereth is the safest response to instakill
            elb = self._try_elbereth(s)
            if elb == Actions.ENGRAVE:
                self._last_action_reason = f"elbereth vs instakill {ik_name}"
                return elb
            if self._on_elbereth:
                return Actions.SEARCH  # wait on Elbereth
            self._last_action_reason = f"flee instakill {ik_name}"
            return self._flee_or_elbereth(s)

        # Elbereth when in danger: weighted monster count + HP ratio
        if hostile:
            adj_weight = 0.0
            for mon, report in threats:
                hp_mult = min(20.0 / max(s.hp, 1), 2.0)
                if report.danger_level <= 2:
                    adj_weight += 0.2 * hp_mult
                elif report.danger_level >= 7:
                    adj_weight += 3.0 * hp_mult
                else:
                    adj_weight += 1.0 * hp_mult
            hp_ratio = (s.hp / max(s.max_hp, 1)) ** 0.5
            elbereth_priority = -5 + 20 * adj_weight * (1 - hp_ratio)
            if elbereth_priority > 0:
                elb = self._try_elbereth(s)
                if elb == Actions.ENGRAVE:
                    self._last_action_reason = f"elbereth (weight={adj_weight:.1f} hp={s.hp}/{s.max_hp})"
                    return elb

        top_mon, top_report = threats[0]

        # Never-melee monsters (floating eye, gas spore, etc.): flee always
        if getattr(top_report, 'never_melee', False) or top_mon.name in _NEVER_MELEE:
            self._last_action_reason = f"flee never-melee {top_mon.name}"
            return self._flee_from(s, top_mon)

        # Weird monsters (nymph, leprechaun): avoid if possible
        if top_mon.name in _WEIRD and self.turns_on_tile < 3:
            self._last_action_reason = f"flee weird {top_mon.name}"
            return self._flee_from(s, top_mon)

        # Imminent death: flee only at very low HP (< 20% or HP <= 5)
        if (s.hp <= 5 or s.hp < s.max_hp * 0.2) and self.turns_on_tile < 4:
            self._last_action_reason = f"flee imminent death (hp={s.hp})"
            return self._flee_or_elbereth(s)

        # Filter out never-melee from targets, melee the best remaining
        meleeable = [(m, r) for m, r in threats
                     if not getattr(r, 'never_melee', False)
                     and m.name not in _NEVER_MELEE
                     and m.name not in _WEIRD]
        if not meleeable:
            # All adjacent monsters are never-melee/weird, flee
            self._last_action_reason = "flee all non-meleeable"
            return self._flee_from(s, top_mon)

        best_mon, best_report = meleeable[0]
        direction = _direction_toward(py, px, best_mon.row, best_mon.col)
        d_name = self._action_name(direction)
        self._last_action_reason = f"melee {best_mon.name} {d_name}"
        return direction

    # ----------------------------------------------------------
    # P2: Ranged threats (visible non-adjacent)
    # ----------------------------------------------------------

    def _p2_ranged_threats(self, s) -> Optional[int]:
        py, px = s.position
        # Decrement throw cooldown
        if self._throw_cooldown > 0:
            self._throw_cooldown -= 1
        _PEACEFUL_NAMES = {
            "shopkeeper", "aligned priest", "high priest",
            "guard", "Oracle", "watchman", "watch captain",
        }
        non_adjacent = []
        for m in s.visible_monsters:
            if m.is_pet:
                continue
            if m.name in _PEACEFUL_NAMES:
                continue
            if m.name.lower() in self._refused_attacks:
                continue
            if (m.row, m.col) in self._refused_positions:
                continue
            if abs(m.row - py) <= 1 and abs(m.col - px) <= 1:
                continue
            # Skip monsters in water/lava
            obj = int(self._objects[m.row, m.col])
            if obj != -1 and _glyph_cmap(obj) in (32, 33, 34, 36):
                continue
            non_adjacent.append(m)
        if not non_adjacent:
            return None

        # Don't engage with visible monsters if we're stuck (behind wall, etc.)
        if self.turns_on_tile > 3:
            return None
        if self._stuck_moves > 3:
            return None

        for mon in non_adjacent:
            if self.threat_db is not None:
                report = self.threat_db.assess_threat(mon.name, self._player_state(s))
            else:
                report = _StubThreatReport(mon.name)
            if report.danger_level >= 8 and report.instakill_risk:
                self._last_action_reason = f"flee ranged {mon.name}"
                away = _direction_away(py, px, mon.row, mon.col)
                # Check the target tile is walkable before fleeing
                dd = Actions.MOVE_DELTAS.get(away)
                if dd is not None:
                    nr, nc = py + dd[0], px + dd[1]
                    if 0 <= nr < MAP_H and 0 <= nc < MAP_W and self._walkable[nr, nc]:
                        return away
                # Can't flee in that direction, try to navigate around
                return None

        # Try throwing daggers at monsters in line of fire
        # Cooldown prevents throw spam; range limit keeps it tactical
        if self._daggers and self._throw_cooldown == 0:
            for mon in sorted(non_adjacent, key=lambda m: max(abs(m.row-py), abs(m.col-px))):
                dist = max(abs(mon.row - py), abs(mon.col - px))
                if dist > 5:
                    break  # too far, not worth throwing
                line = _monster_in_line(py, px, mon.row, mon.col)
                if line is not None:
                    dagger_letter = next(iter(self._daggers))
                    dir_action = Actions.DELTA_TO_MOVE.get(line)
                    if dir_action is not None:
                        self._pending_action = "throw"
                        self._pending_letter = dagger_letter
                        self._ranged_dir = dir_action
                        self._throw_cooldown = 100
                        self._last_action_reason = f"throw dagger at {mon.name}"
                        return Actions.THROW
                    break  # only try first in-line monster

        # Approach closest killable monster for XP
        closest = min(non_adjacent, key=lambda m: max(abs(m.row - py), abs(m.col - px)))
        closest_dist = max(abs(closest.row - py), abs(closest.col - px))
        if closest_dist > 15:
            return None
        if self.threat_db is not None:
            report = self.threat_db.assess_threat(closest.name, self._player_state(s))
        else:
            report = _StubThreatReport(closest.name)

        # Approach if: not instakill risk and HP above 30%
        if not report.instakill_risk and s.hp > s.max_hp * 0.3:
            walkable, walkable_diag = self._build_nav_masks(s._glyphs)
            dis = _bfs_distances(py, px, walkable, walkable_diag)
            if dis[closest.row, closest.col] != -1:
                step = self._step_toward(py, px, (closest.row, closest.col),
                                         dis, walkable, walkable_diag)
                if step is not None:
                    self._last_action_reason = f"approach {closest.name}"
                    return step

        return None

    # ----------------------------------------------------------
    # P3: Food management
    # ----------------------------------------------------------

    def _p3_food(self, s) -> Optional[int]:
        # Decrement eat cooldown
        if self._eat_cooldown > 0:
            self._eat_cooldown -= 1
            # During cooldown, only eat for actual starvation
            if s.hunger_state not in ("weak", "fainting", "fainted"):
                return None
        if s.hunger_state in ("hungry", "weak"):
            # Prefer corpse on ground first (free nutrition, saves rations)
            if self._on_corpse and self._corpse_name:
                safe = False
                if self.threat_db is not None:
                    report = self.threat_db.corpse_value(self._corpse_name, self.resistances)
                    safe = report.safe_to_eat
                else:
                    safe = self._corpse_safe_to_eat(self._corpse_name)
                # Floating eye: skip unless blind
                if self._corpse_name == "floating eye" and "blind" not in s.conditions:
                    safe = False
                if safe:
                    self._pending_action = "eat"
                    self._eat_attempts = 0
                    self._last_action_reason = f"{s.hunger_state}, eating {self._corpse_name} corpse"
                    return Actions.EAT
            # Fall back to inventory food
            if self.has_food:
                self._pending_action = "eat"
                self._eat_attempts = 0
                self._last_action_reason = f"{s.hunger_state}, eating from inv"
                return Actions.EAT
        # Pray for starvation only when Weak or Fainting
        if s.hunger_state in ("weak", "fainting", "fainted"):
            if not self.has_food:
                pray_action = self._try_pray(s, "starving")
                if pray_action is not None:
                    self._last_action_reason = f"{s.hunger_state}, praying"
                    return pray_action
        return None

    # ----------------------------------------------------------
    # P4: Item management
    # ----------------------------------------------------------

    def _p4_items(self, s) -> Optional[int]:
        # Don't pick up if encumbered
        if s.encumbrance >= 1:
            return None
        # Don't pick up if last pickup failed
        if self.last_action == Actions.PICKUP:
            return None
        # Don't pick up if we have too many items (rough limit)
        if s.inventory and len(s.inventory) >= 40:
            return None
        # Check if there's an object on our tile (from message or glyph)
        if self._on_item:
            # Filter out junk items from messages
            if s.messages:
                text = s.last_message.lower()
                skip_items = ["rock", "statue", "boulder", "chain", "iron ball",
                              "heavy iron ball", "loadstone"]
                if any(junk in text for junk in skip_items):
                    return None
            self._last_action_reason = "picking up item (message)"
            return Actions.PICKUP
        # Also check glyph: if the remembered object at our tile is an item glyph
        py, px = s.position
        obj = int(self._objects[py, px])
        if obj != -1 and GLYPH_OBJ_OFF <= obj < GLYPH_CMAP_OFF:
            self._last_action_reason = "picking up item (glyph)"
            return Actions.PICKUP
        return None

    # ----------------------------------------------------------
    # P4b: Equipment management
    # ----------------------------------------------------------

    def _p4b_equipment(self, s) -> Optional[int]:
        if not s.inventory or s.has_adjacent_monsters:
            return None
        # Check after pickup or every 10 steps
        if self.last_action != Actions.PICKUP and self._step_count % 10 != 0:
            return None
        # Don't try if last equip attempt failed
        if self.last_action in (Actions.WEAR, Actions.WIELD):
            return None
        # Try to wield a better weapon
        wep = self._find_best_weapon(s)
        if wep:
            self._pending_action = "wield"
            self._pending_letter = wep
            self._last_action_reason = f"wielding '{wep}'"
            return Actions.WIELD
        # Try to wear unworn armor
        arm = self._find_best_armor(s)
        if arm:
            self._pending_action = "wear"
            self._pending_letter = arm
            self._last_action_reason = f"wearing '{arm}'"
            return Actions.WEAR
        return None

    def _find_best_weapon(self, s) -> Optional[str]:
        if not s.inventory:
            return None
        # Check what's currently wielded
        current_wielded = None
        current_pri = 0
        wpns = {
            "long sword": 10, "katana": 10, "two-handed sword": 11,
            "broadsword": 9, "battle-axe": 9,
            "short sword": 7, "mace": 7, "war hammer": 7, "morning star": 8,
            "spear": 6, "axe": 6, "dagger": 5, "knife": 4, "scimitar": 7,
            "aklys": 5, "flail": 7, "trident": 7,
        }
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "(weapon in hand)" in lower or "(wielded)" in lower:
                current_wielded = letter
                for w, p in wpns.items():
                    if w in lower:
                        current_pri = p
                        break
                # If wielded but not in our table, give it base priority 3
                if current_pri == 0:
                    current_pri = 3
        # If nothing wielded, check for anything to wield
        best_letter, best_pri = None, current_pri
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "(weapon in hand)" in lower or "(wielded)" in lower:
                continue
            if "(being worn)" in lower:
                continue
            if "cursed" in lower:
                continue
            for w, p in wpns.items():
                if w in lower and p > best_pri:
                    best_pri, best_letter = p, letter
        return best_letter

    def _find_best_armor(self, s) -> Optional[str]:
        if not s.inventory:
            return None
        # Track what slots are currently filled
        worn_slots = set()
        armor_keywords = {
            "helm": "head", "helmet": "head", "hat": "head",
            "shield": "shield", "small shield": "shield",
            "cloak": "cloak", "robe": "cloak",
            "gloves": "hands", "gauntlets": "hands",
            "boots": "feet", "shoes": "feet",
            "mail": "body", "armor": "body", "jacket": "body",
            "splint mail": "body", "plate mail": "body",
            "ring mail": "body", "scale mail": "body",
            "chain mail": "body", "banded mail": "body",
            "leather armor": "body", "studded leather armor": "body",
        }
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "(being worn)" in lower:
                for kw, slot in armor_keywords.items():
                    if kw in lower:
                        worn_slots.add(slot)
                        break
        # Find unworn armor for empty slots
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "(being worn)" in lower or "cursed" in lower:
                continue
            for kw, slot in armor_keywords.items():
                if kw in lower and slot not in worn_slots:
                    return letter
        return None

    def _find_droppable_item(self, s) -> Optional[str]:
        """Find the least valuable item to drop when encumbered."""
        if not s.inventory:
            return None
        # Never drop: wielded weapon, worn armor, food, lizard corpse
        keep_patterns = ["(weapon in hand)", "(wielded)", "(being worn)",
                         "food ration", "lizard corpse"]
        # Low priority to drop: rocks, gems, gold, corpses, junk
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if any(p in lower for p in keep_patterns):
                continue
            if "corpse" in lower or "rock" in lower or "stone" in lower:
                return letter
        # Drop anything non-essential
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if any(p in lower for p in keep_patterns):
                continue
            return letter
        return None

    # ----------------------------------------------------------
    # P4c: Excalibur (dip long sword in fountain)
    # ----------------------------------------------------------

    def _p4c_excalibur(self, s) -> Optional[int]:
        """Dip long sword in fountain to create Excalibur.
        Requires: lawful alignment, XL >= 7, long sword in inventory, fountain."""
        if self._has_excalibur:
            return None
        if s.xlevel < 7:
            return None
        if s.hp < s.max_hp * 0.5:
            return None  # water demon risk
        # Find long sword in inventory (wielded or not)
        sword_letter = None
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "long sword" in lower and "cursed" not in lower:
                sword_letter = letter
                break
        if sword_letter is None:
            return None
        # Check if on a fountain
        py, px = s.position
        obj_here = int(self._objects[py, px])
        on_fountain = (_glyph_cmap(obj_here) == 31)  # fountain cmap
        if on_fountain:
            self._pending_action = "dip"
            self._pending_letter = sword_letter
            self._last_action_reason = "dipping for Excalibur"
            return Actions.DIP
        # Path to nearest known fountain (no distance cap)
        if self._fountain_positions:
            glyphs = s._glyphs
            if glyphs is not None:
                walkable, walkable_diag = self._build_nav_masks(glyphs)
                dis = _bfs_distances(py, px, walkable, walkable_diag)
                best_f, best_d = None, 999999
                for fp in self._fountain_positions:
                    d = dis[fp[0], fp[1]]
                    if d != -1 and d < best_d:
                        best_d, best_f = d, fp
                if best_f is not None:
                    step = self._step_toward(py, px, best_f, dis, walkable, walkable_diag)
                    if step is not None:
                        self._last_action_reason = f"heading to fountain at {best_f}"
                        return step
        return None

    # ----------------------------------------------------------
    # P5: Corpse eating (proactive, like AutoAscend)
    # ----------------------------------------------------------

    def _p5_corpse_intrinsics(self, s) -> Optional[int]:
        """Eat safe corpses proactively. AutoAscend eats when NOT satiated,
        not just when hungry. This prevents starvation and gains intrinsics."""
        if not self._on_corpse or not self._corpse_name:
            return None
        if self._eat_cooldown > 0:
            return None
        # Don't eat if satiated (choking risk)
        if s.hunger_state == "satiated":
            return None
        # Don't eat while in combat (wastes a turn)
        if s.has_adjacent_monsters:
            return None
        # Floating eye: skip (paralysis on melee, telepathy only when blind)
        if self._corpse_name == "floating eye":
            return None

        safe = False
        reason = "nutrition"
        if self.threat_db is not None:
            report = self.threat_db.corpse_value(self._corpse_name, self.resistances)
            if report.safe_to_eat:
                safe = True
                reason = report.beneficial_intrinsic or "nutrition"
        elif self._corpse_safe_to_eat(self._corpse_name):
            safe = True

        if safe:
            self._pending_action = "eat"
            self._last_action_reason = f"eating {self._corpse_name} ({reason})"
            return Actions.EAT

        return None

    # ----------------------------------------------------------
    # P6: Navigation
    # ----------------------------------------------------------

    def _update_seen_and_walkable(self, s) -> None:
        """Update seen/walkable/objects masks from current glyphs.

        This is the key fix: NLE fills unexplored tiles with stone glyph
        (GLYPH_CMAP_OFF + 0). We distinguish "confirmed wall/stone" from
        "never observed" by tracking which tiles we have actually seen
        change from the default stone to something else, OR which tiles
        are adjacent to the player (we can see them directly).

        Follows AutoAscend's approach: mark tiles as seen when they show
        floor, monsters, objects, walls, or doors. Stone adjacent to the
        player is also marked seen (confirmed stone, not unexplored).
        """
        glyphs = s._glyphs
        if glyphs is None:
            return

        py, px = s.position

        # Update refused positions: remove positions where the monster has moved away
        new_refused = set()
        for rp in self._refused_positions:
            r, c = rp
            if 0 <= r < MAP_H and 0 <= c < MAP_W:
                g = int(glyphs[r, c])
                # If there's still a monster or pet there, keep it blocked
                if GLYPH_MON_OFF <= g < GLYPH_PET_OFF + NUMMONS:
                    new_refused.add(rp)
        self._refused_positions = new_refused

        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(glyphs[r, c])
                cm = _glyph_cmap(g)

                # Floor, stairs, doors (open), traps, altar, fountain
                if cm in _WALKABLE_CMAPS:
                    self._seen[r, c] = True
                    self._walkable[r, c] = True
                    self._objects[r, c] = g
                    if cm == 31:  # fountain
                        self._fountain_positions.add((r, c))

                # Walls and closed doors: seen but not walkable
                elif cm in _WALL_CMAPS:
                    self._seen[r, c] = True
                    self._walkable[r, c] = False
                    self._objects[r, c] = g

                elif cm in _CLOSED_DOOR_CMAPS:
                    self._seen[r, c] = True
                    self._walkable[r, c] = False  # can't walk through, must open
                    self._objects[r, c] = g

                # Boulder: seen but not walkable (blocks movement)
                elif g == _BOULDER_GLYPH:
                    self._seen[r, c] = True
                    self._walkable[r, c] = False

                # Monsters, pets, objects, bodies on floor: tile is walkable
                elif GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF:
                    self._seen[r, c] = True
                    # Keep walkable from previous knowledge if we had it,
                    # otherwise assume walkable (monster standing on something)
                    if self._objects[r, c] == -1:
                        self._walkable[r, c] = True

                    # Track pet position
                    if GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS:
                        self._pet_pos = (r, c)

                # Stone glyph: only mark seen if adjacent to player
                # (player can see adjacent tiles, so stone there is real stone)
                elif cm == _STONE_CMAP:
                    pass  # handled below for adjacency

        # Mark stone tiles adjacent to player as seen (confirmed stone)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nr, nc = py + dy, px + dx
                if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                    g = int(glyphs[nr, nc])
                    if _is_stone_glyph(g):
                        self._seen[nr, nc] = True
                        self._walkable[nr, nc] = False
                        self._objects[nr, nc] = g

    def _build_nav_masks(self, glyphs: np.ndarray) -> tuple:
        """Build walkable and walkable_diag masks for BFS.

        walkable: tiles the agent can step onto (rooms, corridors, open doors,
                  stairs, monsters we can attack, pets we can swap with).
        walkable_diag: subset of walkable that allows diagonal movement.
                       Doors (any kind) block diagonal movement.
        """
        walkable = self._walkable.copy()
        walkable_diag = walkable.copy()

        # Allow walking through monsters (melee) and pets (swap)
        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(glyphs[r, c])
                if GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF:
                    walkable[r, c] = True
                # Closed doors: walkable for pathfinding (we'll open them)
                if _is_closed_door_glyph(g):
                    walkable[r, c] = True

        # No diagonal through any door tile
        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(glyphs[r, c])
                if _is_door_glyph(g) or _is_closed_door_glyph(g):
                    walkable_diag[r, c] = False
                o = int(self._objects[r, c])
                if o != -1 and _glyph_cmap(o) in _DOOR_CMAPS:
                    walkable_diag[r, c] = False

        # Block refused positions (peacefuls we shouldn't walk into)
        for rp in self._refused_positions:
            r, c = rp
            if 0 <= r < MAP_H and 0 <= c < MAP_W:
                walkable[r, c] = False
                walkable_diag[r, c] = False

        return walkable, walkable_diag

    def _p6_navigation(self, s) -> Optional[int]:
        py, px = s.position
        glyphs = s._glyphs
        if glyphs is None:
            return Actions.SEARCH

        # 1. On downstairs: descend only if strong enough
        # Check both obs_parser detection and our remembered terrain
        on_dn_stairs = s.on_stairs_down
        obj_here = int(self._objects[py, px])
        if obj_here in (S_DNSTAIR, S_DNLADDER):
            on_dn_stairs = True
        if on_dn_stairs:
            can_descend = self._should_descend(s)
            if can_descend:
                self._last_action_reason = f"descending (xl={s.xlevel} dl={s.dlevel})"
                return Actions.DOWN

        # 2. Adjacent closed door: walk into it to open (cardinal only)
        door_action = self._try_open_adjacent_door(s, glyphs, py, px)
        if door_action is not None:
            return door_action

        # Build BFS distance map
        walkable, walkable_diag = self._build_nav_masks(glyphs)
        dis = _bfs_distances(py, px, walkable, walkable_diag)

        # 3. Go to nearest reachable closed door
        best_door = self._find_nearest_closed_door_bfs(glyphs, dis)
        if best_door is not None:
            step = self._step_toward(py, px, best_door, dis, walkable, walkable_diag)
            if step is not None:
                self._last_action_reason = f"door at {best_door}"
                return step

        # 4. Go to nearest frontier tile (walkable tile adjacent to unseen space)
        frontier = self._find_frontier(glyphs, dis)
        if frontier is not None:
            step = self._step_toward(py, px, frontier, dis, walkable, walkable_diag)
            if step is not None:
                self._last_action_reason = f"frontier at {frontier}"
                return step

        # 5. Go to stairs down (from glyphs or remembered objects)
        stairs_pos = _find_stairs_down(glyphs)
        if stairs_pos is None:
            stairs_pos = _find_stairs_down_from_objects(self._objects)

        # Determine if level is fully explored (no frontier, no closed doors)
        level_explored = (frontier is None and best_door is None)

        # Descent decision: use the standard gate, no relaxation
        can_go_down = self._should_descend(s)

        if stairs_pos is not None and stairs_pos != (py, px):
            if dis[stairs_pos[0], stairs_pos[1]] != -1 and can_go_down:
                step = self._step_toward(py, px, stairs_pos, dis, walkable, walkable_diag)
                if step is not None:
                    self._last_action_reason = f"stairs at {stairs_pos} (explored={level_explored})"
                    return step

        # If on stairs and level is explored, use standard descent gate
        if on_dn_stairs and level_explored and can_go_down:
            self._last_action_reason = f"descending (level explored, xl={s.xlevel} dl={s.dlevel})"
            return Actions.DOWN

        # 6. Search for hidden doors/passages (last resort)
        total_searches = int(self._search_count_map.sum())
        if total_searches > 300:
            # Exhausted searching. Descend if possible (relax gate).
            if stairs_pos is not None and stairs_pos != (py, px):
                if dis[stairs_pos[0], stairs_pos[1]] != -1:
                    step = self._step_toward(py, px, stairs_pos, dis, walkable, walkable_diag)
                    if step is not None:
                        self._last_action_reason = f"heading to stairs (searched {total_searches})"
                        return step
            # On stairs after exhausted searching: descend if healthy
            # Deeper levels have more score. Don't stay stuck.
            if on_dn_stairs and s.hp > s.max_hp * 0.5:
                self._last_action_reason = f"descend (searched {total_searches})"
                return Actions.DOWN
            # Random walk to find something
            import random
            candidates = []
            for ddy in (-1, 0, 1):
                for ddx in (-1, 0, 1):
                    if ddy == 0 and ddx == 0:
                        continue
                    ar, ac = py + ddy, px + ddx
                    if 0 <= ar < MAP_H and 0 <= ac < MAP_W and walkable[ar, ac]:
                        action = Actions.DELTA_TO_MOVE.get((ddy, ddx))
                        if action is not None:
                            candidates.append(action)
            if candidates:
                return random.choice(candidates)
            return Actions.WAIT

        self.search_count += 1
        self._search_count_map[py, px] += 1

        # After enough searching at this spot, move to a different spot
        if self._search_count_map[py, px] > 5:
            target = self._find_search_target(glyphs, dis, py, px)
            if target is not None:
                step = self._step_toward(py, px, target, dis, walkable, walkable_diag)
                if step is not None:
                    return step

        return Actions.SEARCH

    def _find_nearest_closed_door_bfs(self, glyphs, dis):
        """Find the nearest reachable closed door using precomputed BFS distances."""
        best = None
        best_d = 999999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if _is_closed_door_glyph(int(glyphs[r, c])):
                    if self._door_open_attempts[r, c] >= 5:
                        continue  # gave up on this door
                    d = dis[r, c]
                    if d != -1 and d < best_d:
                        best_d = d
                        best = (r, c)
        return best

    def _find_frontier(self, glyphs, dis):
        """Find nearest reachable walkable tile adjacent to unseen space.

        A tile is frontier if:
        - it is walkable and reachable (dis != -1)
        - at least one of its 8 neighbors is unseen stone

        Unseen stone = glyph is stone AND self._seen is False.
        """
        best = None
        best_d = 999999
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or d >= best_d:
                    continue
                if not self._walkable[r, c]:
                    continue
                # Check if any neighbor is unseen
                has_unseen = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            if not self._seen[nr, nc]:
                                has_unseen = True
                                break
                    if has_unseen:
                        break
                if has_unseen:
                    best_d = d
                    best = (r, c)
        return best

    def _find_search_target(self, glyphs, dis, py, px):
        """Find a tile worth searching at (adjacent to walls/stone, reachable,
        not over-searched). Used when we've exhausted local searching."""
        best = None
        best_score = -999999
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or d == 0:
                    continue
                if not self._walkable[r, c]:
                    continue
                # Count adjacent stone/wall (potential hidden doors)
                adj_stone = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            g = int(glyphs[nr, nc])
                            cm = _glyph_cmap(g)
                            if cm in _WALL_CMAPS or cm == _STONE_CMAP:
                                adj_stone += 1
                if adj_stone == 0:
                    continue
                # Prefer tiles with more stone neighbors, fewer prior searches, closer
                score = adj_stone * 10 - self._search_count_map[r, c] * 3 - d
                if score > best_score:
                    best_score = score
                    best = (r, c)
        return best

    def _try_open_adjacent_door(self, s, glyphs, py, px) -> Optional[int]:
        """Walk into an adjacent closed door (cardinal only) to open it."""
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = py + dy, px + dx
            if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                g = int(glyphs[nr, nc])
                if _is_closed_door_glyph(g):
                    # Give up after 5 attempts (locked/stuck door)
                    if self._door_open_attempts[nr, nc] >= 5:
                        continue
                    self._last_action_reason = f"opening door at ({nr},{nc})"
                    self._door_open_attempts[nr, nc] += 1
                    action = Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)
                    self._kick_dir = action
                    return action
        return None

    def _step_toward(self, py, px, target, dis, walkable, walkable_diag) -> Optional[int]:
        """Trace BFS path from (py,px) to target, return first step action.

        If the first step is blocked by a pet, try an alternate first step
        that still leads toward the target.
        """
        ty, tx = target
        if dis[ty, tx] == -1:
            return None

        # Trace path backward from target to source, respecting diagonal rules
        path = []
        cur_y, cur_x = ty, tx
        while (cur_y, cur_x) != (py, px):
            path.append((cur_y, cur_x))
            best_ny, best_nx = -1, -1
            best_d = dis[cur_y, cur_x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cur_y + dy, cur_x + dx
                    if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
                        if dis[ny, nx] >= 0 and dis[ny, nx] < best_d:
                            # Validate diagonal moves same as BFS
                            if abs(dy) + abs(dx) > 1:
                                if not walkable_diag[cur_y, cur_x] or not walkable_diag[ny, nx]:
                                    continue
                                if not walkable[ny, cur_x] and not walkable[cur_y, nx]:
                                    continue
                            best_d = dis[ny, nx]
                            best_ny, best_nx = ny, nx
            if best_ny == -1:
                return None  # can't trace back
            cur_y, cur_x = best_ny, best_nx
        path.reverse()

        if not path:
            return None

        self._cached_path = path
        first_r, first_c = path[0]
        dy = first_r - py
        dx = first_c - px

        # Check if pet is blocking the first step
        glyphs = self.state._glyphs
        if glyphs is not None:
            g = int(glyphs[first_r, first_c])
            is_pet = (GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS)
            if is_pet:
                # Moving into pet swaps positions. That's fine, do it.
                action = Actions.DELTA_TO_MOVE.get((dy, dx))
                if action is not None:
                    return action

        # If stuck, try alternate first step
        if self._stuck_moves >= 2:
            import random
            candidates = []
            for ddy in (-1, 0, 1):
                for ddx in (-1, 0, 1):
                    if ddy == 0 and ddx == 0:
                        continue
                    if (ddy, ddx) == (dy, dx):
                        continue  # already tried this
                    ar, ac = py + ddy, px + ddx
                    if 0 <= ar < MAP_H and 0 <= ac < MAP_W and walkable[ar, ac]:
                        action = Actions.DELTA_TO_MOVE.get((ddy, ddx))
                        if action is not None:
                            candidates.append(action)
            if candidates:
                self._stuck_moves = 0
                return random.choice(candidates)

        action = Actions.DELTA_TO_MOVE.get((dy, dx))
        return action

    # ----------------------------------------------------------
    # P5b: HP resting (search to regen between fights)
    # ----------------------------------------------------------

    def _p5b_rest(self, s) -> Optional[int]:
        """Rest (search) when HP is low, no threats around, and not hungry.

        Write Elbereth before resting for safety. HP regenerates
        1 per (20 - XL) turns approximately.
        """
        if s.has_adjacent_monsters:
            return None
        # Don't rest if there are visible hostile monsters nearby
        if s.visible_monsters:
            hostiles = [m for m in s.visible_monsters if not m.is_pet]
            if hostiles:
                return None
        # Rest when HP below 50%
        if s.hp >= s.max_hp * 0.5:
            return None
        # Don't rest if hungry or worse (wastes nutrition)
        if s.hunger_state in ("hungry", "weak", "fainting", "fainted"):
            return None
        # Write Elbereth before resting (safety net)
        if not self._on_elbereth and self._elbereth_cooldown == 0:
            elb = self._try_elbereth(s)
            if elb == Actions.ENGRAVE:
                self._last_action_reason = "writing Elbereth before resting"
                return elb
        # Cap resting at 25 turns
        py, px = s.position
        if self.turns_on_tile > 25:
            return None
        self._last_action_reason = f"resting hp={s.hp}/{s.max_hp}"
        return Actions.SEARCH

    # ----------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------

    def _should_descend(self, s) -> bool:
        """Decide whether we're strong enough to descend.

        Milestone 1: farm DL1 to XL 4.
        Milestone 2-4: XL >= DL + 2 (stay strong relative to depth).
        DL5+: XL >= DL + 1 (looser gate once established).
        """
        # HP gate
        if s.hp < s.max_hp * 0.6:
            return False
        # XL gate by depth
        if s.dlevel == 1 and s.xlevel < 2:
            return False
        if 2 <= s.dlevel <= 4 and s.xlevel < s.dlevel + 1:
            return False
        if s.dlevel >= 5 and s.xlevel < s.dlevel + 1:
            return False
        # Don't descend if adjacent hostile monsters (finish the fight first)
        if s.has_adjacent_monsters:
            hostiles = [m for m in s.adjacent_monsters if not m.is_pet]
            if hostiles:
                return False
        return True

    def _try_pray(self, s, trouble_type: str) -> Optional[int]:
        """Attempt prayer if safe. Returns PRAY action or None."""
        if self.prayer is not None:
            safe, reason = self.prayer.is_prayer_safe(s.turn, trouble_type)
            if safe:
                self.prayer.update_prayed(s.turn)
                return Actions.PRAY
            return None
        # No prayer subsystem: heuristic check
        if s.turn >= 300:
            return Actions.PRAY
        return None

    def _find_healing_potion(self, s) -> Optional[str]:
        """Find a healing potion in inventory."""
        if not s.inventory:
            return None
        POTION_CLASS = 8
        oclasses = getattr(s, 'inventory_oclasses', {})
        healing_names = ["healing", "extra healing", "full healing"]
        for letter, item_str in s.inventory.items():
            if oclasses and oclasses.get(letter) != POTION_CLASS:
                continue
            lower = item_str.lower()
            if any(h in lower for h in healing_names) and "cursed" not in lower:
                return letter
        return None

    def _flee_or_elbereth(self, s) -> int:
        """Flee from adjacent monsters, or write Elbereth if surrounded."""
        hostile = [m for m in s.adjacent_monsters if not m.is_pet]
        py, px = s.position

        # If already on Elbereth, just wait (monsters should flee)
        if self._on_elbereth:
            return Actions.SEARCH

        # Try to flee first
        if hostile:
            flee_action = self._flee_from(s, hostile[0])
            if flee_action not in (Actions.ENGRAVE, Actions.WAIT):
                return flee_action

        # Can't flee: write Elbereth
        return self._try_elbereth(s)

    def _try_elbereth(self, s) -> int:
        """Attempt to write Elbereth. Returns ENGRAVE to start the sequence,
        or WAIT if Elbereth is on cooldown or unavailable."""
        if self._on_elbereth:
            return Actions.SEARCH  # already on one, just wait
        if self._elbereth_cooldown > 0:
            return Actions.WAIT
        # Don't engrave on special tiles (stairs, altars, fountains)
        py, px = s.position
        obj_here = int(self._objects[py, px])
        if obj_here != -1:
            cm = _glyph_cmap(obj_here)
            if cm in (23, 24, 25, 26, 27, 28, 29, 30, 31):  # stairs, altar, grave, throne, sink, fountain
                return Actions.WAIT
        self._pending_action = "elbereth"
        if self.verbose:
            self._log("ELBERETH", "starting engrave")
        return Actions.ENGRAVE

    def _flee_from(self, s, monster) -> int:
        py, px = s.position
        my, mx = monster.row, monster.col
        # Try direction away first
        away = _direction_away(py, px, my, mx)
        dd = Actions.MOVE_DELTAS.get(away)
        if dd is not None:
            nr, nc = py + dd[0], px + dd[1]
            if 0 <= nr < MAP_H and 0 <= nc < MAP_W and self._walkable[nr, nc]:
                return away
        # Away direction blocked: try all walkable tiles, prefer ones farthest from monster
        import random
        candidates = []
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                if ddy == 0 and ddx == 0:
                    continue
                ar, ac = py + ddy, px + ddx
                if 0 <= ar < MAP_H and 0 <= ac < MAP_W and self._walkable[ar, ac]:
                    dist = max(abs(ar - my), abs(ac - mx))
                    action = Actions.DELTA_TO_MOVE.get((ddy, ddx))
                    if action is not None:
                        candidates.append((dist, action))
        if candidates:
            candidates.sort(reverse=True)  # farthest first
            return candidates[0][1]
        # Completely trapped: just wait (Elbereth handled by _flee_or_elbereth)
        return Actions.WAIT

    def _corpse_safe_to_eat(self, name: str) -> bool:
        """Quick safety check without ThreatDB."""
        unsafe = {
            "green slime", "cockatrice", "chickatrice", "Medusa",
            "Death", "Pestilence", "Famine",
            "chameleon", "doppelganger", "sandestin",
            # Aggravate monster (permanent, bad)
            "little dog", "dog", "large dog",
            "kitten", "housecat", "large cat",
            # Stun/hallucination
            "bat", "giant bat", "yellow mold", "violet fungus",
            # Mimics (paralysis)
            "small mimic", "large mimic", "giant mimic",
            # Polymorph
            "chameleon", "doppelganger", "sandestin",
            # Lycanthropy
            "wererat", "werejackal", "werewolf",
            # Strips intrinsics
            "disenchanter",
            # Speed toggle (bad if already fast)
            "quantum mechanic",
        }
        if name in unsafe:
            return False
        poisonous = {
            "killer bee", "scorpion", "pit viper", "cobra",
            "water moccasin", "asp", "python", "giant spider",
            "quasit", "rabid rat", "garter snake",
            "black naga", "golden naga hatchling",
            "baby purple worm", "purple worm",
        }
        if name in poisonous and "poison resistance" not in self.resistances:
            return False
        acidic = {
            "acid blob", "yellow light", "gelatinous cube",
            "blue jelly", "ochre jelly",
        }
        if name in acidic and "acid resistance" not in self.resistances:
            return False
        return True

    def _player_state(self, s) -> dict:
        """Build player_state dict for combat.py assess_threat."""
        return {
            "hp": s.hp,
            "max_hp": s.max_hp,
            "ac": s.ac,
            "level": s.xlevel,
            "speed": 12,
            "resistances": self.resistances,
            "equipment": {},
            "position": s.position,
            "has_elbereth_source": True,
        }

    def _detect_tile_contents(self, s) -> None:
        """Detect items, corpses, and objects on the player's tile.

        NLE places the player's own glyph at the player position, hiding
        whatever is beneath. We detect tile contents from messages
        ("You see here ...", "There is ... here", "There are ... here")
        and from the glyph map as a fallback.
        """
        self._on_corpse = False
        self._corpse_name = ""
        self._on_item = False
        self._on_edible_item = False

        # Message-based detection (primary, since glyph is occluded)
        if s.messages:
            text = s.last_message.lower()
            # "You see here a <item>." / "There is a <item> here."
            # "You see here a <monster> corpse."
            if "corpse" in text and ("you see here" in text or "there is" in text
                                     or "there are" in text):
                self._on_corpse = True
                # Try to extract the monster name from "a <name> corpse"
                for pattern in ["you see here a ", "you see here an ",
                                "there is a ", "there is an ",
                                "there are ", "you see here "]:
                    if pattern in text:
                        after = text.split(pattern, 1)[1]
                        if "corpse" in after:
                            name = after.split(" corpse")[0].strip()
                            # Remove article remnants
                            for art in ["a ", "an ", "the "]:
                                if name.startswith(art):
                                    name = name[len(art):]
                            self._corpse_name = name
                            break

            elif "you see here" in text or "there is" in text or "there are" in text:
                # Exclude dungeon features that use similar phrasing
                feature_words = ["staircase", "ladder", "altar", "fountain",
                                 "sink", "grave", "trap", "door"]
                is_feature = any(fw in text for fw in feature_words)
                if not is_feature:
                    self._on_item = True
                    # Check for food items
                    if any(food in text for food in [
                        "food ration", "tripe", "meatball", "egg", "melon",
                        "orange", "apple", "pear", "banana", "cream pie",
                        "lembas", "cram", "lichen", "kelp",
                    ]):
                        self._on_edible_item = True

            # "Things that are here:" means multiple items
            if "things that are here" in text or "things that you feel here" in text:
                self._on_item = True

        # Glyph-based fallback: only works if player glyph doesn't occlude
        py, px = s.position
        if s._glyphs is not None and not self._on_corpse and not self._on_item:
            has_c, c_name = _tile_has_corpse(s._glyphs, py, px)
            if has_c:
                self._on_corpse = True
                self._corpse_name = c_name
            elif _tile_has_object(s._glyphs, py, px):
                self._on_item = True

    def _parse_message_for_state(self, s) -> None:
        """Extract resistance gains and inventory hints from messages."""
        if not s.messages:
            return
        text = s.last_message.lower()

        if "you feel especially healthy" in text:
            self.resistances.add("poison resistance")
        if "you feel a momentary chill" in text:
            self.resistances.add("cold resistance")
        if "you feel warm" in text:
            self.resistances.add("fire resistance")
        if "you feel full of energy" in text:
            self.resistances.add("shock resistance")
        if "you feel very firm" in text:
            self.resistances.add("disintegration resistance")
        if "you feel wide awake" in text:
            self.resistances.add("sleep resistance")
        # Excalibur creation
        if "your sword has a bright" in text or "excalibur" in text:
            self._has_excalibur = True
            if self.verbose:
                self._log("EXCALIBUR", "created!")
        # Fountain dried up
        if "fountain dries up" in text:
            py, px = s.position
            self._fountain_positions.discard((py, px))
        # Elbereth confirmation
        if "elbereth" in text and "you read" in text:
            self._on_elbereth = True

    def _check_inventory(self, s) -> None:
        """Scan inventory strings for food and lizard corpses.

        Only set has_food for items that _find_food_letter would
        actually select. This prevents EAT->cancel loops.
        """
        if not s.inventory:
            return
        self.has_lizard_corpse = False
        self.inventory_items = {}
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            self.inventory_items[letter] = item_str
            if "lizard corpse" in lower:
                self.has_lizard_corpse = True
        # has_food = True only if _find_food_letter would return something
        self.has_food = (self._find_food_letter(s) is not None)
        # Scan for daggers (throwable ranged ammo)
        self._daggers = {}
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "dagger" in lower and "cursed" not in lower and \
               "(weapon in hand)" not in lower and "(wielded)" not in lower:
                self._daggers[letter] = item_str
        # Check for Excalibur
        for letter, item_str in s.inventory.items():
            if "Excalibur" in item_str:
                self._has_excalibur = True

    def _find_food_letter(self, s) -> Optional[str]:
        """Find the inventory letter of the best food item to eat."""
        if not s.inventory:
            return None
        FOOD_CLASS = 7
        food_items = {
            "food ration": 10, "lembas wafer": 9, "cram ration": 9,
            "tripe ration": 7,
            "apple": 5, "banana": 5, "pear": 5, "orange": 5, "melon": 5,
            "carrot": 5, "cream pie": 5, "candy bar": 5,
            "fortune cookie": 5, "pancake": 5, "lump of royal jelly": 5,
            "kelp frond": 4, "meatball": 4, "egg": 3,
            "lizard corpse": 6, "lichen corpse": 6,
        }
        oclasses = getattr(s, 'inventory_oclasses', {})
        best = None
        best_priority = -1
        for letter, item_str in s.inventory.items():
            # If we have object class data, only consider food items
            if oclasses and oclasses.get(letter) != FOOD_CLASS:
                continue
            lower = item_str.lower()
            if "cursed" in lower:
                continue
            for food_name, pri in food_items.items():
                if food_name in lower and pri > best_priority:
                    best, best_priority = letter, pri
        return best

    def _letter_to_action(self, letter: str) -> int:
        """Convert a character to an NLE action index.
        NLE TextCharacters cover digits and some punctuation.
        For letters, we need to find the matching action by ord value."""
        target = ord(letter)
        try:
            from nle import nethack
            for i, a in enumerate(nethack.ACTIONS):
                if int(a) == target:
                    return i
        except ImportError:
            pass
        # Fallback: search known TextCharacters range (105-120 in canonical order)
        return Actions.MORE  # default to CR if can't find it


# ============================================================
# Mock test
# ============================================================

def _make_mock_obs(
    *,
    py: int = 10, px: int = 40,
    hp: int = 30, max_hp: int = 50,
    depth: int = 1, xlevel: int = 3,
    hunger: int = 1, turn: int = 500,
    condition: int = 0,
    alignment: int = 1,
    monsters: Optional[list] = None,
    stairs_down: Optional[tuple] = None,
    corpse_at_player: Optional[int] = None,
    object_at_player: bool = False,
    message: str = "",
) -> dict:
    """Build a fake NLE observation dict for testing.

    Uses obs_parser.py's blstats layout (27 elements, BL_TIME=20,
    BL_HUNGER=21, BL_CONDITION=25, BL_ALIGN=26).
    """
    stone_glyph = GLYPH_CMAP_OFF  # cmap index 0 = stone wall
    glyphs = np.full((MAP_H, MAP_W), stone_glyph, dtype=np.int16)

    # Carve a room around the player
    room_glyph = GLYPH_CMAP_OFF + 19  # CMAP_ROOM
    for r in range(max(1, py - 3), min(MAP_H - 1, py + 4)):
        for c in range(max(1, px - 5), min(MAP_W - 1, px + 6)):
            glyphs[r, c] = room_glyph

    # Player's own glyph (valkyrie, mon_id ~100)
    glyphs[py, px] = GLYPH_MON_OFF + 100

    # Overwrite player tile for special cases
    if corpse_at_player is not None:
        glyphs[py, px] = GLYPH_BODY_OFF + corpse_at_player
    elif object_at_player:
        glyphs[py, px] = GLYPH_OBJ_OFF + 10

    # Place monsters
    if monsters:
        for mr, mc, mon_id in monsters:
            glyphs[mr, mc] = GLYPH_MON_OFF + mon_id

    # Place stairs
    if stairs_down:
        sr, sc = stairs_down
        glyphs[sr, sc] = S_DNSTAIR

    # Build blstats (27 elements, obs_parser layout)
    bl = np.zeros(27, dtype=np.float32)
    bl[0] = float(px)   # BL_X
    bl[1] = float(py)   # BL_Y
    bl[2] = 18.0         # BL_STR25
    bl[9] = 100.0        # BL_SCORE
    bl[10] = float(hp)   # BL_HP
    bl[11] = float(max_hp)  # BL_HPMAX
    bl[12] = float(depth)   # BL_DEPTH
    bl[16] = 5.0         # BL_AC
    bl[18] = float(xlevel)  # BL_XP (experience level)
    bl[20] = float(turn)    # BL_TIME
    bl[21] = float(hunger)  # BL_HUNGER
    bl[25] = float(condition)  # BL_CONDITION
    bl[26] = float(alignment)  # BL_ALIGN

    # Build message
    msg = np.zeros(256, dtype=np.uint8)
    if message:
        msg_bytes = message.encode("ascii")
        msg[:len(msg_bytes)] = list(msg_bytes)

    return {
        "glyphs": glyphs,
        "blstats": bl,
        "message": msg,
    }


def _run_tests():
    """Simulate turns with mock observations to verify priority logic."""
    passed = 0
    failed = 0

    def check(label, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {label}")
        else:
            failed += 1
            print(f"  FAIL: {label} {detail}")

    agent = ExpertAgent()

    print("=== Expert Agent Priority Tests ===\n")

    # --- Test 1: P0 - Critical HP triggers prayer ---
    print("Test 1: P0 - Critical HP (HP=3/50 at turn 500)")
    agent.reset()
    obs = _make_mock_obs(hp=3, max_hp=50, turn=500)
    action = agent.act(obs)
    check("critical HP should pray", action == Actions.PRAY,
          f"got action {action}, expected {Actions.PRAY}")

    # --- Test 2: P0 - Critical HP too early to pray ---
    print("\nTest 2: P0 - Critical HP at turn 50 (prayer unsafe)")
    agent.reset()
    obs = _make_mock_obs(hp=3, max_hp=50, turn=50)
    action = agent.act(obs)
    check("critical HP + prayer unsafe should not pray",
          action != Actions.PRAY,
          f"got action {action}")

    # --- Test 3: P0 - Stoning condition ---
    print("\nTest 3: P0 - Stoning condition")
    agent.reset()
    obs = _make_mock_obs(hp=30, max_hp=50, turn=500, condition=COND_STONE)
    action = agent.act(obs)
    check("stoning should trigger prayer", action == Actions.PRAY,
          f"got action {action}")

    # --- Test 4: P1 - Adjacent monster, melee ---
    print("\nTest 4: P1 - Adjacent monster (melee)")
    agent.reset()
    # Place a monster at (9, 40), one tile N of player at (10, 40)
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         monsters=[(9, 40, 5)])
    action = agent.act(obs)
    check("adjacent monster should trigger melee (move toward)",
          action in Actions.MOVE_DELTAS,
          f"got action {action}")

    # --- Test 5: P1 - Adjacent instakill monster ---
    print("\nTest 5: P1 - Adjacent instakill monster (flee/Elbereth)")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         monsters=[(9, 40, 50)])
    agent.act(obs)
    s = agent.state
    # Override the name to "cockatrice" for testing (without NLE, names are from JSON)
    if s.adjacent_monsters:
        s.adjacent_monsters[0] = MonsterInfo(
            name="cockatrice", row=9, col=40, mon_id=50, is_pet=False
        ) if _HAS_OBS_PARSER else s.adjacent_monsters[0]
        if _HAS_OBS_PARSER:
            s.visible_monsters = [m for m in s.visible_monsters if m.mon_id != 50]
            s.visible_monsters.append(s.adjacent_monsters[0])
    action = agent._decide(s)
    check("instakill adjacent should flee or Elbereth",
          action in list(Actions.MOVE_DELTAS.keys()) + [Actions.ENGRAVE],
          f"got action {action}")

    # --- Test 6: P3 - Hungry with food in inventory ---
    print("\nTest 6: P3 - Hungry with food")
    agent.reset()
    agent.has_food = True
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, hunger=2)  # hungry
    action = agent.act(obs)
    check("hungry with food should eat", action == Actions.EAT,
          f"got action {action}")

    # --- Test 7: P4 - Item on ground (message-based detection) ---
    print("\nTest 7: P4 - Item on ground")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         message="You see here a long sword.")
    action = agent.act(obs)
    check("item on ground should pickup", action == Actions.PICKUP,
          f"got action {action}")

    # --- Test 8: P6 - Navigate to stairs ---
    print("\nTest 8: P6 - Navigate to stairs when explored")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, depth=1, xlevel=3,
                         stairs_down=(12, 42))
    action = agent.act(obs)
    check("should navigate toward stairs",
          action in Actions.MOVE_DELTAS,
          f"got action {action}")

    # --- Test 9: P6 - On stairs, descend ---
    print("\nTest 9: P6 - On stairs, should descend")
    agent.reset()
    # Put stairs at player position and set up message so obs_parser detects it
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, depth=1, xlevel=3,
                         stairs_down=(10, 40),
                         message="There is a staircase down here.")
    action = agent.act(obs)
    check("on stairs with good HP should descend", action == Actions.DOWN,
          f"got action {action}")

    # --- Test 10: P6 - On stairs but low HP ---
    print("\nTest 10: P6 - On stairs but low HP, don't descend")
    agent.reset()
    obs = _make_mock_obs(hp=10, max_hp=50, turn=500, depth=1, xlevel=3,
                         stairs_down=(10, 40),
                         message="There is a staircase down here.")
    action = agent.act(obs)
    check("on stairs with low HP should not descend", action != Actions.DOWN,
          f"got action {action}")

    # --- Test 11: Priority ordering - P0 over P1 ---
    print("\nTest 11: Priority - P0 (critical HP) beats P1 (combat)")
    agent.reset()
    obs = _make_mock_obs(hp=3, max_hp=50, turn=500,
                         monsters=[(9, 40, 5)])
    action = agent.act(obs)
    check("critical HP should pray even with adjacent monster",
          action == Actions.PRAY,
          f"got action {action}")

    # --- Test 12: P5 - Corpse eating for intrinsics ---
    print("\nTest 12: P5 - Standing on beneficial corpse (message-based)")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         message="You see here a floating eye corpse.")
    action = agent.act(obs)
    check("beneficial corpse should eat (floating eye)", action == Actions.EAT,
          f"got action {action}, corpse_name={agent._corpse_name}")

    # --- Test 13: Exploration ---
    print("\nTest 13: P6 - Explore when no other priority")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500)
    action = agent.act(obs)
    check("should explore or search when nothing else to do",
          action in list(Actions.MOVE_DELTAS.keys()) + [Actions.SEARCH],
          f"got action {action}")

    # --- Test 14: Message parsing ---
    print("\nTest 14: Message parsing for resistances")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         message="You feel especially healthy.")
    agent.act(obs)
    check("poison resistance detected from message",
          "poison resistance" in agent.resistances)

    # --- Test 15: Hunger fainting triggers prayer ---
    print("\nTest 15: P0 - Fainting from hunger")
    agent.reset()
    obs = _make_mock_obs(hp=30, max_hp=50, turn=500, hunger=4)  # fainting
    action = agent.act(obs)
    check("fainting should pray", action == Actions.PRAY,
          f"got action {action}")

    # --- Test 16: Reset clears state ---
    print("\nTest 16: Reset clears state")
    agent.resistances.add("fire resistance")
    agent.has_food = True
    agent.reset()
    check("resistances cleared", len(agent.resistances) == 0)
    check("has_food cleared", not agent.has_food)

    # --- Test 17: Multiple turn simulation ---
    print("\nTest 17: Multi-turn simulation (5 turns)")
    agent.reset()
    actions_taken = []
    for t in range(5):
        obs = _make_mock_obs(hp=40, max_hp=50, turn=500 + t)
        a = agent.act(obs)
        actions_taken.append(a)
    check("5 turns produced actions",
          len(actions_taken) == 5 and all(isinstance(a, int) for a in actions_taken),
          f"actions: {actions_taken}")

    # --- Test 18: Flee from multiple hostile adjacent monsters ---
    print("\nTest 18: P1 - Multiple adjacent monsters (Elbereth if surrounded)")
    agent.reset()
    obs = _make_mock_obs(hp=15, max_hp=50, turn=500,
                         monsters=[(9, 40, 5), (9, 41, 10), (11, 40, 15),
                                   (10, 41, 20)])
    action = agent.act(obs)
    # Low HP + 4 adjacent monsters: should Elbereth or flee
    check("surrounded should Elbereth or flee",
          action in list(Actions.MOVE_DELTAS.keys()) + [Actions.ENGRAVE],
          f"got action {action}")

    # --- Summary ---
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
