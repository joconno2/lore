"""Expert system agent for NetHack.

Priority-based decision loop that ties together combat, prayer, item ID,
navigation, and observation parsing subsystems. Drop-in replacement for
a neural policy: takes NLE obs dict, returns action index.

Subsystem imports are fault-tolerant. Missing modules get stubbed so
the agent still runs (with degraded capability) during incremental
development.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Conditional imports: stub anything not yet built
# ============================================================

try:
    from nhc.combat import ThreatDB, ThreatReport, CorpseReport
except ImportError:
    ThreatDB = None  # type: ignore
    ThreatReport = None  # type: ignore
    CorpseReport = None  # type: ignore

try:
    from nhc.prayer import (
        PrayerState, HungerState, TroubleSeverity,
        COND_STONE, COND_SLIME, COND_STRNGL, COND_ILL,
    )
except ImportError:
    PrayerState = None  # type: ignore
    HungerState = None  # type: ignore
    TroubleSeverity = None  # type: ignore
    COND_STONE = 0x00000001
    COND_SLIME = 0x00000002
    COND_STRNGL = 0x00000004
    COND_ILL = 0x00001000

try:
    from nhc.item_id import AppearanceTracker
except ImportError:
    AppearanceTracker = None  # type: ignore

try:
    from nhc.obs_parser import GameState
except ImportError:
    GameState = None  # type: ignore

try:
    from nhc.navigation import DungeonMap
except ImportError:
    DungeonMap = None  # type: ignore


# ============================================================
# NLE action indices (from nle.nethack.ACTIONS canonical order)
# ============================================================
# The full NLE action space has 113 actions. These are the indices
# into that canonical tuple that the expert agent uses.

# Compass movement: indices 0-7
# nle.nethack.CompassDirection values in ACTIONS order:
#   N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7
# But the actual canonical ACTIONS tuple starts with CompassCardinalDirection
# then CompassIntercardinalDirection, then MiscDirection, then Commands.
# The standard ordering in nle.nethack.ACTIONS is:
#   0:N, 1:E, 2:S, 3:W, 4:NE, 5:SE, 6:SW, 7:NW,
#   8:up, 9:down, 10:wait,
#   11-onwards: commands
#
# We build a symbolic map and resolve at runtime if nle is available.

class Actions:
    """Static action index mapping for the canonical NLE action space.

    If nle is importable, indices are resolved from nle.nethack.ACTIONS.
    Otherwise, hardcoded defaults from NLE 0.9.x are used.
    """
    # Movement directions (CompassCardinal + CompassIntercardinal)
    N = 0; E = 1; S = 2; W = 3
    NE = 4; SE = 5; SW = 6; NW = 7
    # MiscDirection
    UP = 8; DOWN = 9; WAIT = 10
    # Commands (alphabetical in NLE, but we only need a subset)
    APPLY = 11
    CLOSE = 12
    DROP = 13
    EAT = 14
    ENGRAVE = 15
    FIRE = 16
    INV = 17
    KICK = 18
    LOOT = 19
    OPEN = 20
    PAY = 21
    PICKUP = 22
    PRAY = 23
    PUTON = 24
    QUAFF = 25
    READ = 26
    REMOVE = 27
    RIDE = 28
    SEARCH = 29
    TAKEOFF = 30
    THROW = 31
    WEAR = 32
    WIELD = 33
    ZAP = 34
    # Special
    MORE = 35
    # Total count (not all are used)
    NUM_ACTIONS = 113

    # Direction deltas: index -> (dy, dx) in NLE screen coords
    # NLE uses (row, col) where row increases downward
    MOVE_DELTAS = {
        0: (-1, 0),   # N
        1: (0, 1),    # E
        2: (1, 0),    # S
        3: (0, -1),   # W
        4: (-1, 1),   # NE
        5: (1, 1),    # SE
        6: (1, -1),   # SW
        7: (-1, -1),  # NW
    }

    # Reverse: (dy, dx) -> action index
    DELTA_TO_MOVE = {v: k for k, v in MOVE_DELTAS.items()}


def _try_resolve_actions():
    """Attempt to resolve action indices from NLE at import time.
    Silently falls back to hardcoded defaults if nle is not installed.
    """
    try:
        from nle import nethack
        act_list = list(nethack.ACTIONS)
        lookup = {}
        for i, a in enumerate(act_list):
            name = a.name if hasattr(a, "name") else str(a)
            lookup[name] = i

        # Map our symbolic names to resolved indices
        compass = {
            "N": "CompassDirection.N", "E": "CompassDirection.E",
            "S": "CompassDirection.S", "W": "CompassDirection.W",
            "NE": "CompassDirection.NE", "SE": "CompassDirection.SE",
            "SW": "CompassDirection.SW", "NW": "CompassDirection.NW",
        }
        for attr, nle_name in compass.items():
            if nle_name in lookup:
                setattr(Actions, attr, lookup[nle_name])

        misc = {"UP": "MiscDirection.UP", "DOWN": "MiscDirection.DOWN",
                "WAIT": "MiscDirection.WAIT"}
        for attr, nle_name in misc.items():
            if nle_name in lookup:
                setattr(Actions, attr, lookup[nle_name])

        cmds = {
            "APPLY": "Command.APPLY", "CLOSE": "Command.CLOSE",
            "DROP": "Command.DROP", "EAT": "Command.EAT",
            "ENGRAVE": "Command.ENGRAVE", "FIRE": "Command.FIRE",
            "KICK": "Command.KICK", "LOOT": "Command.LOOT",
            "OPEN": "Command.OPEN", "PICKUP": "Command.PICKUP",
            "PRAY": "Command.PRAY", "PUTON": "Command.PUTON",
            "QUAFF": "Command.QUAFF", "READ": "Command.READ",
            "SEARCH": "Command.SEARCH", "TAKEOFF": "Command.TAKEOFF",
            "THROW": "Command.THROW", "WEAR": "Command.WEAR",
            "WIELD": "Command.WIELD", "ZAP": "Command.ZAP",
            "MORE": "TextCharacters.MORE",
        }
        for attr, nle_name in cmds.items():
            if nle_name in lookup:
                setattr(Actions, attr, lookup[nle_name])

        Actions.NUM_ACTIONS = len(act_list)

        # Rebuild delta maps
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
# Blstats indices (canonical NLE order, matches nhc/kb.py)
# ============================================================

BL_X = 0
BL_Y = 1
BL_STR25 = 2
BL_STR125 = 3
BL_DEX = 4
BL_CON = 5
BL_INT = 6
BL_WIS = 7
BL_CHA = 8
BL_SCORE = 9
BL_HP = 10
BL_MAXHP = 11
BL_DEPTH = 12
BL_GOLD = 13
BL_ENE = 14
BL_MAXENE = 15
BL_AC = 16
BL_XP = 17
BL_HUNGER = 18
BL_CAP = 19
BL_DNUM = 20
BL_XL = 21
BL_ALIGNMENT = 22
BL_CONDITION = 23
BL_MONSTER = 24
BL_TIME = 25

# NLE glyph offsets (from nle.nethack constants)
try:
    from nle import nethack as _nh
    GLYPH_MON_OFF = _nh.GLYPH_MON_OFF
    GLYPH_PET_OFF = _nh.GLYPH_PET_OFF
    GLYPH_BODY_OFF = _nh.GLYPH_BODY_OFF
    GLYPH_OBJ_OFF = _nh.GLYPH_OBJ_OFF
    GLYPH_CMAP_OFF = _nh.GLYPH_CMAP_OFF
    GLYPH_RIDDEN_OFF = _nh.GLYPH_RIDDEN_OFF
    MAX_GLYPH = _nh.MAX_GLYPH
    _NLE_AVAILABLE = True
except ImportError:
    # Fallback values from NLE 0.9.x
    GLYPH_MON_OFF = 0
    GLYPH_PET_OFF = 381
    GLYPH_BODY_OFF = 1524
    GLYPH_OBJ_OFF = 1905
    GLYPH_CMAP_OFF = 2840
    GLYPH_RIDDEN_OFF = 1143
    MAX_GLYPH = 5976
    _NLE_AVAILABLE = False

# Dungeon tile glyphs (cmap indices for navigation)
# floor, corridor, open door, stairs down, stairs up
CMAP_FLOOR = 23   # room floor
CMAP_CORR = 24    # corridor
CMAP_DOOR_OPEN = 28
CMAP_STAIRDN = 30
CMAP_STAIRUP = 29
CMAP_DOOR_CLOSED = 27


# ============================================================
# Lightweight obs parser (stub for when obs_parser.py is missing)
# ============================================================

@dataclass
class _BasicGameState:
    """Minimal game state extracted from NLE observations.
    Used as a fallback when nhc.obs_parser.GameState is unavailable.
    """
    # Player position and stats
    py: int = 0
    px: int = 0
    hp: int = 0
    max_hp: int = 0
    depth: int = 1
    xl: int = 1
    hunger: int = 1  # HungerState enum value
    alignment: int = 0
    ac: int = 10
    turn: int = 0
    condition: int = 0
    score: int = 0

    # Map data
    glyphs: object = None  # 21x79 array
    message: bytes = b""

    # Derived
    adjacent_monsters: list = field(default_factory=list)
    visible_monsters: list = field(default_factory=list)
    on_stairs_down: bool = False
    on_stairs_up: bool = False
    on_food: bool = False
    on_item: bool = False
    on_corpse: bool = False
    corpse_name: str = ""
    stairs_down_pos: Optional[tuple] = None
    explored_ratio: float = 0.0

    def update(self, obs: dict) -> None:
        """Parse raw NLE observation dict into structured state."""
        bl = obs.get("blstats")
        if bl is None:
            return
        bl = list(bl)

        self.px = int(bl[BL_X])
        self.py = int(bl[BL_Y])
        self.hp = int(bl[BL_HP])
        self.max_hp = max(1, int(bl[BL_MAXHP]))
        self.depth = int(bl[BL_DEPTH])
        self.xl = int(bl[BL_XL])
        self.hunger = int(bl[BL_HUNGER])
        self.alignment = int(bl[BL_ALIGNMENT])
        self.ac = int(bl[BL_AC])
        self.turn = int(bl[BL_TIME]) if len(bl) > BL_TIME else 0
        self.condition = int(bl[BL_CONDITION]) if len(bl) > BL_CONDITION else 0
        self.score = int(bl[BL_SCORE])

        glyphs = obs.get("glyphs")
        if glyphs is not None:
            self.glyphs = glyphs
            self._parse_surroundings(glyphs)

        msg = obs.get("message")
        if msg is not None:
            if hasattr(msg, "tobytes"):
                self.message = msg.tobytes().rstrip(b"\x00")
            elif isinstance(msg, bytes):
                self.message = msg.rstrip(b"\x00")
            else:
                self.message = bytes(msg)

    def _parse_surroundings(self, glyphs) -> None:
        """Scan the glyph grid for monsters, items, and terrain near player."""
        self.adjacent_monsters = []
        self.visible_monsters = []
        self.on_stairs_down = False
        self.on_stairs_up = False
        self.on_food = False
        self.on_item = False
        self.on_corpse = False
        self.corpse_name = ""
        self.stairs_down_pos = None

        rows = len(glyphs) if hasattr(glyphs, "__len__") else 0
        if rows == 0:
            return

        cols = len(glyphs[0]) if rows > 0 and hasattr(glyphs[0], "__len__") else 0
        py, px = self.py, self.px

        # Check tile under player
        if 0 <= py < rows and 0 <= px < cols:
            g = int(glyphs[py][px])
            if _glyph_is_stairs_down(g):
                self.on_stairs_down = True
            if _glyph_is_stairs_up(g):
                self.on_stairs_up = True
            if _glyph_is_corpse(g):
                self.on_corpse = True
                self.corpse_name = _corpse_name(g)
            if _glyph_is_object(g):
                self.on_item = True

        # Scan 21x79 for monsters
        explored = 0
        total_tiles = 0
        for r in range(rows):
            for c in range(cols):
                g = int(glyphs[r][c])
                if g == GLYPH_CMAP_OFF + 32:
                    # unexplored / dark
                    pass
                else:
                    if _glyph_is_floor_or_corridor(g):
                        explored += 1
                    total_tiles += 1

                if _glyph_is_stairs_down(g):
                    self.stairs_down_pos = (r, c)

                if _glyph_is_monster(g) and (r != py or c != px):
                    mon_name = _monster_name(g)
                    dist = max(abs(r - py), abs(c - px))
                    entry = {"name": mon_name, "row": r, "col": c,
                             "dist": dist, "glyph": g}
                    self.visible_monsters.append(entry)
                    if dist == 1:
                        self.adjacent_monsters.append(entry)

        self.explored_ratio = explored / max(1, total_tiles)


# ============================================================
# Glyph helper functions
# ============================================================

def _glyph_is_monster(g: int) -> bool:
    return GLYPH_MON_OFF <= g < GLYPH_PET_OFF

def _glyph_is_pet(g: int) -> bool:
    return GLYPH_PET_OFF <= g < GLYPH_PET_OFF + (GLYPH_PET_OFF - GLYPH_MON_OFF)

def _glyph_is_corpse(g: int) -> bool:
    return GLYPH_BODY_OFF <= g < GLYPH_BODY_OFF + (GLYPH_PET_OFF - GLYPH_MON_OFF)

def _glyph_is_object(g: int) -> bool:
    return GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF

def _glyph_is_stairs_down(g: int) -> bool:
    return g == GLYPH_CMAP_OFF + CMAP_STAIRDN

def _glyph_is_stairs_up(g: int) -> bool:
    return g == GLYPH_CMAP_OFF + CMAP_STAIRUP

def _glyph_is_floor_or_corridor(g: int) -> bool:
    if g == GLYPH_CMAP_OFF + CMAP_FLOOR:
        return True
    if g == GLYPH_CMAP_OFF + CMAP_CORR:
        return True
    if g == GLYPH_CMAP_OFF + CMAP_DOOR_OPEN:
        return True
    return False

def _glyph_is_closed_door(g: int) -> bool:
    return g == GLYPH_CMAP_OFF + CMAP_DOOR_CLOSED

def _monster_name(g: int) -> str:
    """Get monster name from glyph. Requires NLE for real names."""
    if _NLE_AVAILABLE:
        mon_id = g - GLYPH_MON_OFF
        try:
            return _nh.permonst(mon_id).mname
        except Exception:
            return f"monster_{mon_id}"
    return f"monster_{g - GLYPH_MON_OFF}"

def _corpse_name(g: int) -> str:
    """Get monster name whose corpse this is."""
    if _NLE_AVAILABLE:
        mon_id = g - GLYPH_BODY_OFF
        try:
            return _nh.permonst(mon_id).mname
        except Exception:
            return f"corpse_{mon_id}"
    return f"corpse_{g - GLYPH_BODY_OFF}"


# ============================================================
# Navigation helpers (stub for when navigation.py is missing)
# ============================================================

def _direction_toward(py: int, px: int, ty: int, tx: int) -> int:
    """Return the movement action index to step from (py,px) toward (ty,tx).
    Uses Chebyshev-adjacent step (diagonal OK).
    """
    dy = 0 if ty == py else (1 if ty > py else -1)
    dx = 0 if tx == px else (1 if tx > px else -1)
    if dy == 0 and dx == 0:
        return Actions.WAIT
    return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)


def _direction_away(py: int, px: int, ty: int, tx: int) -> int:
    """Return movement action to step away from (ty, tx)."""
    dy = 0 if ty == py else (-1 if ty > py else 1)
    dx = 0 if tx == px else (-1 if tx > px else 1)
    if dy == 0 and dx == 0:
        return Actions.SEARCH
    return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)


def _find_unexplored(glyphs, py: int, px: int) -> Optional[tuple]:
    """BFS for nearest unexplored tile adjacent to an explored tile.
    Returns (row, col) or None.
    """
    if glyphs is None:
        return None
    rows = len(glyphs) if hasattr(glyphs, "__len__") else 0
    if rows == 0:
        return None
    cols = len(glyphs[0]) if hasattr(glyphs[0], "__len__") else 0

    visited = set()
    queue = [(py, px)]
    visited.add((py, px))
    head = 0

    dark_glyph = GLYPH_CMAP_OFF + 32  # unexplored/dark space

    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if (nr, nc) in visited:
                    continue
                visited.add((nr, nc))
                g = int(glyphs[nr][nc])
                if g == dark_glyph or g == 0:
                    # This is unexplored. Return the explored tile
                    # adjacent to it as the target.
                    return (r, c) if (r, c) != (py, px) else (nr, nc)
                if _glyph_is_floor_or_corridor(g) or _glyph_is_monster(g) or _glyph_is_object(g):
                    queue.append((nr, nc))

    return None


# ============================================================
# ExpertAgent
# ============================================================

class ExpertAgent:
    """Priority-based expert system agent for NetHack.

    Takes NLE observations, returns canonical action indices.
    Integrates combat threat assessment, prayer safety, item
    identification, and navigation.
    """

    def __init__(self):
        # Game state parser
        if GameState is not None:
            self.state = GameState()
        else:
            self.state = _BasicGameState()

        # Subsystems
        self.threat_db = ThreatDB() if ThreatDB is not None else None
        self.prayer = PrayerState() if PrayerState is not None else None
        self.item_tracker = AppearanceTracker() if AppearanceTracker is not None else None
        self.dungeon_map = DungeonMap() if DungeonMap is not None else None

        # Per-episode state
        self.resistances: set[str] = set()
        self.has_food: bool = False
        self.has_lizard_corpse: bool = False
        self.inventory_items: list[str] = []
        self.last_action: int = Actions.SEARCH
        self.turns_on_tile: int = 0
        self.last_pos: tuple = (0, 0)
        self.search_count: int = 0

    def reset(self) -> None:
        """Reset state for a new episode."""
        if GameState is not None:
            self.state = GameState()
        else:
            self.state = _BasicGameState()
        if PrayerState is not None:
            self.prayer = PrayerState()
        self.resistances = set()
        self.has_food = False
        self.has_lizard_corpse = False
        self.inventory_items = []
        self.last_action = Actions.SEARCH
        self.turns_on_tile = 0
        self.last_pos = (0, 0)
        self.search_count = 0

    def act(self, obs: dict) -> int:
        """Main decision function. Takes NLE observation, returns action index."""
        self.state.update(obs)
        s = self.state

        # Track stuck detection
        pos = (s.py, s.px)
        if pos == self.last_pos:
            self.turns_on_tile += 1
        else:
            self.turns_on_tile = 0
            self.search_count = 0
        self.last_pos = pos

        # Update prayer subsystem
        if self.prayer is not None:
            self.prayer.update_from_blstats(obs.get("blstats", []))

        # Parse message for clues
        self._parse_message(s.message)

        # Priority-based decision cascade
        action = self._decide(s)
        self.last_action = action
        return action

    def _decide(self, s) -> int:
        """Priority cascade. Returns the action for the highest-priority
        applicable rule.
        """
        # P0: Critical emergencies
        a = self._p0_emergencies(s)
        if a is not None:
            return a

        # P1: Adjacent combat
        a = self._p1_adjacent_combat(s)
        if a is not None:
            return a

        # P2: Ranged threats
        a = self._p2_ranged_threats(s)
        if a is not None:
            return a

        # P3: Food management
        a = self._p3_food(s)
        if a is not None:
            return a

        # P4: Item management (pickup)
        a = self._p4_items(s)
        if a is not None:
            return a

        # P5: Corpse eating for intrinsics
        a = self._p5_corpse_intrinsics(s)
        if a is not None:
            return a

        # P6: Navigation
        a = self._p6_navigation(s)
        if a is not None:
            return a

        # Fallback: search
        return Actions.SEARCH

    # ----------------------------------------------------------
    # P0: Critical emergencies
    # ----------------------------------------------------------

    def _p0_emergencies(self, s) -> Optional[int]:
        cond = s.condition

        # Stoning
        if cond & COND_STONE:
            if self.has_lizard_corpse:
                return Actions.EAT
            return self._try_pray(s, "stoning")

        # Sliming
        if cond & COND_SLIME:
            if self.has_lizard_corpse:
                return Actions.EAT  # lizard cures sliming too
            return self._try_pray(s, "sliming")

        # HP critical: <= maxHP/7
        if s.hp <= max(5, s.max_hp // 7):
            pray_action = self._try_pray(s, "hp_critical")
            if pray_action is not None:
                return pray_action
            # Can't pray: flee or Elbereth
            if s.adjacent_monsters:
                return self._flee_or_elbereth(s)
            return Actions.SEARCH  # rest and hope

        # Fainting from hunger
        if s.hunger >= 4:  # FAINTING
            pray_action = self._try_pray(s, "starving")
            if pray_action is not None:
                return pray_action
            # Try to eat anything
            if self.has_food or s.on_food or s.on_corpse:
                return Actions.EAT
            return None  # fall through, maybe food on ground

        # Terminal illness
        if cond & COND_ILL:
            return self._try_pray(s, "illness")

        return None

    # ----------------------------------------------------------
    # P1: Adjacent combat
    # ----------------------------------------------------------

    def _p1_adjacent_combat(self, s) -> Optional[int]:
        if not s.adjacent_monsters:
            return None

        # Assess each adjacent monster
        threats = []
        for mon in s.adjacent_monsters:
            name = mon["name"]
            if self.threat_db is not None:
                report = self.threat_db.assess_threat(name, self._player_state(s))
            else:
                report = _StubThreatReport(name)
            threats.append((mon, report))

        # Sort by danger (highest first)
        threats.sort(key=lambda x: -x[1].danger_level)

        # Check for instakill risks
        any_instakill = any(r.instakill_risk for _, r in threats)
        if any_instakill:
            # Flee or Elbereth
            return self._flee_or_elbereth(s)

        top_mon, top_report = threats[0]

        # Elbereth if recommended and danger is high
        if (top_report.recommended_action == "elbereth"
                and top_report.elbereth_effective
                and top_report.danger_level >= 6):
            return Actions.ENGRAVE  # write Elbereth

        # Ranged preferred: step away if possible
        if top_report.ranged_preferred:
            return _direction_away(s.py, s.px, top_mon["row"], top_mon["col"])

        # Flee recommendation
        if top_report.recommended_action == "flee":
            return self._flee_from(s, top_mon)

        # Melee: attack the highest-priority target
        return _direction_toward(s.py, s.px, top_mon["row"], top_mon["col"])

    # ----------------------------------------------------------
    # P2: Ranged threats (visible non-adjacent)
    # ----------------------------------------------------------

    def _p2_ranged_threats(self, s) -> Optional[int]:
        non_adjacent = [m for m in s.visible_monsters if m["dist"] > 1]
        if not non_adjacent:
            return None

        # Check for dangerous ranged monsters
        for mon in non_adjacent:
            name = mon["name"]
            if self.threat_db is not None:
                report = self.threat_db.assess_threat(name, self._player_state(s))
            else:
                report = _StubThreatReport(name)

            if report.danger_level >= 8 and report.instakill_risk:
                # Dangerous ranged monster: flee
                return _direction_away(s.py, s.px, mon["row"], mon["col"])

        # Non-critical ranged monsters: approach to melee (score comes from kills)
        closest = min(non_adjacent, key=lambda m: m["dist"])
        if self.threat_db is not None:
            report = self.threat_db.assess_threat(closest["name"], self._player_state(s))
        else:
            report = _StubThreatReport(closest["name"])

        if report.danger_level <= 6 and s.hp > s.max_hp * 0.5:
            return _direction_toward(s.py, s.px, closest["row"], closest["col"])

        return None

    # ----------------------------------------------------------
    # P3: Food management
    # ----------------------------------------------------------

    def _p3_food(self, s) -> Optional[int]:
        # Hungry or worse: eat
        if s.hunger >= 2:  # HUNGRY
            if self.has_food:
                return Actions.EAT
            if s.on_corpse and self._corpse_safe_to_eat(s.corpse_name):
                return Actions.EAT
            if s.on_food:
                return Actions.EAT

        # Standing on a safe corpse and not satiated
        if s.on_corpse and s.hunger > 0 and self._corpse_safe_to_eat(s.corpse_name):
            return Actions.EAT

        return None

    # ----------------------------------------------------------
    # P4: Item management
    # ----------------------------------------------------------

    def _p4_items(self, s) -> Optional[int]:
        if s.on_item:
            return Actions.PICKUP
        return None

    # ----------------------------------------------------------
    # P5: Corpse eating for intrinsics
    # ----------------------------------------------------------

    def _p5_corpse_intrinsics(self, s) -> Optional[int]:
        if not s.on_corpse:
            return None

        name = s.corpse_name
        if not name:
            return None

        if self.threat_db is not None:
            report = self.threat_db.corpse_value(name, self.resistances)
            if report.safe_to_eat and report.priority >= 5:
                return Actions.EAT
        elif self._corpse_safe_to_eat(name):
            return Actions.EAT

        return None

    # ----------------------------------------------------------
    # P6: Navigation
    # ----------------------------------------------------------

    def _p6_navigation(self, s) -> Optional[int]:
        # If on stairs down and explored enough and healthy enough: descend
        if s.on_stairs_down:
            # Only descend if HP > 50% and XL is reasonable for depth
            if s.hp > s.max_hp * 0.5 and s.depth <= s.xl * 2:
                return Actions.DOWN

        # If stairs down found, go there (if explored enough)
        if s.stairs_down_pos is not None and s.explored_ratio > 0.4:
            if s.hp > s.max_hp * 0.5:
                sr, sc = s.stairs_down_pos
                if (sr, sc) != (s.py, s.px):
                    return _direction_toward(s.py, s.px, sr, sc)

        # Explore: BFS to nearest unexplored
        target = _find_unexplored(s.glyphs, s.py, s.px)
        if target is not None:
            return _direction_toward(s.py, s.px, target[0], target[1])

        # If stairs down known and fully explored: go there
        if s.stairs_down_pos is not None:
            sr, sc = s.stairs_down_pos
            if (sr, sc) != (s.py, s.px):
                return _direction_toward(s.py, s.px, sr, sc)

        # Stuck: search for hidden doors (up to a limit, then random move)
        self.search_count += 1
        if self.search_count > 20:
            # Try a random direction to get unstuck
            import random
            dirs = list(Actions.MOVE_DELTAS.keys())
            return random.choice(dirs)

        return Actions.SEARCH

    # ----------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------

    def _try_pray(self, s, trouble_type: str) -> Optional[int]:
        """Attempt prayer if safe. Returns PRAY action or None."""
        if self.prayer is not None:
            safe, reason = self.prayer.is_prayer_safe(s.turn, trouble_type)
            if safe:
                self.prayer.update_prayed(s.turn)
                return Actions.PRAY
            return None
        # No prayer subsystem: heuristic check
        # Assume first prayer at turn >= 300 is safe
        if s.turn >= 300 and s.alignment >= 0:
            return Actions.PRAY
        return None

    def _flee_or_elbereth(self, s) -> int:
        """Flee from adjacent monsters, or write Elbereth if surrounded."""
        if len(s.adjacent_monsters) >= 3:
            # Surrounded: Elbereth is better than running
            return Actions.ENGRAVE

        if s.adjacent_monsters:
            return self._flee_from(s, s.adjacent_monsters[0])

        return Actions.SEARCH

    def _flee_from(self, s, monster: dict) -> int:
        """Move away from a specific monster."""
        return _direction_away(s.py, s.px, monster["row"], monster["col"])

    def _corpse_safe_to_eat(self, name: str) -> bool:
        """Quick safety check without ThreatDB."""
        unsafe = {
            "green slime", "cockatrice", "chickatrice", "Medusa",
            "Death", "Pestilence", "Famine",
            "chameleon", "doppelganger", "sandestin",
        }
        if name in unsafe:
            return False
        # Poisonous corpses need poison resistance
        poisonous = {
            "killer bee", "scorpion", "pit viper", "cobra",
            "water moccasin", "asp", "python", "giant spider",
            "quasit", "rabid rat",
        }
        if name in poisonous and "poison resistance" not in self.resistances:
            return False
        return True

    def _player_state(self, s) -> dict:
        """Build player_state dict for combat.py assess_threat calls."""
        return {
            "hp": s.hp,
            "max_hp": s.max_hp,
            "ac": s.ac,
            "level": s.xl,
            "speed": 12,  # default human speed
            "resistances": self.resistances,
            "equipment": {},
            "position": (s.py, s.px),
            "has_elbereth_source": True,  # assume we can always write
        }

    def _parse_message(self, msg: bytes) -> None:
        """Extract useful info from game messages."""
        if not msg:
            return
        text = msg.decode("ascii", errors="replace").lower()

        # Track food in inventory
        if "food ration" in text or "ration" in text:
            self.has_food = True
        if "lizard corpse" in text:
            self.has_lizard_corpse = True
        if "you feel full" in text or "you are beginning to feel hungry" in text:
            pass  # hunger state tracked via blstats
        # Track resistance gains
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


# ============================================================
# Stub threat report for when combat.py is unavailable
# ============================================================

@dataclass
class _StubThreatReport:
    """Minimal threat report when ThreatDB is not loaded."""
    name: str
    danger_level: int = 5
    special_attacks: list = field(default_factory=list)
    required_resistances: list = field(default_factory=list)
    elbereth_effective: bool = True
    ranged_preferred: bool = False
    instakill_risk: bool = False
    recommended_action: str = "melee"

    def __post_init__(self):
        # Flag known instakill monsters
        instakill_names = {
            "cockatrice", "chickatrice", "Medusa",
            "green slime", "Death", "Pestilence", "Famine",
        }
        if self.name in instakill_names:
            self.instakill_risk = True
            self.danger_level = 10
            self.recommended_action = "flee"

        ranged_names = {
            "floating eye", "cockatrice", "chickatrice",
            "rust monster", "acid blob", "brown mold",
        }
        if self.name in ranged_names:
            self.ranged_preferred = True
            self.recommended_action = "ranged"


# ============================================================
# Mock test
# ============================================================

def _make_mock_obs(
    *,
    py: int = 10, px: int = 40,
    hp: int = 30, max_hp: int = 50,
    depth: int = 1, xl: int = 3,
    hunger: int = 1, turn: int = 500,
    condition: int = 0,
    alignment: int = 5,
    monsters: Optional[list] = None,
    stairs_down: Optional[tuple] = None,
    corpse_at_player: Optional[int] = None,
    object_at_player: bool = False,
) -> dict:
    """Build a fake NLE observation dict for testing."""
    # 21x79 glyph grid, filled with floor
    floor_g = GLYPH_CMAP_OFF + CMAP_FLOOR
    dark_g = GLYPH_CMAP_OFF + 32
    glyphs = [[dark_g] * 79 for _ in range(21)]

    # Carve a room around the player
    for r in range(max(1, py - 3), min(20, py + 4)):
        for c in range(max(1, px - 5), min(78, px + 6)):
            glyphs[r][c] = floor_g

    # Place player's tile
    if corpse_at_player is not None:
        glyphs[py][px] = GLYPH_BODY_OFF + corpse_at_player
    elif object_at_player:
        glyphs[py][px] = GLYPH_OBJ_OFF + 10  # arbitrary object
    else:
        glyphs[py][px] = floor_g

    # Place monsters
    if monsters:
        for mr, mc, mon_id in monsters:
            glyphs[mr][mc] = GLYPH_MON_OFF + mon_id

    # Place stairs
    if stairs_down:
        sr, sc = stairs_down
        glyphs[sr][sc] = GLYPH_CMAP_OFF + CMAP_STAIRDN

    # Build blstats (26 elements)
    bl = [0.0] * 26
    bl[BL_X] = float(px)
    bl[BL_Y] = float(py)
    bl[BL_HP] = float(hp)
    bl[BL_MAXHP] = float(max_hp)
    bl[BL_DEPTH] = float(depth)
    bl[BL_XL] = float(xl)
    bl[BL_HUNGER] = float(hunger)
    bl[BL_ALIGNMENT] = float(alignment)
    bl[BL_AC] = 5.0
    bl[BL_CONDITION] = float(condition)
    bl[BL_TIME] = float(turn)
    bl[BL_SCORE] = 100.0

    return {
        "glyphs": glyphs,
        "blstats": bl,
        "message": b"",
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
    # Can't pray, no adjacent monsters, should search/rest
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
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         monsters=[(9, 40, 5)])  # monster at (9, 40), one tile N
    action = agent.act(obs)
    check("adjacent monster should trigger melee (move toward)",
          action in Actions.MOVE_DELTAS,
          f"got action {action}")

    # --- Test 5: P1 - Adjacent instakill monster ---
    print("\nTest 5: P1 - Adjacent cockatrice (instakill)")
    agent.reset()
    # cockatrice mon_id: use a high ID, but we need to know the actual ID.
    # Without NLE, _StubThreatReport flags "cockatrice" as instakill by name.
    # But _monster_name won't return "cockatrice" without NLE.
    # So test with the stub: manually set adjacent_monsters.
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500,
                         monsters=[(9, 40, 50)])  # some monster
    # Override the parsed name for testing
    agent.act(obs)
    s = agent.state
    if s.adjacent_monsters:
        s.adjacent_monsters[0]["name"] = "cockatrice"
        s.visible_monsters[0]["name"] = "cockatrice"
    action = agent._decide(s)
    check("instakill adjacent should flee or Elbereth",
          action in list(Actions.MOVE_DELTAS.keys()) + [Actions.ENGRAVE],
          f"got action {action}")

    # --- Test 6: P3 - Hungry with food ---
    print("\nTest 6: P3 - Hungry with food")
    agent.reset()
    agent.has_food = True
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, hunger=2)  # HUNGRY
    action = agent.act(obs)
    check("hungry with food should eat", action == Actions.EAT,
          f"got action {action}")

    # --- Test 7: P4 - Item on ground ---
    print("\nTest 7: P4 - Item on ground")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, object_at_player=True)
    action = agent.act(obs)
    check("item on ground should pickup", action == Actions.PICKUP,
          f"got action {action}")

    # --- Test 8: P6 - Navigate to stairs ---
    print("\nTest 8: P6 - Navigate to stairs when explored")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, depth=1, xl=3,
                         stairs_down=(12, 42))
    action = agent.act(obs)
    # Should move toward stairs
    check("should navigate toward stairs",
          action in Actions.MOVE_DELTAS,
          f"got action {action}")

    # --- Test 9: P6 - On stairs, descend ---
    print("\nTest 9: P6 - On stairs, should descend")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, depth=1, xl=3,
                         stairs_down=(10, 40))  # player position
    action = agent.act(obs)
    check("on stairs with good HP should descend", action == Actions.DOWN,
          f"got action {action}")

    # --- Test 10: P6 - On stairs but low HP ---
    print("\nTest 10: P6 - On stairs but low HP, don't descend")
    agent.reset()
    obs = _make_mock_obs(hp=10, max_hp=50, turn=500, depth=1, xl=3,
                         stairs_down=(10, 40))
    action = agent.act(obs)
    check("on stairs with low HP should not descend", action != Actions.DOWN,
          f"got action {action}")

    # --- Test 11: Priority ordering - P0 over P1 ---
    print("\nTest 11: Priority - P0 (critical HP) beats P1 (combat)")
    agent.reset()
    obs = _make_mock_obs(hp=3, max_hp=50, turn=500,
                         monsters=[(9, 40, 5)])  # adjacent monster
    action = agent.act(obs)
    check("critical HP should pray even with adjacent monster",
          action == Actions.PRAY,
          f"got action {action}")

    # --- Test 12: P5 - Corpse eating for intrinsics ---
    print("\nTest 12: P5 - Standing on beneficial corpse")
    agent.reset()
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500, corpse_at_player=0)
    # Without NLE, corpse name is "corpse_0". The stub won't flag it
    # as unsafe, so it should eat. We override the name.
    agent.act(obs)
    s = agent.state
    s.on_corpse = True
    s.corpse_name = "floating eye"
    action = agent._decide(s)
    check("beneficial corpse should eat (floating eye)",
          action == Actions.EAT,
          f"got action {action}")

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
    obs = _make_mock_obs(hp=40, max_hp=50, turn=500)
    obs["message"] = b"You feel especially healthy."
    agent.act(obs)
    check("poison resistance detected from message",
          "poison resistance" in agent.resistances)

    # --- Test 15: Hunger fainting triggers prayer ---
    print("\nTest 15: P0 - Fainting from hunger")
    agent.reset()
    obs = _make_mock_obs(hp=30, max_hp=50, turn=500, hunger=4)  # FAINTING
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

    # --- Summary ---
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
