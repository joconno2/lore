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
        S_DNSTAIR, S_UPSTAIR,
        MAP_H, MAP_W,
        COND_STONE, COND_SLIME,
    )
    _HAS_OBS_PARSER = True
except ImportError:
    _HAS_OBS_PARSER = False
    GameState = None
    # Fallback constants
    GLYPH_MON_OFF = 0
    GLYPH_PET_OFF = 381
    GLYPH_BODY_OFF = 1144
    GLYPH_OBJ_OFF = 1906
    GLYPH_CMAP_OFF = 2359
    GLYPH_RIDDEN_OFF = 1525
    MAP_H = 21
    MAP_W = 79
    S_DNSTAIR = GLYPH_CMAP_OFF + 24
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
    APPLY = 22; CLOSE = 22; DROP = 22; ENGRAVE = 22
    FIRE = 22; INV = 22; LOOT = 22; OPEN = 22
    PAY = 22; PICKUP = 22; PRAY = 22; PUTON = 22
    QUAFF = 22; READ = 22; REMOVE = 22; RIDE = 22
    TAKEOFF = 22; THROW = 22; WEAR = 22; WIELD = 22
    ZAP = 22
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
            "APPLY": "APPLY", "CLOSE": "CLOSE", "DROP": "DROP",
            "EAT": "EAT", "ENGRAVE": "ENGRAVE", "FIRE": "FIRE",
            "KICK": "KICK", "LOOT": "LOOT", "OPEN": "OPEN",
            "PICKUP": "PICKUP", "PRAY": "PRAY", "PUTON": "PUTON",
            "QUAFF": "QUAFF", "READ": "READ", "SEARCH": "SEARCH",
            "TAKEOFF": "TAKEOFF", "THROW": "THROW", "WEAR": "WEAR",
            "WIELD": "WIELD", "ZAP": "ZAP", "MORE": "MORE",
            "INV": "INVENTORY", "REMOVE": "REMOVE",
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

def _direction_toward(py: int, px: int, ty: int, tx: int) -> int:
    """Movement action to step from (py,px) toward (ty,tx). Chebyshev step."""
    dy = 0 if ty == py else (1 if ty > py else -1)
    dx = 0 if tx == px else (1 if tx > px else -1)
    if dy == 0 and dx == 0:
        return Actions.WAIT
    return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)


def _direction_away(py: int, px: int, ty: int, tx: int) -> int:
    """Movement action to step away from (ty, tx)."""
    dy = 0 if ty == py else (-1 if ty > py else 1)
    dx = 0 if tx == px else (-1 if tx > px else 1)
    if dy == 0 and dx == 0:
        return Actions.SEARCH
    return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)


def _find_unexplored(glyphs: np.ndarray, py: int, px: int) -> Optional[tuple]:
    """BFS for nearest reachable tile adjacent to unexplored space.
    Returns (row, col) or None.
    """
    if glyphs is None:
        return None
    rows, cols = glyphs.shape
    stone_glyph = GLYPH_CMAP_OFF  # cmap index 0 = stone/unexplored

    visited = set()
    queue = [(py, px)]
    visited.add((py, px))
    head = 0

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
                g = int(glyphs[nr, nc])
                if g == stone_glyph:
                    # Unexplored stone. Return the explored neighbor as target.
                    return (r, c) if (r, c) != (py, px) else None
                # Walkable OR closed door (we can open doors)
                if _is_walkable_glyph(g) or _is_closed_door_glyph(g):
                    queue.append((nr, nc))

    return None


def _find_stairs_down(glyphs: np.ndarray) -> Optional[tuple]:
    """Scan the glyph map for downstairs. Returns (row, col) or None."""
    if glyphs is None:
        return None
    rows, cols = glyphs.shape
    for r in range(rows):
        for c in range(cols):
            g = int(glyphs[r, c])
            if g == S_DNSTAIR or g == S_DNSTAIR + 2:  # stairs or ladder
                return (r, c)
    return None


def _is_walkable_glyph(g: int) -> bool:
    """Check if a glyph represents a tile the player can walk on."""
    # Monsters (we can walk toward them to attack)
    if GLYPH_MON_OFF <= g < GLYPH_OBJ_OFF:
        return True
    # Objects on the floor
    if GLYPH_OBJ_OFF <= g < GLYPH_CMAP_OFF:
        return True
    # Dungeon features
    if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87:
        cmap_idx = g - GLYPH_CMAP_OFF
        # 12=doorway, 13=vertical open door, 14=horizontal open door
        # 19=room, 20=dark room, 21=corridor, 22=lit corridor
        # 23=upstair, 24=downstair, 25=upladder, 26=downladder
        # 27=altar, 28=grave, 29=throne, 30=sink, 31=fountain
        # 32=pool, 33=moat, 34=water, 35=drawbridge, 36=lava (NOT walkable)
        walkable_cmaps = {12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
        return cmap_idx in walkable_cmaps
    return False


def _is_door_glyph(g: int) -> bool:
    """Check if glyph is any door (open or closed)."""
    if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87:
        return (g - GLYPH_CMAP_OFF) in (12, 13, 14, 15, 16)  # doorway, open, closed
    return False


def _is_closed_door_glyph(g: int) -> bool:
    """Check if glyph is a closed door."""
    if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87:
        return (g - GLYPH_CMAP_OFF) in (15, 16)  # vertical/horizontal closed door
    return False


def _count_explored(glyphs: np.ndarray) -> float:
    """Return fraction of non-stone tiles (rough exploration estimate)."""
    if glyphs is None:
        return 0.0
    total = glyphs.size
    stone_count = np.sum(glyphs == GLYPH_CMAP_OFF)
    explored = total - stone_count
    return explored / max(1, total)


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

@dataclass
class _StubThreatReport:
    name: str
    danger_level: int = 5
    special_attacks: list = field(default_factory=list)
    required_resistances: list = field(default_factory=list)
    elbereth_effective: bool = True
    ranged_preferred: bool = False
    instakill_risk: bool = False
    recommended_action: str = "melee"

    def __post_init__(self):
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

        # Per-turn tile state (detected from messages + glyphs)
        self._on_corpse: bool = False
        self._corpse_name: str = ""
        self._on_item: bool = False
        self._on_edible_item: bool = False
        # Multi-step action state
        self._pending_action: Optional[str] = None  # "eat", "pray", etc.
        self._eat_attempts: int = 0

    def reset(self) -> None:
        """Reset state for a new episode."""
        self._step_count = 0
        self.state = GameState() if _HAS_OBS_PARSER else None
        if _HAS_PRAYER:
            self.prayer = PrayerState()
        self.resistances = set()
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

    def act(self, obs: dict) -> int:
        """Main decision function. Takes NLE observation, returns action index."""
        self._step_count += 1
        s = self.state
        s.update(obs)

        # Check for --More-- prompt. Must clear before anything else works.
        msg_raw = obs.get("message")
        if msg_raw is not None:
            msg_bytes = bytes(msg_raw).rstrip(b'\x00')
            msg_str = msg_bytes.decode("latin-1", errors="replace").strip()
            if "--More--" in msg_str or msg_str.endswith("--More--"):
                if self.verbose:
                    self._log("MORE", f"clearing prompt: {msg_str[:60]}")
                return Actions.MORE
            # Handle eat prompts
            if "What do you want to eat?" in msg_str:
                # Find first food item letter in inventory
                food_letter = self._find_food_letter(s)
                if food_letter is not None:
                    if self.verbose:
                        self._log("EAT", f"selecting item '{food_letter}'")
                    self._pending_action = None
                    return self._letter_to_action(food_letter)
                else:
                    # No food found, cancel with ESC
                    if self.verbose:
                        self._log("EAT", "no food in inventory, canceling")
                    self._pending_action = None
                    self.has_food = False
                    return Actions.MORE  # ESC or space to cancel
            # "eat it? [yn]" for corpse on ground
            if "eat it?" in msg_str or "eat this?" in msg_str:
                if self.verbose:
                    self._log("EAT", f"confirming: {msg_str[:40]}")
                self._pending_action = None
                return self._letter_to_action('y')
            # Handle yn prompts (default yes)
            if msg_str.endswith("[yn]") or msg_str.endswith("[ynq]"):
                if self.verbose:
                    self._log("YN", f"answering yes to: {msg_str[:60]}")
                return self._letter_to_action('y')
            # Handle menu dismissal
            if msg_str.endswith("[q]") or msg_str.endswith("(end)"):
                if self.verbose:
                    self._log("MENU", f"dismissing: {msg_str[:60]}")
                return Actions.MORE
            # "Never mind" means last action failed, clear pending
            if "Never mind" in msg_str or "You cannot eat that" in msg_str:
                if self._pending_action == "eat":
                    self._eat_attempts += 1
                    if self._eat_attempts >= 3:
                        self._pending_action = None
                        self.has_food = False
                        self._eat_attempts = 0

        # Track stuck detection
        pos = s.position
        if pos == self.last_pos:
            self.turns_on_tile += 1
        else:
            self.turns_on_tile = 0
            self.search_count = 0
        self.last_pos = pos

        # Update prayer subsystem from blstats
        if self.prayer is not None:
            bl = obs.get("blstats")
            if bl is not None:
                self.prayer.update_from_blstats(bl)

        # Parse messages for resistance gains and inventory clues
        self._parse_message_for_state(s)

        # Detect what's on the player's tile (from messages + glyphs)
        self._detect_tile_contents(s)

        # Scan inventory for food / lizard
        self._check_inventory(s)

        action = self._decide(s)
        if self.verbose:
            self._log_decision(s, action)
        return action

    def _log(self, tag: str, msg: str):
        """Print a tagged log line."""
        s = self.state
        print(f"  [{self._step_count:>5} t{s.turn:>5} dl{s.dlevel} hp{s.hp}/{s.max_hp}] {tag}: {msg}")

    def _log_decision(self, s, action: int):
        """Log the chosen action every N steps."""
        if self._step_count % 50 != 0:
            return
        action_name = self._action_name(action)
        monsters = [m.name for m in s.adjacent_monsters if not getattr(m, 'is_pet', False)]
        msg = s.messages[-1] if s.messages else ""
        self._log("ACT", f"{action_name} | adj={monsters} hunger={s.hunger_state} "
                  f"pos={s.position} msg={msg[:40]}")

    @staticmethod
    def _action_name(action: int) -> str:
        """Reverse-lookup action index to name."""
        for name in dir(Actions):
            if name.startswith('_') or name in ('MOVE_DELTAS', 'DELTA_TO_MOVE', 'NUM_ACTIONS'):
                continue
            if getattr(Actions, name) == action:
                return name
        return f"action_{action}"

    def _decide(self, s) -> int:
        """Priority cascade. Returns action for highest-priority applicable rule."""
        priorities = [
            ("P0-emerg", self._p0_emergencies),
            ("P1-combat", self._p1_adjacent_combat),
            ("P2-ranged", self._p2_ranged_threats),
            ("P3-food", self._p3_food),
            ("P4-items", self._p4_items),
            ("P5-corpse", self._p5_corpse_intrinsics),
            ("P6-nav", self._p6_navigation),
        ]
        for name, fn in priorities:
            a = fn(s)
            if a is not None:
                if self.verbose and self._step_count % 10 == 0:
                    self._log(name, f"-> {self._action_name(a)} (idx={a})")
                return a

        if self.verbose and self._step_count % 10 == 0:
            self._log("FALLBACK", "no priority matched, searching")
        return Actions.SEARCH

    # ----------------------------------------------------------
    # P0: Critical emergencies
    # ----------------------------------------------------------

    def _p0_emergencies(self, s) -> Optional[int]:
        conds = s.conditions

        # Stoning
        if "stoned" in conds:
            if self.has_lizard_corpse:
                return Actions.EAT
            return self._try_pray(s, "stoning")

        # Sliming
        if "slimed" in conds:
            if self.has_lizard_corpse:
                return Actions.EAT
            return self._try_pray(s, "sliming")

        # HP critical: HP <= max(5, maxHP // 7)
        if s.hp <= max(5, s.max_hp // 7):
            pray_action = self._try_pray(s, "hp_critical")
            if pray_action is not None:
                return pray_action
            # Can't pray: flee or Elbereth
            if s.has_adjacent_monsters:
                return self._flee_or_elbereth(s)
            return Actions.SEARCH  # rest

        # Fainting from hunger
        if s.hunger_state in ("fainting", "fainted", "starved"):
            pray_action = self._try_pray(s, "starving")
            if pray_action is not None:
                return pray_action
            if self.has_food:
                return Actions.EAT
            return None

        # Terminal illness / food poisoning
        if "foodpois" in conds or "termill" in conds:
            return self._try_pray(s, "illness")

        return None

    # ----------------------------------------------------------
    # P1: Adjacent combat
    # ----------------------------------------------------------

    def _p1_adjacent_combat(self, s) -> Optional[int]:
        # Filter out pets
        hostile = [m for m in s.adjacent_monsters if not m.is_pet]
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

        # Instakill risks: flee or Elbereth
        any_instakill = any(r.instakill_risk for _, r in threats)
        if any_instakill:
            return self._flee_or_elbereth(s)

        top_mon, top_report = threats[0]

        # Elbereth if recommended and high danger
        if (top_report.recommended_action == "elbereth"
                and top_report.elbereth_effective
                and top_report.danger_level >= 6):
            return Actions.ENGRAVE

        # Ranged preferred: step away
        if top_report.ranged_preferred:
            return _direction_away(py, px, top_mon.row, top_mon.col)

        # Flee recommendation
        if top_report.recommended_action == "flee":
            return self._flee_from(s, top_mon)

        # Melee the highest-priority target
        direction = _direction_toward(py, px, top_mon.row, top_mon.col)
        if self.verbose:
            self._log("P1", f"melee {top_mon.name} at ({top_mon.row},{top_mon.col}) from ({py},{px}) -> dir={direction} ({self._action_name(direction)})")
        return direction

    # ----------------------------------------------------------
    # P2: Ranged threats (visible non-adjacent)
    # ----------------------------------------------------------

    def _p2_ranged_threats(self, s) -> Optional[int]:
        py, px = s.position
        non_adjacent = [
            m for m in s.visible_monsters
            if not m.is_pet and (abs(m.row - py) > 1 or abs(m.col - px) > 1)
        ]
        if not non_adjacent:
            return None

        for mon in non_adjacent:
            if self.threat_db is not None:
                report = self.threat_db.assess_threat(mon.name, self._player_state(s))
            else:
                report = _StubThreatReport(mon.name)
            if report.danger_level >= 8 and report.instakill_risk:
                return _direction_away(py, px, mon.row, mon.col)

        # Approach closest non-critical monster for kills
        closest = min(non_adjacent, key=lambda m: max(abs(m.row - py), abs(m.col - px)))
        if self.threat_db is not None:
            report = self.threat_db.assess_threat(closest.name, self._player_state(s))
        else:
            report = _StubThreatReport(closest.name)

        if report.danger_level <= 6 and s.hp > s.max_hp * 0.5:
            return _direction_toward(py, px, closest.row, closest.col)

        return None

    # ----------------------------------------------------------
    # P3: Food management
    # ----------------------------------------------------------

    def _p3_food(self, s) -> Optional[int]:
        if s.hunger_state in ("hungry", "weak"):
            if self.has_food:
                self._pending_action = "eat"
                self._eat_attempts = 0
                if self.verbose:
                    self._log("P3", "hungry, eating from inventory")
                return Actions.EAT
            # Check for corpse on our tile
            if self._on_corpse and self._corpse_safe_to_eat(self._corpse_name):
                self._pending_action = "eat"
                self._eat_attempts = 0
                if self.verbose:
                    self._log("P3", f"hungry, eating corpse: {self._corpse_name}")
                return Actions.EAT

        return None

    # ----------------------------------------------------------
    # P4: Item management
    # ----------------------------------------------------------

    def _p4_items(self, s) -> Optional[int]:
        if self._on_item:
            return Actions.PICKUP
        return None

    # ----------------------------------------------------------
    # P5: Corpse eating for intrinsics
    # ----------------------------------------------------------

    def _p5_corpse_intrinsics(self, s) -> Optional[int]:
        if not self._on_corpse or not self._corpse_name:
            return None

        if self.threat_db is not None:
            report = self.threat_db.corpse_value(self._corpse_name, self.resistances)
            if report.safe_to_eat and report.priority >= 5:
                return Actions.EAT
        elif self._corpse_safe_to_eat(self._corpse_name):
            return Actions.EAT

        return None

    # ----------------------------------------------------------
    # P6: Navigation
    # ----------------------------------------------------------

    def _p6_navigation(self, s) -> Optional[int]:
        py, px = s.position
        glyphs = s._glyphs
        if glyphs is None:
            return Actions.SEARCH

        # On downstairs and healthy enough: descend
        if s.on_stairs_down:
            if s.hp > s.max_hp * 0.5 and s.depth <= s.xlevel * 2:
                if self.verbose:
                    self._log("P6", "descending stairs")
                return Actions.DOWN

        # Check for adjacent closed door (open it cardinally)
        door_action = self._try_open_adjacent_door(s, glyphs, py, px)
        if door_action is not None:
            return door_action

        # Find stairs down
        stairs_pos = _find_stairs_down(glyphs)
        explored = _count_explored(glyphs)

        # BFS to nearest unexplored tile adjacent to stone
        target = _find_unexplored(glyphs, py, px)
        if target is not None:
            step = self._bfs_step_toward(glyphs, py, px, target[0], target[1])
            if step is not None:
                return step

        # If stairs found and explored enough: path to stairs
        if stairs_pos is not None and explored > 0.3:
            if s.hp > s.max_hp * 0.5:
                sr, sc = stairs_pos
                if (sr, sc) != (py, px):
                    step = self._bfs_step_toward(glyphs, py, px, sr, sc)
                    if step is not None:
                        return step

        # Stuck: search for hidden doors
        self.search_count += 1
        if self.search_count > 20:
            import random
            dirs = list(Actions.MOVE_DELTAS.keys())
            return random.choice(dirs)

        return Actions.SEARCH

    def _try_open_adjacent_door(self, s, glyphs, py, px) -> Optional[int]:
        """Try to open a closed door adjacent to the player (cardinal only)."""
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = py + dy, px + dx
            if 0 <= nr < 21 and 0 <= nc < 79:
                g = int(glyphs[nr, nc])
                if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87:
                    cmap_idx = g - GLYPH_CMAP_OFF
                    # 15 = vertical closed door, 16 = horizontal closed door
                    if cmap_idx in (15, 16):
                        if self.verbose:
                            self._log("P6", f"opening door at ({nr},{nc})")
                        # Walk into the door to open it (NetHack auto-opens)
                        return Actions.DELTA_TO_MOVE.get((dy, dx), Actions.SEARCH)
        return None

    def _bfs_step_toward(self, glyphs, py, px, ty, tx) -> Optional[int]:
        """BFS pathfind from (py,px) to (ty,tx), return first step action.
        Respects walls, doors (cardinal only through doors), no diagonal squeeze."""
        rows, cols = glyphs.shape
        parent = {}
        visited = set()
        from collections import deque
        queue = deque([(py, px)])
        visited.add((py, px))
        found = False

        while queue:
            r, c = queue.popleft()
            if (r, c) == (ty, tx):
                found = True
                break
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nr, nc = r + dy, c + dx
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if (nr, nc) in visited:
                        continue
                    g = int(glyphs[nr, nc])
                    if not _is_walkable_glyph(g):
                        # Also allow closed doors (we can open them)
                        if not _is_closed_door_glyph(g):
                            continue
                    is_diagonal = abs(dy) + abs(dx) > 1
                    if is_diagonal:
                        # Can't move diagonally through doors
                        src_g = int(glyphs[r, c])
                        if _is_door_glyph(g) or _is_door_glyph(src_g):
                            continue
                        # Can't squeeze diagonally between two walls
                        adj1 = int(glyphs[r, c + dx]) if 0 <= c + dx < cols else 0
                        adj2 = int(glyphs[r + dy, c]) if 0 <= r + dy < rows else 0
                        if not (_is_walkable_glyph(adj1) or _is_walkable_glyph(adj2)):
                            continue
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

        if not found:
            return None

        # Trace path back to first step
        cur = (ty, tx)
        while parent.get(cur) != (py, px):
            cur = parent.get(cur)
            if cur is None:
                return None
        # cur is now the first step from (py, px)
        dy, dx = cur[0] - py, cur[1] - px
        return Actions.DELTA_TO_MOVE.get((dy, dx), None)

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
        if s.turn >= 300:
            return Actions.PRAY
        return None

    def _flee_or_elbereth(self, s) -> int:
        """Flee from adjacent monsters, or write Elbereth if surrounded."""
        hostile = [m for m in s.adjacent_monsters if not m.is_pet]
        py, px = s.position

        if len(hostile) >= 3:
            return Actions.ENGRAVE

        if hostile:
            return self._flee_from(s, hostile[0])

        return Actions.SEARCH

    def _flee_from(self, s, monster) -> int:
        py, px = s.position
        return _direction_away(py, px, monster.row, monster.col)

    def _corpse_safe_to_eat(self, name: str) -> bool:
        """Quick safety check without ThreatDB."""
        unsafe = {
            "green slime", "cockatrice", "chickatrice", "Medusa",
            "Death", "Pestilence", "Famine",
            "chameleon", "doppelganger", "sandestin",
        }
        if name in unsafe:
            return False
        poisonous = {
            "killer bee", "scorpion", "pit viper", "cobra",
            "water moccasin", "asp", "python", "giant spider",
            "quasit", "rabid rat",
        }
        if name in poisonous and "poison resistance" not in self.resistances:
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

    def _check_inventory(self, s) -> None:
        """Scan inventory strings for food and lizard corpses."""
        if not s.inventory:
            return
        self.has_food = False
        self.has_lizard_corpse = False
        self.inventory_items = {}
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            self.inventory_items[letter] = item_str
            if "food ration" in lower or "ration" in lower or "tripe" in lower:
                self.has_food = True
            if "lizard corpse" in lower:
                self.has_lizard_corpse = True
                self.has_food = True
            if "corpse" in lower or "meatball" in lower or "egg" in lower:
                self.has_food = True

    def _find_food_letter(self, s) -> Optional[str]:
        """Find the inventory letter of the best food item to eat."""
        if not s.inventory:
            return None
        # Prefer: food ration > tripe > corpse > other
        best = None
        best_priority = -1
        for letter, item_str in s.inventory.items():
            lower = item_str.lower()
            if "food ration" in lower:
                if best_priority < 3:
                    best, best_priority = letter, 3
            elif "tripe" in lower:
                if best_priority < 2:
                    best, best_priority = letter, 2
            elif "corpse" in lower:
                if best_priority < 1:
                    best, best_priority = letter, 1
            elif "meatball" in lower or "egg" in lower or "apple" in lower:
                if best_priority < 1:
                    best, best_priority = letter, 1
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
