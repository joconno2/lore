"""Parse raw NLE observations into semantic game state for the expert system.

Converts the glyphs/blstats/message observation dict produced by env.py into
a GameState object with typed fields for player stats, nearby monsters,
dungeon features, conditions, and inventory. Maintains persistent state
across turns.

Glyph ranges and blstats layout match NLE 0.9+ (27-element blstats,
MAX_GLYPH=5976, NUMMONS=381).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ============================================================
# NLE constants (hardcoded to avoid NLE import at runtime)
# ============================================================

NUMMONS = 381
MAX_GLYPH = 5976
NUM_OBJECTS = 453

# Glyph range offsets
GLYPH_MON_OFF = 0
GLYPH_PET_OFF = 381
GLYPH_INVIS_OFF = 762
GLYPH_DETECT_OFF = 763
GLYPH_BODY_OFF = 1144
GLYPH_RIDDEN_OFF = 1525
GLYPH_OBJ_OFF = 1906
GLYPH_CMAP_OFF = 2359
GLYPH_EXPLODE_OFF = 2446
GLYPH_ZAP_OFF = 2509
GLYPH_SWALLOW_OFF = 2541
GLYPH_WARNING_OFF = 5589
GLYPH_STATUE_OFF = 5595

# Dungeon feature cmap indices (offset from GLYPH_CMAP_OFF)
CMAP_UPSTAIR = 23
CMAP_DNSTAIR = 24
CMAP_UPLADDER = 25
CMAP_DNLADDER = 26
CMAP_ALTAR = 27
CMAP_FOUNTAIN = 31
CMAP_POOL = 32
CMAP_LAVA = 34
CMAP_ROOM = 19
CMAP_DARKROOM = 20
CMAP_CORR = 21
CMAP_LITCORR = 22

# Precompute glyph values for stairs and altar
S_UPSTAIR = GLYPH_CMAP_OFF + CMAP_UPSTAIR
S_DNSTAIR = GLYPH_CMAP_OFF + CMAP_DNSTAIR
S_UPLADDER = GLYPH_CMAP_OFF + CMAP_UPLADDER
S_DNLADDER = GLYPH_CMAP_OFF + CMAP_DNLADDER
S_ALTAR = GLYPH_CMAP_OFF + CMAP_ALTAR
S_FOUNTAIN = GLYPH_CMAP_OFF + CMAP_FOUNTAIN

# BLStats field indices (NLE 0.9+, 27-element vector)
BL_X = 0
BL_Y = 1
BL_STR25 = 2     # strength (25-scale: 3-18, 18/01-18/99 = 19-117, 18/** = 118+)
BL_STR125 = 3    # strength percentage (for display)
BL_DEX = 4
BL_CON = 5
BL_INT = 6
BL_WIS = 7
BL_CHA = 8
BL_SCORE = 9
BL_HP = 10
BL_HPMAX = 11
BL_DEPTH = 12
BL_GOLD = 13
BL_ENE = 14       # power (mana)
BL_ENEMAX = 15    # max power
BL_AC = 16        # armor class
BL_HD = 17        # monster level (hit dice if polymorphed)
BL_XP = 18        # experience level
BL_EXP = 19       # experience points
BL_TIME = 20      # turn number
BL_HUNGER = 21
BL_CAP = 22       # encumbrance
BL_DNUM = 23      # dungeon number
BL_DLEVEL = 24    # dungeon level
BL_CONDITION = 25 # condition bitmask
BL_ALIGN = 26     # alignment (-1 chaotic, 0 neutral, 1 lawful)

# Condition bitmask flags (from NLE, matches botl.h)
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
COND_LEV = 0x00000400
COND_FLY = 0x00000800
COND_RIDE = 0x00001000

# Condition flag name mapping
_COND_FLAGS = {
    "stoned": COND_STONE,
    "slimed": COND_SLIME,
    "strangled": COND_STRNGL,
    "foodpois": COND_FOODPOIS,
    "termill": COND_TERMILL,
    "blind": COND_BLIND,
    "deaf": COND_DEAF,
    "stunned": COND_STUN,
    "confused": COND_CONF,
    "hallucinating": COND_HALLU,
    "levitating": COND_LEV,
    "flying": COND_FLY,
    "riding": COND_RIDE,
}

# Hunger state enum
HUNGER_LABELS = {
    0: "satiated",
    1: "not_hungry",
    2: "hungry",
    3: "weak",
    4: "fainting",
    5: "fainted",
    6: "starved",
}

# Alignment
ALIGN_LABELS = {
    -1: "chaotic",
    0: "neutral",
    1: "lawful",
}

MAP_H = 21
MAP_W = 79

# ============================================================
# Monster name table (loaded from monsters.json)
# ============================================================

_MONSTER_NAMES: Optional[list[str]] = None


def _load_monster_names() -> list[str]:
    """Load monster names from parsed JSON. Returns list indexed by mon_id."""
    global _MONSTER_NAMES
    if _MONSTER_NAMES is not None:
        return _MONSTER_NAMES

    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "parsed", "monsters.json",
    )
    try:
        with open(json_path) as f:
            monsters = json.load(f)
        # monsters.json has 384 entries; NLE uses first 381 (NUMMONS).
        _MONSTER_NAMES = [m["name"] for m in monsters[:NUMMONS]]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Fallback: generic names
        _MONSTER_NAMES = [f"monster_{i}" for i in range(NUMMONS)]
    return _MONSTER_NAMES


def glyph_to_monster_name(glyph: int) -> Optional[str]:
    """Convert a glyph to a monster name, or None if not a monster glyph."""
    mon_id = glyph_to_mon_id(glyph)
    if mon_id is None:
        return None
    names = _load_monster_names()
    if 0 <= mon_id < len(names):
        return names[mon_id]
    return f"monster_{mon_id}"


def glyph_to_mon_id(glyph: int) -> Optional[int]:
    """Extract monster ID from any monster-category glyph."""
    if GLYPH_MON_OFF <= glyph < GLYPH_PET_OFF:
        return glyph - GLYPH_MON_OFF
    if GLYPH_PET_OFF <= glyph < GLYPH_INVIS_OFF:
        return glyph - GLYPH_PET_OFF
    if GLYPH_DETECT_OFF <= glyph < GLYPH_BODY_OFF:
        return glyph - GLYPH_DETECT_OFF
    if GLYPH_RIDDEN_OFF <= glyph < GLYPH_OBJ_OFF:
        return glyph - GLYPH_RIDDEN_OFF
    return None


def glyph_is_monster(glyph: int) -> bool:
    """True if the glyph represents a monster (wild, pet, detected, ridden)."""
    return glyph_to_mon_id(glyph) is not None


def glyph_is_pet(glyph: int) -> bool:
    return GLYPH_PET_OFF <= glyph < GLYPH_INVIS_OFF


def glyph_is_stairs_up(glyph: int) -> bool:
    return glyph == S_UPSTAIR or glyph == S_UPLADDER


def glyph_is_stairs_down(glyph: int) -> bool:
    return glyph == S_DNSTAIR or glyph == S_DNLADDER


def glyph_is_altar(glyph: int) -> bool:
    return glyph == S_ALTAR


def glyph_is_fountain(glyph: int) -> bool:
    return glyph == S_FOUNTAIN


# ============================================================
# Condition parsing
# ============================================================

def parse_conditions(bitmask: int) -> set[str]:
    """Parse the BL_CONDITION bitmask into a set of condition names."""
    conds = set()
    bitmask = int(bitmask)
    for name, flag in _COND_FLAGS.items():
        if bitmask & flag:
            conds.add(name)
    return conds


# ============================================================
# GameState
# ============================================================

@dataclass
class MonsterInfo:
    name: str
    row: int
    col: int
    mon_id: int
    is_pet: bool


@dataclass
class GameState:
    """Parsed semantic game state from NLE observations.

    Updated each turn via update(obs). Maintains persistent state
    (resistances, message history) across turns.
    """
    # Player vitals
    turn: int = 0
    dlevel: int = 1
    dnum: int = 0
    depth: int = 1
    hp: int = 0
    max_hp: int = 0
    pw: int = 0
    max_pw: int = 0
    ac: int = 10
    xlevel: int = 1
    xp: int = 0
    gold: int = 0
    score: int = 0
    strength: int = 0
    dexterity: int = 0
    constitution: int = 0
    intelligence: int = 0
    wisdom: int = 0
    charisma: int = 0
    encumbrance: int = 0

    # Status
    hunger_state: str = "not_hungry"
    alignment: str = "neutral"
    conditions: set = field(default_factory=set)

    # Persistent knowledge
    resistances: set = field(default_factory=set)

    # Position
    position: tuple[int, int] = (0, 0)  # (row, col)

    @property
    def py(self) -> int:
        return self.position[0]

    @property
    def px(self) -> int:
        return self.position[1]

    # Surroundings
    adjacent_monsters: list = field(default_factory=list)
    visible_monsters: list = field(default_factory=list)
    on_stairs_down: bool = False
    on_stairs_up: bool = False
    on_altar: bool = False
    on_fountain: bool = False

    # Messages
    messages: list = field(default_factory=list)
    _max_messages: int = field(default=20, repr=False)

    # Inventory (parsed from inv_strs if available)
    inventory: dict = field(default_factory=dict)

    # Raw observation cache (for downstream modules that need it)
    _glyphs: Optional[np.ndarray] = field(default=None, repr=False)
    _blstats: Optional[np.ndarray] = field(default=None, repr=False)

    def update(self, obs: dict) -> None:
        """Update state from an NLE observation dict.

        Expected keys: glyphs (21x79 int), blstats (27-element float vector),
        message (256 bytes). Optional: inv_strs, inv_letters, inv_oclasses,
        inv_glyphs.
        """
        self._parse_blstats(obs)
        self._parse_message(obs)
        self._parse_glyphs(obs)
        self._parse_inventory(obs)

    def _parse_blstats(self, obs: dict) -> None:
        bl = obs.get("blstats")
        if bl is None:
            return
        bl = np.asarray(bl, dtype=np.float32).ravel()
        self._blstats = bl
        n = len(bl)

        # Position
        if n > BL_Y:
            self.position = (int(bl[BL_Y]), int(bl[BL_X]))

        # Vitals
        if n > BL_HPMAX:
            self.hp = int(bl[BL_HP])
            self.max_hp = int(bl[BL_HPMAX])
        if n > BL_ENEMAX:
            self.pw = int(bl[BL_ENE])
            self.max_pw = int(bl[BL_ENEMAX])
        if n > BL_AC:
            self.ac = int(bl[BL_AC])
        if n > BL_XP:
            self.xlevel = int(bl[BL_XP])
        if n > BL_EXP:
            self.xp = int(bl[BL_EXP])
        if n > BL_GOLD:
            self.gold = int(bl[BL_GOLD])
        if n > BL_SCORE:
            self.score = int(bl[BL_SCORE])
        if n > BL_DEPTH:
            self.depth = int(bl[BL_DEPTH])

        # Ability scores
        if n > BL_STR25:
            self.strength = int(bl[BL_STR25])
        if n > BL_DEX:
            self.dexterity = int(bl[BL_DEX])
        if n > BL_CON:
            self.constitution = int(bl[BL_CON])
        if n > BL_INT:
            self.intelligence = int(bl[BL_INT])
        if n > BL_WIS:
            self.wisdom = int(bl[BL_WIS])
        if n > BL_CHA:
            self.charisma = int(bl[BL_CHA])

        # Turn
        if n > BL_TIME:
            self.turn = int(bl[BL_TIME])

        # Hunger
        if n > BL_HUNGER:
            self.hunger_state = HUNGER_LABELS.get(int(bl[BL_HUNGER]), "unknown")

        # Encumbrance
        if n > BL_CAP:
            self.encumbrance = int(bl[BL_CAP])

        # Dungeon
        if n > BL_DNUM:
            self.dnum = int(bl[BL_DNUM])
        if n > BL_DLEVEL:
            self.dlevel = int(bl[BL_DLEVEL])

        # Conditions
        if n > BL_CONDITION:
            self.conditions = parse_conditions(bl[BL_CONDITION])

        # Alignment
        if n > BL_ALIGN:
            self.alignment = ALIGN_LABELS.get(int(bl[BL_ALIGN]), "neutral")

    def _parse_message(self, obs: dict) -> None:
        msg_raw = obs.get("message")
        if msg_raw is None:
            return
        msg_bytes = np.asarray(msg_raw, dtype=np.uint8)
        # Decode bytes, strip null padding
        try:
            msg_str = bytes(msg_bytes).decode("ascii", errors="replace").rstrip("\x00").strip()
        except Exception:
            msg_str = ""
        if msg_str:
            self.messages.append(msg_str)
            if len(self.messages) > self._max_messages:
                self.messages = self.messages[-self._max_messages:]

    def _parse_glyphs(self, obs: dict) -> None:
        glyphs = obs.get("glyphs")
        if glyphs is None:
            return
        glyphs = np.asarray(glyphs, dtype=np.int16)
        if glyphs.ndim == 1:
            glyphs = glyphs.reshape(MAP_H, MAP_W)
        self._glyphs = glyphs
        row, col = self.position

        # Player tile features
        player_glyph = int(glyphs[row, col]) if 0 <= row < MAP_H and 0 <= col < MAP_W else -1
        self.on_stairs_up = glyph_is_stairs_up(player_glyph)
        self.on_stairs_down = glyph_is_stairs_down(player_glyph)
        self.on_altar = glyph_is_altar(player_glyph)
        self.on_fountain = glyph_is_fountain(player_glyph)

        # Note: the player's tile usually shows the player's own glyph (a
        # monster glyph), so stairs/altar detection from the player tile
        # only works if the player glyph is NOT occupying the cell in the
        # observation. NLE does place the player glyph over dungeon
        # features, so we also check the feature from blstats depth or
        # from message context. For a more reliable check, we scan the
        # 3x3 neighborhood for the feature under the player.
        #
        # In practice the player glyph IS at (row, col), so we won't see
        # stairs beneath. We check the message for stair indicators instead
        # and scan neighborhood for the feature.
        if not (self.on_stairs_up or self.on_stairs_down or self.on_altar or self.on_fountain):
            # The player glyph is on top. Check messages for "staircase"
            # or "altar" indicators from the previous observation.
            if self.messages:
                last = self.messages[-1].lower()
                if "staircase up" in last or "ladder up" in last:
                    self.on_stairs_up = True
                elif "staircase down" in last or "ladder down" in last:
                    self.on_stairs_down = True
                elif "altar" in last and "here" in last:
                    self.on_altar = True
                elif "fountain" in last and "here" in last:
                    self.on_fountain = True

        # Scan for monsters
        self.visible_monsters = []
        self.adjacent_monsters = []
        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(glyphs[r, c])
                if r == row and c == col:
                    continue  # skip player's own tile
                if glyph_is_monster(g):
                    name = glyph_to_monster_name(g) or "unknown"
                    mon_id = glyph_to_mon_id(g)
                    pet = glyph_is_pet(g)
                    info = MonsterInfo(
                        name=name, row=r, col=c,
                        mon_id=mon_id if mon_id is not None else -1,
                        is_pet=pet,
                    )
                    self.visible_monsters.append(info)
                    if abs(r - row) <= 1 and abs(c - col) <= 1:
                        self.adjacent_monsters.append(info)

    def _parse_inventory(self, obs: dict) -> None:
        inv_strs = obs.get("inv_strs")
        inv_letters = obs.get("inv_letters")
        if inv_strs is None or inv_letters is None:
            return

        self.inventory = {}
        for i, letter_val in enumerate(inv_letters):
            letter = int(letter_val)
            if letter == 0:
                continue
            letter_chr = chr(letter)
            # inv_strs entries are byte arrays
            raw = inv_strs[i]
            try:
                item_str = bytes(np.asarray(raw, dtype=np.uint8)).decode(
                    "ascii", errors="replace"
                ).rstrip("\x00").strip()
            except Exception:
                item_str = ""
            if item_str:
                self.inventory[letter_chr] = item_str

    # ---- Convenience properties ----

    @property
    def hp_fraction(self) -> float:
        if self.max_hp <= 0:
            return 0.0
        return self.hp / self.max_hp

    @property
    def is_critical(self) -> bool:
        """HP below 20% or absolute HP <= 5."""
        return self.hp <= 5 or self.hp_fraction < 0.2

    @property
    def has_adjacent_monsters(self) -> bool:
        return len(self.adjacent_monsters) > 0

    @property
    def num_adjacent_monsters(self) -> int:
        return len(self.adjacent_monsters)

    @property
    def num_visible_monsters(self) -> int:
        return len(self.visible_monsters)

    @property
    def is_blind(self) -> bool:
        return "blind" in self.conditions

    @property
    def is_confused(self) -> bool:
        return "confused" in self.conditions

    @property
    def is_stunned(self) -> bool:
        return "stunned" in self.conditions

    @property
    def is_hallucinating(self) -> bool:
        return "hallucinating" in self.conditions

    @property
    def is_stoned(self) -> bool:
        return "stoned" in self.conditions

    @property
    def is_sick(self) -> bool:
        return "foodpois" in self.conditions or "termill" in self.conditions

    @property
    def last_message(self) -> str:
        return self.messages[-1] if self.messages else ""


# ============================================================
# Test
# ============================================================

def _run_test():
    """Create a GameState and feed it a synthetic observation dict."""
    print("=== obs_parser test ===")

    gs = GameState()
    assert gs.hp == 0
    assert gs.hunger_state == "not_hungry"
    assert gs.conditions == set()

    # Build a fake blstats (27 elements)
    bl = np.zeros(27, dtype=np.float32)
    bl[BL_X] = 40        # col
    bl[BL_Y] = 10        # row
    bl[BL_STR25] = 18
    bl[BL_STR125] = 100
    bl[BL_DEX] = 14
    bl[BL_CON] = 16
    bl[BL_INT] = 10
    bl[BL_WIS] = 12
    bl[BL_CHA] = 8
    bl[BL_SCORE] = 1234
    bl[BL_HP] = 42
    bl[BL_HPMAX] = 50
    bl[BL_DEPTH] = 3
    bl[BL_GOLD] = 200
    bl[BL_ENE] = 15
    bl[BL_ENEMAX] = 20
    bl[BL_AC] = 4
    bl[BL_HD] = 0
    bl[BL_XP] = 7
    bl[BL_EXP] = 3500
    bl[BL_TIME] = 1500
    bl[BL_HUNGER] = 2     # hungry
    bl[BL_CAP] = 0
    bl[BL_DNUM] = 0
    bl[BL_DLEVEL] = 5
    bl[BL_CONDITION] = COND_CONF | COND_BLIND
    bl[BL_ALIGN] = 1      # lawful

    # Build a fake glyph map (fill with room floor glyph, not 0 which is giant ant)
    glyphs = np.full((MAP_H, MAP_W), GLYPH_CMAP_OFF + CMAP_ROOM, dtype=np.int16)
    # Put a monster adjacent to player (mon_id=0 = giant ant, glyph = 0)
    glyphs[9, 41] = GLYPH_MON_OFF + 0    # giant ant at (9, 41)
    # Put a pet at (10, 39)
    glyphs[10, 39] = GLYPH_PET_OFF + 5   # pet monster 5
    # Put stairs down at player position (won't be visible because
    # player glyph is on top, but we test via message)
    glyphs[10, 40] = GLYPH_MON_OFF + 100  # player's own glyph (valkyrie-ish)
    # Put a far monster
    glyphs[5, 60] = GLYPH_MON_OFF + 50

    # Build a fake message
    msg = np.zeros(256, dtype=np.uint8)
    text = b"There is a staircase down here."
    msg[:len(text)] = list(text)

    obs = {
        "glyphs": glyphs,
        "blstats": bl,
        "message": msg,
    }

    gs.update(obs)

    # Verify blstats parsing
    assert gs.position == (10, 40), f"position: {gs.position}"
    assert gs.hp == 42, f"hp: {gs.hp}"
    assert gs.max_hp == 50
    assert gs.pw == 15
    assert gs.max_pw == 20
    assert gs.ac == 4
    assert gs.xlevel == 7
    assert gs.gold == 200
    assert gs.score == 1234
    assert gs.turn == 1500
    assert gs.depth == 3
    assert gs.dlevel == 5
    assert gs.strength == 18
    assert gs.dexterity == 14
    assert gs.hunger_state == "hungry", f"hunger: {gs.hunger_state}"
    assert gs.alignment == "lawful", f"alignment: {gs.alignment}"
    assert "confused" in gs.conditions, f"conditions: {gs.conditions}"
    assert "blind" in gs.conditions
    assert gs.is_confused
    assert gs.is_blind
    assert not gs.is_stunned

    # Verify message parsing
    assert len(gs.messages) == 1
    assert "staircase down" in gs.messages[0]
    assert gs.on_stairs_down, "should detect stairs down from message"

    # Verify monster parsing
    assert gs.num_visible_monsters == 3, f"visible: {gs.num_visible_monsters}"
    adj = gs.adjacent_monsters
    assert len(adj) == 2, f"adjacent: {len(adj)}"
    # giant ant should be adjacent
    ant_found = any(m.name == "giant ant" and m.row == 9 and m.col == 41 for m in adj)
    assert ant_found, f"no giant ant in adjacent: {adj}"
    # pet should be adjacent
    pet_found = any(m.is_pet and m.row == 10 and m.col == 39 for m in adj)
    assert pet_found, f"no pet in adjacent: {adj}"

    # HP fraction
    assert abs(gs.hp_fraction - 0.84) < 0.01
    assert not gs.is_critical

    # Test critical HP
    bl2 = bl.copy()
    bl2[BL_HP] = 3
    obs2 = {"glyphs": glyphs, "blstats": bl2, "message": np.zeros(256, dtype=np.uint8)}
    gs.update(obs2)
    assert gs.is_critical

    # Test inventory parsing
    inv_letters = np.zeros(55, dtype=np.uint8)
    inv_letters[0] = ord('a')
    inv_letters[1] = ord('b')
    inv_strs = np.zeros((55, 80), dtype=np.uint8)
    item_a = b"+3 long sword (weapon in hand)"
    item_b = b"a blessed +1 small shield (being worn)"
    inv_strs[0, :len(item_a)] = list(item_a)
    inv_strs[1, :len(item_b)] = list(item_b)
    obs3 = {
        "glyphs": glyphs,
        "blstats": bl,
        "message": np.zeros(256, dtype=np.uint8),
        "inv_letters": inv_letters,
        "inv_strs": inv_strs,
    }
    gs.update(obs3)
    assert "a" in gs.inventory
    assert "long sword" in gs.inventory["a"]
    assert "b" in gs.inventory
    assert "shield" in gs.inventory["b"]

    # Test condition parsing standalone
    conds = parse_conditions(COND_STONE | COND_HALLU | COND_STUN)
    assert conds == {"stoned", "hallucinating", "stunned"}, f"conds: {conds}"

    # Test glyph helpers
    assert glyph_is_monster(GLYPH_MON_OFF + 50)
    assert glyph_is_monster(GLYPH_PET_OFF + 10)
    assert glyph_is_pet(GLYPH_PET_OFF + 10)
    assert not glyph_is_pet(GLYPH_MON_OFF + 10)
    assert not glyph_is_monster(GLYPH_CMAP_OFF + 5)
    assert glyph_is_stairs_down(S_DNSTAIR)
    assert glyph_is_stairs_up(S_UPSTAIR)
    assert glyph_is_altar(S_ALTAR)

    # Test monster name lookup
    name = glyph_to_monster_name(GLYPH_MON_OFF + 0)
    assert name == "giant ant", f"name: {name}"

    # Test message history cap
    gs2 = GameState(_max_messages=3)
    for i in range(10):
        msg = np.zeros(256, dtype=np.uint8)
        text = f"message {i}".encode()
        msg[:len(text)] = list(text)
        gs2.update({"message": msg})
    assert len(gs2.messages) == 3
    assert "message 9" in gs2.messages[-1]

    print("All tests passed.")


if __name__ == "__main__":
    _run_test()
