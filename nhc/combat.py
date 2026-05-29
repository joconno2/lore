"""Combat threat evaluation for NetHack agents.

Loads parsed monster data and corpse effects, provides threat assessment
for individual monsters, corpse eating decisions, and multi-monster
tactical evaluation. Works purely from the JSON knowledge base; does
not require NLE at runtime.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================
# Data types
# ============================================================

@dataclass
class ThreatReport:
    danger_level: int                    # 1-10
    special_attacks: list[str]           # human-readable warnings
    required_resistances: list[str]      # resistances player should have
    elbereth_effective: bool
    ranged_preferred: bool
    instakill_risk: bool
    recommended_action: str              # melee | ranged | elbereth | flee | use_item


@dataclass
class CorpseReport:
    safe_to_eat: bool
    beneficial_intrinsic: Optional[str]
    intrinsic_probability: float         # mlevel / 15, capped at 1.0
    special_effect: Optional[str]
    priority: int                        # 1-10


@dataclass
class RoomThreatReport:
    total_danger: float
    target_list: list[str]               # kill order (highest priority first)
    recommended_approach: str
    flee_recommended: bool


# ============================================================
# Attack / damage type mappings
# ============================================================

# Maps damage type codes to human-readable special attack descriptions.
_SPECIAL_ATTACK_LABELS: dict[str, str] = {
    "AD_PLYS": "paralysis on hit",
    "AD_STON": "stoning touch",
    "AD_DRLI": "level drain",
    "AD_DRIN": "brain eating",
    "AD_DRST": "poison sting",
    "AD_DRDX": "dexterity drain (poison)",
    "AD_DRCO": "constitution drain (poison)",
    "AD_SLEE": "sleep attack",
    "AD_BLND": "blindness attack",
    "AD_STUN": "stun attack",
    "AD_HALU": "hallucination attack",
    "AD_CONF": "confusion attack",
    "AD_ACID": "acid attack",
    "AD_FIRE": "fire attack",
    "AD_COLD": "cold attack",
    "AD_ELEC": "shock attack",
    "AD_DETH": "touch of death",
    "AD_PEST": "terminal illness",
    "AD_FAMN": "famine (food drain)",
    "AD_SLIM": "sliming",
    "AD_DISN": "disintegration",
    "AD_RUST": "rust (destroys metal)",
    "AD_CORR": "corrosion (destroys armor)",
    "AD_DCAY": "decay (destroys organic items)",
    "AD_DREN": "energy drain",
    "AD_SEDU": "seduction (steals armor)",
    "AD_SGLD": "steals gold",
    "AD_SITM": "steals items",
    "AD_TLPT": "teleportation",
    "AD_WERE": "lycanthropy",
    "AD_WRAP": "drowning grab",
    "AD_DGST": "digestion (swallow)",
    "AD_DISE": "disease",
    "AD_ENCH": "disenchant",
    "AD_SLOW": "slow",
    "AD_LEGS": "leg wound",
    "AD_STCK": "sticky (holds you)",
    "AD_CURS": "curse items",
}

# Damage types that require specific resistances.
_DAMAGE_TO_RESISTANCE: dict[str, str] = {
    "AD_FIRE": "fire resistance",
    "AD_COLD": "cold resistance",
    "AD_ELEC": "shock resistance",
    "AD_DRST": "poison resistance",
    "AD_DRDX": "poison resistance",
    "AD_DRCO": "poison resistance",
    "AD_SLEE": "sleep resistance",
    "AD_ACID": "acid resistance",
    "AD_STON": "stoning resistance",
    "AD_DISN": "disintegration resistance",
    "AD_DETH": "magic resistance",
    "AD_SLIM": "fire resistance",      # burn it off, or unchanging
    "AD_PEST": "sickness resistance",
    "AD_DRLI": "drain resistance",
}

# Damage types that represent instakill or near-instakill threats.
_INSTAKILL_DAMAGE = {"AD_STON", "AD_DETH", "AD_SLIM", "AD_DISN", "AD_DGST"}

# Monsters where ranged combat is strongly preferred.
_RANGED_PREFERRED_NAMES = {
    "floating eye", "cockatrice", "chickatrice",
    "green slime", "Medusa",
    "rust monster", "disenchanter",
    "acid blob", "brown mold", "yellow mold",
    "blue jelly", "gelatinous cube",
    "black pudding", "brown pudding",
}

# ============================================================
# Elbereth logic
# ============================================================

# Monsters immune to Elbereth by name.
_ELBERETH_IMMUNE_NAMES = {
    # Riders
    "Death", "Pestilence", "Famine",
    # Quest nemeses and special
    "Wizard of Yendor", "Medusa",
    # Minotaur
    "minotaur",
    # Named demons
    "Demogorgon", "Asmodeus", "Baalzebub", "Orcus", "Juiblex",
    "Yeenoghu", "Dispater", "Geryon",
    # Shopkeepers, guards, priests
    "shopkeeper", "guard", "aligned priest", "high priest",
    # Archon
    "Archon",
    # Quest leaders/nemeses (all @-class humans are covered by symbol check)
}

# Symbol classes that ignore Elbereth.
# @ = humans/elves/etc, A = angels/archons
_ELBERETH_IMMUNE_SYMBOLS = {"@", "A"}


def _is_elbereth_immune(name: str, symbol: str, flags2: list[str]) -> bool:
    """Check if a monster ignores Elbereth.

    Immune: all @ monsters, all A monsters, minotaurs, Riders,
    quest nemeses, named demons. Also blind monsters in practice,
    but blindness is runtime state and not in static data.
    """
    if name in _ELBERETH_IMMUNE_NAMES:
        return True
    if symbol in _ELBERETH_IMMUNE_SYMBOLS:
        return True
    return False


# ============================================================
# Corpse knowledge
# ============================================================

# Monster name -> (intrinsic_name, special_effect)
# intrinsic_name is the resistance/ability gained from eating the corpse.
# special_effect is a non-intrinsic effect (gain level, teleportitis, etc.)
_CORPSE_INTRINSICS: dict[str, tuple[Optional[str], Optional[str]]] = {
    # Telepathy
    "floating eye": ("telepathy", None),
    # Invisibility
    "stalker": ("invisibility", None),
    # Fire resistance
    "fire ant": ("fire resistance", None),
    "fire giant": ("fire resistance", None),
    "red dragon": ("fire resistance", None),
    "baby red dragon": ("fire resistance", None),
    "hell hound pup": ("fire resistance", None),
    "hell hound": ("fire resistance", None),
    "pyrolisk": ("fire resistance", None),
    "red naga": ("fire resistance", None),
    "red naga hatchling": ("fire resistance", None),
    "Lord Surtur": ("fire resistance", None),
    # Cold resistance
    "frost giant": ("cold resistance", None),
    "blue dragon": ("cold resistance", None),
    "baby blue dragon": ("cold resistance", None),
    "winter wolf cub": ("cold resistance", None),
    "winter wolf": ("cold resistance", None),
    "brown mold": ("cold resistance", None),
    "white dragon": ("cold resistance", None),
    "baby white dragon": ("cold resistance", None),
    "blue jelly": ("cold resistance", None),
    # Shock resistance
    "storm giant": ("shock resistance", None),
    "blue jelly": ("cold resistance", None),  # actually cold, not shock
    "electric eel": ("shock resistance", None),
    # Disintegration resistance
    "black dragon": ("disintegration resistance", None),
    "baby black dragon": ("disintegration resistance", None),
    # Poison resistance
    "green dragon": ("poison resistance", None),
    "baby green dragon": ("poison resistance", None),
    "killer bee": ("poison resistance", None),
    "scorpion": ("poison resistance", None),
    "pit viper": ("poison resistance", None),
    "cobra": ("poison resistance", None),
    "water moccasin": ("poison resistance", None),
    "garter snake": ("poison resistance", None),
    # Sleep resistance
    "orange dragon": ("sleep resistance", None),
    "baby orange dragon": ("sleep resistance", None),
    # Special effects
    "wraith": (None, "gain level"),
    "newt": (None, "mana restore (2/3 chance)"),
    "tengu": ("teleport control", "teleportitis"),
    "quantum mechanic": (None, "toggle speed"),
    "lizard": (None, "cures stoning, reduces stun/confusion"),
    "lichen": (None, "safe food, never rots"),
    "nurse": (None, "full HP heal"),
    "mind flayer": ("telepathy", "+1 INT (50% chance)"),
    "master mind flayer": ("telepathy", "+1 INT (50% chance)"),
    # Strength
    "giant": (None, "strength gain"),
    "hill giant": (None, "strength gain"),
    "stone giant": (None, "strength gain"),
    "fire giant": ("fire resistance", "strength gain"),
    "frost giant": ("cold resistance", "strength gain"),
    "storm giant": ("shock resistance", "strength gain"),
    # Polymorph
    "chameleon": (None, "polymorph self"),
    "doppelganger": (None, "polymorph self"),
    "sandestin": (None, "polymorph self"),
    # Lycanthropy
    "wererat": (None, "lycanthropy (wererat)"),
    "werejackal": (None, "lycanthropy (werejackal)"),
    "werewolf": (None, "lycanthropy (werewolf)"),
    # Hallucination
    "violet fungus": (None, "+200 turns hallucination"),
    # Disenchant
    "disenchanter": (None, "strips a random intrinsic"),
    # Speed toggle
    "quantum mechanic": (None, "toggle speed"),
}

# Poisonous corpses (need poison resistance to eat safely).
_POISONOUS_NAMES = {
    "killer bee", "scorpion", "pit viper", "cobra", "water moccasin",
    "asp", "python", "giant spider", "black naga", "golden naga hatchling",
    "baby purple worm", "purple worm", "quasit", "rabid rat", "garter snake",
    "green slime",
}

# Petrifying corpses (never eat without stoning resistance).
_PETRIFYING_NAMES = {"cockatrice", "chickatrice", "Medusa"}

# Acidic corpses (hurt without acid resistance).
_ACIDIC_NAMES = {
    "acid blob", "yellow light", "gelatinous cube", "blue jelly", "ochre jelly",
}

# Corpses that are always dangerous regardless of resistances.
_NEVER_EAT = {
    "green slime",  # sliming, no simple resistance
    "Death", "Pestilence", "Famine",  # instantly fatal
}

# Corpses that give aggravate monster (dogs/cats).
_AGGRAVATE_CORPSES = {
    "little dog", "dog", "large dog", "kitten", "housecat", "large cat",
}


# ============================================================
# ThreatDB
# ============================================================

class ThreatDB:
    """Monster threat database. Loads parsed JSON data and provides
    threat assessment, Elbereth checks, corpse evaluation, and
    multi-monster room analysis.
    """

    def __init__(self, monsters_path: Optional[str] = None,
                 corpse_path: Optional[str] = None):
        base = Path(__file__).resolve().parent.parent / "data" / "parsed"
        if monsters_path is None:
            monsters_path = str(base / "monsters.json")
        if corpse_path is None:
            corpse_path = str(base / "corpse_effects.json")

        with open(monsters_path) as f:
            raw = json.load(f)
        with open(corpse_path) as f:
            self.corpse_effects = json.load(f)

        self._monsters: dict[str, dict] = {}
        for m in raw:
            self._monsters[m["name"]] = m

    def _get_monster(self, name: str) -> Optional[dict]:
        return self._monsters.get(name)

    def _active_attacks(self, monster: dict) -> list[dict]:
        """Return attacks that are actually used.

        Includes AT_NONE passive attacks when the damage type is
        non-physical (e.g., floating eye's AD_PLYS passive).
        """
        return [a for a in monster["attacks"]
                if a["attack_type"] != "AT_NONE"
                or a["damage_type"] != "AD_PHYS"]

    def _max_damage(self, monster: dict) -> int:
        """Max single-round damage from all active attacks."""
        return sum(a["dice"] * a["sides"] for a in self._active_attacks(monster))

    # ----------------------------------------------------------
    # Threat assessment
    # ----------------------------------------------------------

    def assess_threat(self, monster_name: str,
                      player_state: dict) -> ThreatReport:
        """Assess threat of a single monster given current player state.

        player_state keys:
            hp, max_hp, ac, level, speed, resistances (set of str),
            equipment (dict), position, has_elbereth_source (bool)
        """
        mon = self._get_monster(monster_name)
        if mon is None:
            return self._unknown_threat(monster_name, player_state)

        special_attacks = []
        required_res = []
        instakill = False
        ranged_pref = False

        # Scan attacks for special damage types.
        for atk in self._active_attacks(mon):
            dt = atk["damage_type"]
            if dt in _SPECIAL_ATTACK_LABELS:
                special_attacks.append(_SPECIAL_ATTACK_LABELS[dt])
            if dt in _DAMAGE_TO_RESISTANCE:
                res = _DAMAGE_TO_RESISTANCE[dt]
                if res not in required_res:
                    required_res.append(res)
            if dt in _INSTAKILL_DAMAGE:
                instakill = True

        # Breath and magic attacks also deserve a note.
        for atk in self._active_attacks(mon):
            at = atk["attack_type"]
            if at == "AT_BREA":
                special_attacks.append("breath weapon")
            elif at == "AT_MAGC":
                special_attacks.append("casts spells")
            elif at == "AT_GAZE":
                special_attacks.append("gaze attack")
            elif at == "AT_ENGL":
                special_attacks.append("engulfing attack")
            elif at == "AT_EXPL":
                special_attacks.append("explodes on death")

        # Deduplicate.
        special_attacks = list(dict.fromkeys(special_attacks))
        required_res = list(dict.fromkeys(required_res))

        # Elbereth
        elbereth_ok = not _is_elbereth_immune(
            monster_name, mon["symbol"], mon.get("flags2", []))

        # Ranged preference
        ranged_pref = monster_name in _RANGED_PREFERRED_NAMES
        # Also prefer ranged for anything with passive damage types.
        if any(a["attack_type"] == "AT_NONE" and a["damage_type"] != "AD_PHYS"
               and a["dice"] > 0 for a in mon["attacks"]):
            ranged_pref = True

        # Danger level (1-10).
        danger = self._compute_danger(mon, player_state, instakill)

        # Recommended action.
        action = self._recommend_action(
            mon, monster_name, player_state, danger,
            elbereth_ok, ranged_pref, instakill)

        return ThreatReport(
            danger_level=danger,
            special_attacks=special_attacks,
            required_resistances=required_res,
            elbereth_effective=elbereth_ok,
            ranged_preferred=ranged_pref,
            instakill_risk=instakill,
            recommended_action=action,
        )

    def _compute_danger(self, mon: dict, ps: dict, instakill: bool) -> int:
        """Compute danger level 1-10."""
        if instakill:
            return 10

        diff = mon.get("difficulty", mon["level"])
        max_dmg = self._max_damage(mon)
        hp = ps.get("hp", 50)
        max_hp = ps.get("max_hp", 50)
        plevel = ps.get("level", 1)
        resistances = ps.get("resistances", set())

        # Base danger from difficulty relative to player level.
        if diff <= 0:
            base = 1
        elif diff <= plevel // 2:
            base = 2
        elif diff <= plevel:
            base = 3
        elif diff <= plevel * 1.5:
            base = 5
        elif diff <= plevel * 2:
            base = 7
        else:
            base = 8

        # Damage relative to player HP.
        if max_hp > 0 and max_dmg > 0:
            dmg_ratio = max_dmg / max_hp
            if dmg_ratio > 0.5:
                base = max(base, 7)
            elif dmg_ratio > 0.3:
                base = max(base, 5)

        # HP urgency: if player is low, everything is more dangerous.
        if max_hp > 0 and hp < max_hp * 0.25:
            base = min(10, base + 2)
        elif max_hp > 0 and hp < max_hp * 0.5:
            base = min(10, base + 1)

        # Special attack penalty: each unresisted special attack bumps danger.
        for atk in self._active_attacks(mon):
            dt = atk["damage_type"]
            if dt in _DAMAGE_TO_RESISTANCE:
                needed = _DAMAGE_TO_RESISTANCE[dt]
                if needed not in resistances:
                    base = min(10, base + 1)

        # Named extremely dangerous monsters.
        if mon["name"] in {n for n in (
            "arch-lich", "master lich", "Wizard of Yendor",
            "Demogorgon", "mind flayer", "master mind flayer",
            "minotaur", "titan", "balrog", "purple worm",
        )}:
            base = max(base, 7)

        return max(1, min(10, base))

    def _recommend_action(self, mon: dict, name: str, ps: dict,
                          danger: int, elbereth_ok: bool,
                          ranged_pref: bool, instakill: bool) -> str:
        """Pick a recommended action."""
        hp_ratio = ps.get("hp", 50) / max(1, ps.get("max_hp", 50))
        has_elbereth = ps.get("has_elbereth_source", False)

        # Flee from instakill threats when unprepared.
        if instakill and danger >= 9:
            resistances = ps.get("resistances", set())
            # Check if player has the needed resistance.
            needed = set()
            for atk in self._active_attacks(mon):
                dt = atk["damage_type"]
                if dt in _INSTAKILL_DAMAGE and dt in _DAMAGE_TO_RESISTANCE:
                    needed.add(_DAMAGE_TO_RESISTANCE[dt])
            if needed and not needed.issubset(resistances):
                return "flee"

        # Flee if very low HP and high danger.
        if hp_ratio < 0.2 and danger >= 6:
            return "flee"

        # Elbereth if surrounded/dangerous and we have a source.
        if danger >= 7 and elbereth_ok and has_elbereth and hp_ratio < 0.4:
            return "elbereth"

        # Ranged for contact-dangerous monsters.
        if ranged_pref:
            return "ranged"

        # Use item for specific threats (e.g., wand of death for liches).
        if name in {"arch-lich", "master lich", "Wizard of Yendor"} and danger >= 8:
            return "use_item"

        # Default to melee for manageable threats.
        if danger <= 5:
            return "melee"

        # High danger but manageable.
        if danger <= 7:
            if elbereth_ok and has_elbereth:
                return "elbereth"
            return "melee"

        return "flee"

    def _unknown_threat(self, name: str, ps: dict) -> ThreatReport:
        """Conservative threat report for monsters not in our database."""
        hp_ratio = ps.get("hp", 50) / max(1, ps.get("max_hp", 50))
        danger = 6 if hp_ratio > 0.5 else 8
        return ThreatReport(
            danger_level=danger,
            special_attacks=["unknown monster (no data)"],
            required_resistances=[],
            elbereth_effective=True,  # assume it works
            ranged_preferred=False,
            instakill_risk=False,
            recommended_action="melee" if hp_ratio > 0.5 else "flee",
        )

    # ----------------------------------------------------------
    # Elbereth check
    # ----------------------------------------------------------

    def respects_elbereth(self, monster_name: str) -> bool:
        """Returns True if the monster will flee from Elbereth.
        Returns False for @-class, A-class, minotaurs, Riders,
        quest nemeses, named demons.

        Note: blind monsters also ignore Elbereth in practice,
        but blindness is runtime state not available here.
        """
        mon = self._get_monster(monster_name)
        if mon is None:
            # Unknown monsters: assume they respect it (conservative = safer).
            return True
        return not _is_elbereth_immune(
            monster_name, mon["symbol"], mon.get("flags2", []))

    # ----------------------------------------------------------
    # Corpse value
    # ----------------------------------------------------------

    def corpse_value(self, monster_name: str,
                     player_resistances: set[str]) -> CorpseReport:
        """Evaluate whether a corpse is worth eating."""
        mon = self._get_monster(monster_name)

        # Unknown monster: conservative.
        if mon is None:
            return CorpseReport(
                safe_to_eat=False,
                beneficial_intrinsic=None,
                intrinsic_probability=0.0,
                special_effect=None,
                priority=1,
            )

        name = monster_name
        level = mon["level"]
        resistances_conveyed = mon.get("resistances_conveyed", [])
        flags1 = mon.get("flags1", [])

        # Safety checks.
        safe = True
        if name in _NEVER_EAT:
            safe = False
        elif name in _PETRIFYING_NAMES:
            safe = "stoning resistance" in player_resistances
        elif name in _POISONOUS_NAMES or "M1_POIS" in flags1:
            if "poison resistance" not in player_resistances:
                safe = False
        elif name in _ACIDIC_NAMES or "M1_ACID" in flags1:
            if "acid resistance" not in player_resistances:
                # Acidic corpses deal damage but don't kill; still "unsafe".
                safe = False
        if name in _AGGRAVATE_CORPSES:
            safe = False  # aggravate monster is permanent and bad

        # Intrinsic and special effect.
        intrinsic = None
        special = None
        if name in _CORPSE_INTRINSICS:
            intrinsic, special = _CORPSE_INTRINSICS[name]
        else:
            # Fall back to resistances_conveyed from parsed data.
            res_map = {
                "MR_FIRE": "fire resistance",
                "MR_COLD": "cold resistance",
                "MR_ELEC": "shock resistance",
                "MR_SLEEP": "sleep resistance",
                "MR_POISON": "poison resistance",
                "MR_DISINT": "disintegration resistance",
                "MR_STONE": "stoning resistance",
            }
            for rc in resistances_conveyed:
                if rc in res_map:
                    intrinsic = res_map[rc]
                    break

        # Intrinsic gain probability: mlevel / 15, capped at 1.0.
        # Telepathy from floating eye is guaranteed (chance_denominator=1).
        if intrinsic == "telepathy":
            prob = 1.0
        elif intrinsic is not None:
            prob = min(1.0, level / 15.0)
        else:
            prob = 0.0

        # Priority: how urgently should the agent eat this corpse.
        priority = self._corpse_priority(
            name, intrinsic, special, player_resistances, safe)

        return CorpseReport(
            safe_to_eat=safe,
            beneficial_intrinsic=intrinsic,
            intrinsic_probability=prob,
            special_effect=special,
            priority=priority,
        )

    def _corpse_priority(self, name: str, intrinsic: Optional[str],
                         special: Optional[str],
                         player_res: set[str], safe: bool) -> int:
        """Compute eating priority 1-10."""
        if not safe:
            return 1

        # Wraith corpse: gain level is always top priority.
        if name == "wraith":
            return 10

        # Lizard: cures stoning, safe food, never rots. Always valuable.
        if name == "lizard":
            return 8

        # Lichen: safe food, never rots.
        if name == "lichen":
            return 5

        # If we'd gain an intrinsic we don't have, high priority.
        if intrinsic and intrinsic not in player_res:
            # Key resistances are more valuable.
            if intrinsic in {"poison resistance", "fire resistance",
                             "cold resistance", "disintegration resistance",
                             "telepathy"}:
                return 9
            if intrinsic in {"shock resistance", "sleep resistance"}:
                return 7
            return 6

        # If we already have the intrinsic, low priority (just food).
        if intrinsic and intrinsic in player_res:
            return 3

        # Special effects.
        if special:
            if "strength" in (special or ""):
                return 6
            if "mana" in (special or ""):
                return 4
            return 5

        # Generic safe corpse: just food.
        return 3

    # ----------------------------------------------------------
    # Room threat assessment
    # ----------------------------------------------------------

    def assess_room(self, visible_monsters: list[str],
                    player_state: dict) -> RoomThreatReport:
        """Assess the combined threat of all visible monsters."""
        if not visible_monsters:
            return RoomThreatReport(
                total_danger=0.0,
                target_list=[],
                recommended_approach="explore",
                flee_recommended=False,
            )

        reports: list[tuple[str, ThreatReport]] = []
        for name in visible_monsters:
            r = self.assess_threat(name, player_state)
            reports.append((name, r))

        # Total danger: sum of individual dangers, with diminishing returns.
        total = sum(r.danger_level for _, r in reports)

        # Sort by priority: instakill first, then highest danger, then
        # ranged-preferred (kill those from distance first).
        def sort_key(pair):
            name, r = pair
            return (
                -int(r.instakill_risk),
                -r.danger_level,
                -int(r.ranged_preferred),
            )
        reports.sort(key=sort_key)
        target_list = [name for name, _ in reports]

        # Determine tactical approach.
        hp_ratio = player_state.get("hp", 50) / max(1, player_state.get("max_hp", 50))
        has_elbereth = player_state.get("has_elbereth_source", False)
        any_instakill = any(r.instakill_risk for _, r in reports)
        any_ranged_pref = any(r.ranged_preferred for _, r in reports)
        max_danger = max(r.danger_level for _, r in reports)
        all_respect_elbereth = all(r.elbereth_effective for _, r in reports)

        flee = False
        if any_instakill and hp_ratio < 0.5:
            approach = "flee immediately"
            flee = True
        elif total >= 25 or (max_danger >= 8 and len(reports) >= 3):
            if has_elbereth and all_respect_elbereth:
                approach = "engrave Elbereth, then pick off threats"
            else:
                approach = "retreat to corridor for 1v1"
                flee = hp_ratio < 0.3
        elif len(reports) >= 3 and hp_ratio < 0.5:
            if has_elbereth and all_respect_elbereth:
                approach = "engrave Elbereth, then pick off threats"
            else:
                approach = "retreat to corridor for 1v1"
        elif len(reports) >= 3 and hp_ratio < 0.7:
            approach = "retreat to corridor for 1v1"
        elif any_ranged_pref:
            approach = "ranged attacks on contact-dangerous monsters first"
        elif has_elbereth and all_respect_elbereth and hp_ratio < 0.4:
            approach = "engrave Elbereth for breathing room"
        else:
            approach = "engage in melee, prioritize highest threat"

        if hp_ratio < 0.15 and max_danger >= 5:
            flee = True
            approach = "flee immediately, HP critical"

        return RoomThreatReport(
            total_danger=total,
            target_list=target_list,
            recommended_approach=approach,
            flee_recommended=flee,
        )
