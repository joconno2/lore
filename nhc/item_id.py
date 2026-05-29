"""Item identification constraint engine for NetHack.

Inspired by BotHack's core.logic approach. Tracks the space of possible
identities for each randomized appearance, narrowing via observations
(price, BUC, engrave effects, use effects, direct identification).

Constraint propagation: when an appearance is narrowed to one candidate,
that identity is eliminated from all other appearances in the same class.
This cascades until no further deductions are possible.

Uses data/parsed/items.json for ground truth price tables. Does NOT use
nhc.rules (wrong prices).
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "parsed", "items.json",
)

# Classes with randomized appearances that we track.
TRACKED_CLASSES = ("scroll", "potion", "ring", "wand", "amulet", "spellbook")


def _load_item_db(path: str = _DATA_PATH) -> dict:
    """Load items.json and build lookup structures.

    Returns dict with keys:
        items_by_class: {class_name: [item_dict, ...]}  (only real items, name != None)
        appearances_by_class: {class_name: [description, ...]}  (all descriptions including spares)
        price_groups: {class_name: {price: [item_name, ...]}}
        base_price_of: {(class_name, item_name): price}
    """
    with open(path) as f:
        raw = json.load(f)

    items_by_class: dict[str, list[dict]] = {}
    appearances_by_class: dict[str, list[str]] = {}
    price_groups: dict[str, dict[int, list[str]]] = {}
    base_price_of: dict[tuple[str, str], int] = {}

    for cls in TRACKED_CLASSES:
        real_items = []
        all_descriptions = []
        groups: dict[int, list[str]] = defaultdict(list)

        for item in raw[cls]:
            desc = item.get("description")
            if desc is not None:
                all_descriptions.append(desc)
            if item["name"] is not None:
                real_items.append(item)
                groups[item["cost"]].append(item["name"])
                base_price_of[(cls, item["name"])] = item["cost"]

        items_by_class[cls] = real_items
        appearances_by_class[cls] = all_descriptions
        price_groups[cls] = dict(groups)

    return {
        "items_by_class": items_by_class,
        "appearances_by_class": appearances_by_class,
        "price_groups": price_groups,
        "base_price_of": base_price_of,
    }


# ---------------------------------------------------------------------------
# Charisma-adjusted price calculation
# ---------------------------------------------------------------------------

def _buy_multiplier(charisma: int, surcharge: bool = False) -> float:
    """Shopkeeper buy-price multiplier based on charisma.

    NetHack formula (shk.c):
        price = base * multiplier
    where multiplier depends on CHA and tourist/surcharge status.
    We return the multiplier applied to base_cost.
    """
    # Charisma-based adjustment factor (from shk.c)
    if charisma > 18:
        adj = 1.0
    elif charisma == 18:
        adj = 1.0
    elif charisma >= 16:
        adj = 1.0 + (18 - charisma) * 0.05  # 1.05 .. 1.10
    elif charisma >= 11:
        adj = 1.0 + (18 - charisma) * 0.05  # 1.15 .. 1.35
    elif charisma >= 6:
        adj = 1.5 + (10 - charisma) * 0.1   # 1.5 .. 2.0
    else:
        adj = 2.0 + (5 - charisma) * 0.2    # 2.2 .. 3.0

    if surcharge:
        adj += adj / 3.0  # surcharge adds 33%

    return adj


def _sell_multiplier(charisma: int) -> float:
    """Shopkeeper sell-price multiplier (fraction of base_cost the shop pays).

    NetHack sell price is base_cost / 2 for most items, adjusted by CHA.
    Actually sell = base_cost * sell_mult.
    """
    # Sell price in NetHack: base_cost / 2, further adjusted
    # For CHA >= 15: sell = base / 2
    # For CHA < 15: sell gets worse
    # Simplified: the sell price is always base_cost / 2 in the normal case.
    # CHA affects buy price much more than sell price.
    return 0.5


def _possible_base_prices(observed_price: int, charisma: int, is_buy: bool,
                           surcharge_unknown: bool = True) -> set[int]:
    """Given an observed shop price and charisma, compute the set of base_cost
    values that could produce this observed price.

    For buy: observed = floor(base_cost * multiplier)
    For sell: observed = base_cost / 2 (integer division)
    """
    results = set()

    if is_buy:
        # Try both surcharge and no-surcharge if unknown
        surcharge_opts = [False, True] if surcharge_unknown else [False]
        for sc in surcharge_opts:
            mult = _buy_multiplier(charisma, surcharge=sc)
            # observed = floor(base * mult), so base = observed / mult
            # But NetHack truncates, so base could be in a range.
            # base_cost is always an integer from the item DB.
            # observed = int(base * mult)
            # So base could be floor(observed / mult) or ceil(observed / mult).
            base_low = math.floor(observed_price / mult)
            base_high = math.ceil(observed_price / mult)
            for b in range(max(0, base_low - 1), base_high + 2):
                if int(b * mult) == observed_price:
                    results.add(b)
    else:
        # sell: observed = base // 2
        # so base = observed * 2 or observed * 2 + 1
        results.add(observed_price * 2)
        results.add(observed_price * 2 + 1)

    return results


# ---------------------------------------------------------------------------
# Wand engrave effects table
# ---------------------------------------------------------------------------

# Maps engrave effect description -> set of wand identities that produce it.
ENGRAVE_EFFECTS: dict[str, set[str]] = {
    "engraving now reads": {"fire", "lightning"},
    "ice cubes": {"cold"},
    "fights your attempt": {"striking"},
    "bugs slow down": {"slow monster"},
    "bugs speed up": {"speed monster"},
    "riddled by bullet holes": {"magic missile"},
    "text changes": {"polymorph"},
    "bugs stop moving": {"death", "sleep"},
    "engraving vanishes": {"make invisible", "teleportation", "cancellation"},
    "is now engraved": {"digging"},
    "no effect": {
        "nothing", "light", "probing", "opening", "locking",
        "undead turning", "secret door detection", "enlightenment",
        "create monster", "wishing",
    },
}

# Reverse: wand identity -> engrave effect key
_WAND_TO_ENGRAVE: dict[str, str] = {}
for _eff, _wands in ENGRAVE_EFFECTS.items():
    for _w in _wands:
        _WAND_TO_ENGRAVE[_w] = _eff


# ---------------------------------------------------------------------------
# AppearanceTracker
# ---------------------------------------------------------------------------

class AppearanceTracker:
    """Tracks possible identities for randomized NetHack item appearances.

    One instance per game. Serializable via to_dict / from_dict.

    Each tracked appearance belongs to a class (scroll, potion, etc.) and
    maps to a set of candidate identities. Observations narrow candidates.
    When a candidate set reaches size 1, the appearance is identified and
    that identity is eliminated from all other appearances in the class.
    """

    def __init__(self, data_path: str = _DATA_PATH):
        self._db = _load_item_db(data_path)

        # class -> list of real item names
        self._identities_by_class: dict[str, list[str]] = {
            cls: [item["name"] for item in items]
            for cls, items in self._db["items_by_class"].items()
        }

        # appearance -> item class
        self._appearance_class: dict[str, str] = {}

        # appearance -> set of possible identity names
        self._candidates: dict[str, set[str]] = {}

        # appearance -> identified name (or None)
        self._identified: dict[str, str | None] = {}

        # appearance -> observed BUC status
        self._buc: dict[str, str | None] = {}

        # class -> set of identities that are already assigned
        # (i.e., identified via some appearance)
        self._assigned: dict[str, set[str]] = {cls: set() for cls in TRACKED_CLASSES}

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register(self, appearance: str, item_class: str) -> None:
        """Register an appearance we've seen. Initializes its candidate set
        to all identities in that class minus already-assigned ones."""
        if appearance in self._candidates:
            return  # already registered
        if item_class not in self._identities_by_class:
            raise ValueError(f"Unknown item class: {item_class}")

        self._appearance_class[appearance] = item_class
        self._candidates[appearance] = (
            set(self._identities_by_class[item_class]) - self._assigned[item_class]
        )
        self._identified[appearance] = None
        self._buc[appearance] = None

    def _ensure_registered(self, appearance: str, item_class: str | None = None) -> str:
        """Ensure appearance is registered; return its class."""
        if appearance not in self._candidates:
            if item_class is None:
                raise ValueError(
                    f"Unknown appearance '{appearance}'. "
                    f"Register it first or pass item_class."
                )
            self.register(appearance, item_class)
        return self._appearance_class[appearance]

    # -----------------------------------------------------------------------
    # Constraint propagation
    # -----------------------------------------------------------------------

    def _propagate(self) -> None:
        """Propagate constraints until stable.

        If any appearance has exactly 1 candidate, mark it identified and
        eliminate that identity from all other appearances in the same class.
        Repeat until no more changes.
        """
        changed = True
        while changed:
            changed = False
            for app, cands in list(self._candidates.items()):
                if self._identified[app] is not None:
                    continue
                if len(cands) == 1:
                    identity = next(iter(cands))
                    self._identified[app] = identity
                    cls = self._appearance_class[app]
                    self._assigned[cls].add(identity)
                    # Eliminate from all other appearances in this class
                    for other_app, other_cands in self._candidates.items():
                        if other_app == app:
                            continue
                        if self._appearance_class.get(other_app) != cls:
                            continue
                        if identity in other_cands:
                            other_cands.discard(identity)
                            changed = True
                elif len(cands) == 0:
                    # Contradiction. Should not happen with correct data.
                    pass

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def observe_price(self, appearance: str, buy_price: int | None = None,
                      sell_price: int | None = None, charisma: int = 10,
                      item_class: str | None = None,
                      surcharge_unknown: bool = True) -> None:
        """Narrow candidates by observed shop price."""
        cls = self._ensure_registered(appearance, item_class)

        possible_bases: set[int] = set()

        if buy_price is not None:
            possible_bases |= _possible_base_prices(
                buy_price, charisma, is_buy=True,
                surcharge_unknown=surcharge_unknown,
            )
        if sell_price is not None:
            possible_bases |= _possible_base_prices(
                sell_price, charisma, is_buy=False,
            )

        if not possible_bases:
            return

        # Keep only candidates whose base_cost is in possible_bases
        price_of = self._db["base_price_of"]
        self._candidates[appearance] = {
            name for name in self._candidates[appearance]
            if price_of.get((cls, name)) in possible_bases
        }
        self._propagate()

    def observe_buc(self, appearance: str, buc_status: str,
                    item_class: str | None = None) -> None:
        """Record blessed/uncursed/cursed status.

        BUC doesn't directly narrow identity candidates (most items can be
        any BUC), but we store it for downstream use.
        """
        self._ensure_registered(appearance, item_class)
        self._buc[appearance] = buc_status

    def observe_engrave_effect(self, appearance: str, effect_type: str,
                                item_class: str | None = None) -> None:
        """Narrow wand candidates by engrave-test result.

        effect_type should be one of the keys in ENGRAVE_EFFECTS.
        """
        cls = self._ensure_registered(appearance, item_class or "wand")
        if cls != "wand":
            return

        matching_wands = ENGRAVE_EFFECTS.get(effect_type)
        if matching_wands is None:
            return

        self._candidates[appearance] &= matching_wands
        self._propagate()

    def observe_use_effect(self, appearance: str, effect_description: str,
                           item_class: str | None = None) -> None:
        """Narrow candidates by use effect.

        For scrolls/potions, many use effects uniquely identify the item.
        Pass the identity name as effect_description to fully identify.
        """
        cls = self._ensure_registered(appearance, item_class)

        # If the effect_description matches an identity name, treat as
        # direct identification.
        if effect_description in self._candidates[appearance]:
            self._candidates[appearance] = {effect_description}
            self._propagate()

    def observe_identified(self, appearance: str, identity: str,
                           item_class: str | None = None) -> None:
        """Directly identify an appearance (e.g., via scroll of identify)."""
        cls = self._ensure_registered(appearance, item_class)

        if identity not in set(self._identities_by_class[cls]):
            raise ValueError(
                f"'{identity}' is not a valid {cls} identity."
            )

        self._candidates[appearance] = {identity}
        self._propagate()

    def observe_elimination(self, identity: str, item_class: str) -> None:
        """Eliminate an identity from all unidentified appearances in a class.

        Use when we know this identity maps to a DIFFERENT appearance than
        any we're currently tracking (e.g., we found a shop with "scroll of
        light" by name, so no randomized appearance maps to it).
        """
        if item_class not in self._identities_by_class:
            raise ValueError(f"Unknown class: {item_class}")

        self._assigned[item_class].add(identity)
        for app, cands in self._candidates.items():
            if self._appearance_class.get(app) != item_class:
                continue
            if self._identified[app] is not None:
                continue
            cands.discard(identity)
        self._propagate()

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def get_possibilities(self, appearance: str) -> list[tuple[str, float]]:
        """Return possible identities with uniform probabilities."""
        if appearance not in self._candidates:
            return []
        cands = self._candidates[appearance]
        n = len(cands)
        if n == 0:
            return []
        p = 1.0 / n
        return [(name, p) for name in sorted(cands)]

    def is_identified(self, appearance: str) -> bool:
        return self._identified.get(appearance) is not None

    def get_identity(self, appearance: str) -> str | None:
        return self._identified.get(appearance)

    def get_all_identified(self) -> dict[str, str]:
        """Return {appearance: identity} for all identified appearances."""
        return {
            app: ident for app, ident in self._identified.items()
            if ident is not None
        }

    def get_price_group(self, item_class: str, price: int) -> list[str]:
        """Return item names at the given base price in the given class."""
        groups = self._db["price_groups"].get(item_class, {})
        return groups.get(price, [])

    def get_candidates(self, appearance: str) -> set[str]:
        """Return the raw candidate set for an appearance."""
        return set(self._candidates.get(appearance, set()))

    def get_buc(self, appearance: str) -> str | None:
        return self._buc.get(appearance)

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "appearance_class": dict(self._appearance_class),
            "candidates": {app: sorted(c) for app, c in self._candidates.items()},
            "identified": dict(self._identified),
            "buc": dict(self._buc),
            "assigned": {cls: sorted(s) for cls, s in self._assigned.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], data_path: str = _DATA_PATH) -> "AppearanceTracker":
        tracker = cls(data_path)
        tracker._appearance_class = data["appearance_class"]
        tracker._candidates = {app: set(c) for app, c in data["candidates"].items()}
        tracker._identified = data["identified"]
        tracker._buc = data["buc"]
        tracker._assigned = {c: set(s) for c, s in data["assigned"].items()}
        return tracker

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str, data_path: str = _DATA_PATH) -> "AppearanceTracker":
        return cls.from_dict(json.loads(s), data_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test():
    print("=== Item ID Constraint Engine Tests ===\n")
    tracker = AppearanceTracker()

    # --- Test 1: Basic registration and identification ---
    print("Test 1: Register scroll appearance, identify directly")
    tracker.register("ZELGO MER", "scroll")
    assert not tracker.is_identified("ZELGO MER")
    n_before = len(tracker.get_candidates("ZELGO MER"))
    print(f"  Candidates before: {n_before}")

    tracker.observe_identified("ZELGO MER", "enchant armor")
    assert tracker.is_identified("ZELGO MER")
    assert tracker.get_identity("ZELGO MER") == "enchant armor"
    print(f"  Identified as: {tracker.get_identity('ZELGO MER')}")
    print("  PASS\n")

    # --- Test 2: Price narrowing ---
    print("Test 2: Price narrowing for scrolls")
    tracker.register("JUYED AWK YACC", "scroll")
    n_before = len(tracker.get_candidates("JUYED AWK YACC"))
    print(f"  Candidates before price: {n_before}")
    # enchant armor already assigned, so should be n_scrolls - 1

    # Sell price of 10 -> base = 20. Only scroll at base 20 is "identify".
    tracker.observe_price("JUYED AWK YACC", sell_price=10)
    print(f"  After sell_price=10: {tracker.get_candidates('JUYED AWK YACC')}")
    assert tracker.is_identified("JUYED AWK YACC")
    assert tracker.get_identity("JUYED AWK YACC") == "identify"
    print(f"  Identified as: {tracker.get_identity('JUYED AWK YACC')}")
    print("  PASS\n")

    # --- Test 3: Constraint propagation cascade ---
    print("Test 3: Propagation cascade")
    t2 = AppearanceTracker()

    # Register two wand appearances
    t2.register("oak wand", "wand")
    t2.register("ebony wand", "wand")

    all_wand_names = set(t2._identities_by_class["wand"])
    print(f"  Total wand identities: {len(all_wand_names)}")

    # Eliminate all but 2 identities from both appearances
    # Keep only "light" and "nothing" (both cost 100)
    for name in all_wand_names:
        if name not in ("light", "nothing"):
            t2.observe_elimination(name, "wand")

    print(f"  oak wand candidates: {t2.get_candidates('oak wand')}")
    print(f"  ebony wand candidates: {t2.get_candidates('ebony wand')}")

    # Now identify one
    t2.observe_identified("oak wand", "light")
    assert t2.is_identified("oak wand")
    assert t2.get_identity("oak wand") == "light"
    # Propagation should identify the other
    assert t2.is_identified("ebony wand"), "Propagation failed"
    assert t2.get_identity("ebony wand") == "nothing"
    print(f"  oak wand -> {t2.get_identity('oak wand')}")
    print(f"  ebony wand -> {t2.get_identity('ebony wand')} (via propagation)")
    print("  PASS\n")

    # --- Test 4: Wand engrave effects ---
    print("Test 4: Wand engrave effects")
    t3 = AppearanceTracker()
    t3.register("curved wand", "wand")

    t3.observe_engrave_effect("curved wand", "engraving now reads")
    cands = t3.get_candidates("curved wand")
    print(f"  After 'engraving now reads': {cands}")
    assert cands == {"fire", "lightning"}

    # Now narrow by price. fire=175, lightning=175. Both same price, so
    # price alone won't help. But sell price of 87 -> base = 174 or 175.
    # Both match. Let's use a different distinguishing observation.
    # Identify directly for test purposes.
    t3.observe_identified("curved wand", "fire")
    assert t3.get_identity("curved wand") == "fire"
    print(f"  Identified as: {t3.get_identity('curved wand')}")
    print("  PASS\n")

    # --- Test 5: Engrave disambiguate death vs sleep by price ---
    print("Test 5: Disambiguate death vs sleep by engrave + price")
    t4 = AppearanceTracker()
    t4.register("long wand", "wand")

    t4.observe_engrave_effect("long wand", "bugs stop moving")
    cands = t4.get_candidates("long wand")
    print(f"  After 'bugs stop moving': {cands}")
    assert cands == {"death", "sleep"}

    # death costs 500, sleep costs 175. sell_price=250 -> base=500 -> death
    t4.observe_price("long wand", sell_price=250)
    assert t4.is_identified("long wand")
    assert t4.get_identity("long wand") == "death"
    print(f"  After sell_price=250: {t4.get_identity('long wand')}")
    print("  PASS\n")

    # --- Test 6: Serialization round-trip ---
    print("Test 6: Serialization round-trip")
    state_json = tracker.to_json()
    restored = AppearanceTracker.from_json(state_json)
    assert restored.get_all_identified() == tracker.get_all_identified()
    assert restored.get_candidates("ZELGO MER") == tracker.get_candidates("ZELGO MER")
    print(f"  Round-trip OK, identified: {restored.get_all_identified()}")
    print("  PASS\n")

    # --- Test 7: Price group query ---
    print("Test 7: Price group query")
    group = tracker.get_price_group("scroll", 100)
    print(f"  Scrolls at base 100: {group}")
    assert "teleportation" in group
    assert "fire" in group
    assert "identify" not in group  # identify is 20
    print("  PASS\n")

    # --- Test 8: BUC recording ---
    print("Test 8: BUC recording")
    tracker.observe_buc("ZELGO MER", "blessed")
    assert tracker.get_buc("ZELGO MER") == "blessed"
    print(f"  BUC for ZELGO MER: {tracker.get_buc('ZELGO MER')}")
    print("  PASS\n")

    # --- Test 9: Use effect identification ---
    print("Test 9: Use effect identification")
    t5 = AppearanceTracker()
    t5.register("ruby potion", "potion")
    n_before = len(t5.get_candidates("ruby potion"))
    print(f"  Candidates before: {n_before}")

    t5.observe_use_effect("ruby potion", "gain ability")
    assert t5.is_identified("ruby potion")
    assert t5.get_identity("ruby potion") == "gain ability"
    print(f"  After use effect: {t5.get_identity('ruby potion')}")
    print("  PASS\n")

    # --- Test 10: Multi-step narrowing with propagation ---
    print("Test 10: Multi-step narrowing (price + engrave + propagation)")
    t6 = AppearanceTracker()
    t6.register("tin wand", "wand")
    t6.register("brass wand", "wand")
    t6.register("silver wand", "wand")

    # All three are cost 150 or 200 wands. Let's narrow by price first.
    # sell_price=100 -> base=200. Cost-200 wands: create monster, polymorph,
    # cancellation, teleportation
    t6.observe_price("silver wand", sell_price=100)
    cands_silver = t6.get_candidates("silver wand")
    print(f"  silver wand after sell=100: {cands_silver}")
    assert cands_silver == {"create monster", "polymorph", "cancellation", "teleportation"}

    # Engrave on silver wand: "engraving vanishes" -> make invisible,
    # teleportation, cancellation. Intersect with price candidates.
    t6.observe_engrave_effect("silver wand", "engraving vanishes")
    cands_silver = t6.get_candidates("silver wand")
    print(f"  silver wand after engrave vanishes: {cands_silver}")
    assert cands_silver == {"teleportation", "cancellation"}

    # Identify silver as teleportation
    t6.observe_identified("silver wand", "teleportation")
    print(f"  silver wand identified: {t6.get_identity('silver wand')}")

    # Now if we narrow brass wand to cost 150 engrave "no effect"
    t6.observe_price("brass wand", sell_price=75)
    t6.observe_engrave_effect("brass wand", "no effect")
    cands_brass = t6.get_candidates("brass wand")
    print(f"  brass wand after sell=75 + no effect: {cands_brass}")
    # cost-150 no-effect wands: light (cost 100, no), probing, opening,
    # locking, undead turning, secret door detection, enlightenment
    # Actually "nothing" is cost 100 and "light" is cost 100 too.
    # Cost 150 no-effect: probing, opening, locking, undead turning,
    # secret door detection, enlightenment
    # "create monster" is cost 200 and "wishing" is cost 500, so they're out.
    for name in cands_brass:
        assert name in {
            "probing", "opening", "locking", "undead turning",
            "secret door detection", "enlightenment",
        }, f"Unexpected candidate: {name}"
    print("  PASS\n")

    print("=== All tests passed ===")


if __name__ == "__main__":
    _test()
