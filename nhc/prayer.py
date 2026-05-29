"""Prayer safety checker for NetHack.

Tracks prayer-related state and determines when prayer is safe,
what troubles the player has, and whether praying is the right call.
Rules derived from NetHack 3.6.7 src/pray.c mechanics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

# NLE blstats indices (canonical order, matches nhc/kb.py)
BL_HP = 10
BL_MAXHP = 11
BL_DEPTH = 12
BL_HUNGER = 18
BL_XL = 21
BL_ALIGNMENT = 22
# Extended indices (NLE 0.9+, 27-element blstats)
BL_CONDITION = 23
BL_TIME = 25

# NLE condition bitmask flags
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


class Alignment(IntEnum):
    LAWFUL = 1
    NEUTRAL = 0
    CHAOTIC = -1


class HungerState(IntEnum):
    SATIATED = 0
    NOT_HUNGRY = 1
    HUNGRY = 2
    WEAK = 3
    FAINTING = 4


class TroubleSeverity(IntEnum):
    NONE = 0
    MINOR = 1
    MAJOR = 2


# Prayer timeout thresholds (from pray.c)
TIMEOUT_MAJOR = 200    # pray with major trouble if ublesscnt <= 200
TIMEOUT_MINOR = 100    # pray with minor trouble if ublesscnt <= 100
TIMEOUT_SAFE = 0       # fully safe, any prayer works
INITIAL_TIMEOUT = 300  # default starting timeout (first 300 turns)


@dataclass
class TroubleInfo:
    trouble_type: str
    severity: TroubleSeverity
    priority: int  # lower = more urgent for major, higher abs = more urgent for minor


# Major troubles in priority order (highest priority first)
MAJOR_TROUBLES = [
    "stoning", "sliming", "strangulation", "lava", "illness",
    "starving", "region", "hp_critical", "lycanthropy", "collapsing",
    "stuck_in_wall", "cursed_levitation", "unusable_hands", "cursed_blindfold",
]

# Minor troubles
MINOR_TROUBLES = [
    "punished", "fumbling", "cursed_items", "cursed_saddle",
    "blind", "poisoned", "wounded_legs", "hungry",
    "stunned", "confused", "hallucinating",
]


@dataclass
class PrayerState:
    """Tracks prayer-related state across the game."""
    last_prayer_turn: int = 0
    alignment_record: int = 0
    god_anger: int = 0
    luck: int = 0
    in_gehennom: bool = False
    on_altar: bool = False
    altar_alignment: Optional[Alignment] = None
    player_alignment: Alignment = Alignment.LAWFUL

    def _timeout_remaining(self, current_turn: int) -> int:
        """Estimated prayer timeout remaining.

        ublesscnt decrements by 1 each turn. We approximate it as
        (INITIAL_TIMEOUT - turns_since_last_prayer) for the first prayer,
        and track it explicitly after that.
        """
        if self.last_prayer_turn == 0:
            # Never prayed. Initial 300-turn timeout from game start.
            return max(0, INITIAL_TIMEOUT - current_turn)
        return max(0, self._raw_timeout - (current_turn - self.last_prayer_turn))

    # Internal: raw timeout value set at last prayer
    _raw_timeout: int = field(default=INITIAL_TIMEOUT, repr=False)

    def is_prayer_safe(
        self, current_turn: int, trouble_type: Optional[str] = None
    ) -> tuple[bool, str]:
        """Check whether prayer is safe right now.

        Returns (safe, reason). If safe is False, reason explains why.
        """
        timeout = self._timeout_remaining(current_turn)

        # Determine timeout threshold based on trouble severity
        if trouble_type and trouble_type in MAJOR_TROUBLES:
            threshold = TIMEOUT_MAJOR
        elif trouble_type and trouble_type in MINOR_TROUBLES:
            threshold = TIMEOUT_MINOR
        else:
            threshold = TIMEOUT_SAFE

        if timeout > threshold:
            return False, f"prayer timeout not expired ({timeout} turns remaining, need <= {threshold})"

        if self.god_anger > 0:
            return False, f"god is angry (anger={self.god_anger})"

        if self.alignment_record < 0:
            return False, f"negative alignment record ({self.alignment_record})"

        if self.luck < 0:
            return False, f"negative luck ({self.luck})"

        # Gehennom check: prayer always fails unless on coaligned altar
        if self.in_gehennom:
            if not (self.on_altar and self.altar_alignment == self.player_alignment):
                return False, "in Gehennom without coaligned altar"

        # Cross-aligned altar check
        if self.on_altar and self.altar_alignment is not None:
            if self.altar_alignment != self.player_alignment:
                return False, f"on cross-aligned altar ({self.altar_alignment.name} vs {self.player_alignment.name})"

        return True, "prayer is safe"

    def classify_trouble(self, game_state: dict) -> tuple[Optional[str], TroubleSeverity]:
        """Classify the worst current trouble from game state.

        game_state keys:
            hp, max_hp: current and max hit points
            hunger: HungerState value (0-4)
            condition: NLE condition bitmask
            punished: bool
            cursed_blindfold: bool
            cursed_levitation: bool
            wounded_legs: bool
            cursed_items: bool (any worn cursed items)

        Returns (trouble_type, severity). trouble_type is None if no trouble.
        """
        hp = game_state.get("hp", 100)
        max_hp = game_state.get("max_hp", 100)
        hunger = game_state.get("hunger", HungerState.NOT_HUNGRY)
        cond = game_state.get("condition", 0)

        # Major troubles, checked in priority order
        if cond & COND_STONE:
            return "stoning", TroubleSeverity.MAJOR
        if cond & COND_SLIME:
            return "sliming", TroubleSeverity.MAJOR
        if cond & COND_STRNGL:
            return "strangulation", TroubleSeverity.MAJOR
        if cond & COND_FOODPOIS:
            return "food_poisoning", TroubleSeverity.MAJOR
        if cond & COND_TERMILL:
            return "illness", TroubleSeverity.MAJOR
        if hunger >= HungerState.WEAK:
            return "starving", TroubleSeverity.MAJOR
        if max_hp > 0 and hp <= max(5, max_hp // 7):
            return "hp_critical", TroubleSeverity.MAJOR
        if game_state.get("cursed_levitation", False):
            return "cursed_levitation", TroubleSeverity.MAJOR
        if game_state.get("cursed_blindfold", False):
            return "cursed_blindfold", TroubleSeverity.MAJOR

        # Minor troubles
        if game_state.get("punished", False):
            return "punished", TroubleSeverity.MINOR
        if cond & COND_BLIND:
            return "blind", TroubleSeverity.MINOR
        if game_state.get("wounded_legs", False):
            return "wounded_legs", TroubleSeverity.MINOR
        if hunger >= HungerState.HUNGRY:
            return "hungry", TroubleSeverity.MINOR
        if cond & COND_STUN:
            return "stunned", TroubleSeverity.MINOR
        if cond & COND_CONF:
            return "confused", TroubleSeverity.MINOR
        if cond & COND_HALLU:
            return "hallucinating", TroubleSeverity.MINOR

        return None, TroubleSeverity.NONE

    def should_pray(
        self, current_turn: int, game_state: dict
    ) -> tuple[bool, str]:
        """Combine safety check with trouble assessment.

        Returns (recommend, reason).
        Recommends prayer when:
          - Major trouble AND prayer is safe for major trouble
          - Minor trouble AND timeout fully expired AND no better alternative
        """
        trouble_type, severity = self.classify_trouble(game_state)

        if severity == TroubleSeverity.NONE:
            return False, "no trouble detected"

        safe, reason = self.is_prayer_safe(current_turn, trouble_type)

        if severity == TroubleSeverity.MAJOR:
            if safe:
                return True, f"major trouble ({trouble_type}), prayer safe"
            return False, f"major trouble ({trouble_type}) but prayer unsafe: {reason}"

        # Minor trouble: only recommend if timeout fully expired
        if severity == TroubleSeverity.MINOR:
            timeout = self._timeout_remaining(current_turn)
            if timeout > TIMEOUT_SAFE:
                return False, f"minor trouble ({trouble_type}) but timeout not fully expired ({timeout} remaining)"
            if not safe:
                return False, f"minor trouble ({trouble_type}) but prayer unsafe: {reason}"
            # Check for alternatives before recommending prayer for minor trouble
            if trouble_type == "hungry" and game_state.get("has_food", False):
                return False, "hungry but food available, eat instead"
            if trouble_type == "blind" and game_state.get("has_eyedrops", False):
                return False, "blind but cure available"
            return True, f"minor trouble ({trouble_type}), timeout expired, no better alternative"

        return False, "unknown state"

    # ----------------------------------------------------------------
    # Update methods
    # ----------------------------------------------------------------

    def update_prayed(self, current_turn: int) -> None:
        """Record that we just prayed. Sets timeout to ~350 (pleased)."""
        self.last_prayer_turn = current_turn
        # rnz(350) averages around 350. We use the mean.
        self._raw_timeout = 350

    def update_sacrifice(self, monster_difficulty: int) -> None:
        """Sacrifice reduces prayer timeout.

        From pray.c: value * (500 for chaotic, 300 for others) / MAXVALUE
        subtracted from ublesscnt.
        """
        value = min(monster_difficulty + 1, 24)  # capped at MAXVALUE=24
        if self.player_alignment == Alignment.CHAOTIC:
            reduction = value * 500 // 24
        else:
            reduction = value * 300 // 24
        self._raw_timeout = max(0, self._raw_timeout - reduction)

    def update_alignment(self, delta: int) -> None:
        """Apply alignment record change (kills, sacrifices, etc.)."""
        self.alignment_record += delta

    def update_from_blstats(self, blstats: list | tuple) -> None:
        """Parse NLE blstats array for prayer-relevant state.

        Updates hp, alignment, hunger, turn, depth, condition flags.
        Does NOT update god_anger, luck, altar state, or gehennom status,
        since those aren't directly in blstats.
        """
        if len(blstats) > BL_ALIGNMENT:
            self.alignment_record = int(blstats[BL_ALIGNMENT])
        if len(blstats) > BL_DEPTH:
            depth = int(blstats[BL_DEPTH])
            # Gehennom starts at depth ~25 in standard NetHack
            # but this is a heuristic; real detection needs dungeon_number
            self.in_gehennom = depth >= 25


def _run_tests():
    """Simulate a few game scenarios."""
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

    print("=== Prayer Safety Tests ===\n")

    # --- Test 1: Fresh game, no trouble, timeout not expired ---
    print("Test 1: Fresh game, turn 100, no trouble")
    ps = PrayerState()
    safe, reason = ps.is_prayer_safe(100)
    check("prayer unsafe at turn 100 (timeout)", not safe)
    check("reason mentions timeout", "timeout" in reason, reason)

    # --- Test 2: Fresh game, turn 400, no trouble, safe ---
    print("\nTest 2: Fresh game, turn 400, no trouble")
    ps = PrayerState()
    safe, reason = ps.is_prayer_safe(400)
    check("prayer safe at turn 400", safe, reason)

    # --- Test 3: Major trouble overrides partial timeout ---
    print("\nTest 3: Major trouble (stoning) at turn 150")
    ps = PrayerState()
    # At turn 150, timeout remaining = 300-150 = 150. Major threshold = 200.
    safe, reason = ps.is_prayer_safe(150, trouble_type="stoning")
    check("prayer safe for major trouble at turn 150", safe, reason)

    # --- Test 4: Major trouble but too early ---
    print("\nTest 4: Major trouble (stoning) at turn 50")
    ps = PrayerState()
    # At turn 50, timeout remaining = 300-50 = 250 > 200
    safe, reason = ps.is_prayer_safe(50, trouble_type="stoning")
    check("prayer unsafe for major trouble at turn 50", not safe, reason)

    # --- Test 5: God angry ---
    print("\nTest 5: God angry")
    ps = PrayerState(god_anger=2)
    safe, reason = ps.is_prayer_safe(500)
    check("prayer unsafe when god angry", not safe)
    check("reason mentions angry", "angry" in reason, reason)

    # --- Test 6: Negative alignment ---
    print("\nTest 6: Negative alignment record")
    ps = PrayerState(alignment_record=-5)
    safe, reason = ps.is_prayer_safe(500)
    check("prayer unsafe with negative alignment", not safe)

    # --- Test 7: Negative luck ---
    print("\nTest 7: Negative luck")
    ps = PrayerState(luck=-1)
    safe, reason = ps.is_prayer_safe(500)
    check("prayer unsafe with negative luck", not safe)

    # --- Test 8: In Gehennom without coaligned altar ---
    print("\nTest 8: In Gehennom, no altar")
    ps = PrayerState(in_gehennom=True)
    safe, reason = ps.is_prayer_safe(500)
    check("prayer unsafe in Gehennom", not safe)
    check("reason mentions Gehennom", "Gehennom" in reason, reason)

    # --- Test 9: In Gehennom with coaligned altar ---
    print("\nTest 9: In Gehennom, on coaligned altar")
    ps = PrayerState(
        in_gehennom=True, on_altar=True,
        altar_alignment=Alignment.LAWFUL, player_alignment=Alignment.LAWFUL,
    )
    safe, reason = ps.is_prayer_safe(500)
    check("prayer safe in Gehennom on coaligned altar", safe, reason)

    # --- Test 10: Cross-aligned altar ---
    print("\nTest 10: On cross-aligned altar")
    ps = PrayerState(
        on_altar=True,
        altar_alignment=Alignment.CHAOTIC, player_alignment=Alignment.LAWFUL,
    )
    safe, reason = ps.is_prayer_safe(500)
    check("prayer unsafe on cross-aligned altar", not safe)
    check("reason mentions cross-aligned", "cross-aligned" in reason.lower() or "CHAOTIC" in reason, reason)

    # --- Test 11: Trouble classification ---
    print("\nTest 11: Trouble classification")
    trouble, sev = ps.classify_trouble({"hp": 3, "max_hp": 50, "condition": 0, "hunger": 1})
    check("low HP is major trouble", sev == TroubleSeverity.MAJOR)
    check("trouble type is hp_critical", trouble == "hp_critical")

    trouble, sev = ps.classify_trouble({"hp": 50, "max_hp": 50, "condition": COND_STONE, "hunger": 1})
    check("stoning is major", sev == TroubleSeverity.MAJOR)
    check("stoning identified", trouble == "stoning")

    trouble, sev = ps.classify_trouble({"hp": 50, "max_hp": 50, "condition": COND_STUN, "hunger": 1})
    check("stun is minor", sev == TroubleSeverity.MINOR)
    check("stunned identified", trouble == "stunned")

    trouble, sev = ps.classify_trouble({"hp": 50, "max_hp": 50, "condition": 0, "hunger": 1})
    check("no trouble when healthy", sev == TroubleSeverity.NONE)

    # --- Test 12: Hunger levels ---
    print("\nTest 12: Hunger trouble levels")
    trouble, sev = ps.classify_trouble({"hp": 50, "max_hp": 50, "condition": 0, "hunger": HungerState.WEAK})
    check("WEAK hunger is major (starving)", sev == TroubleSeverity.MAJOR and trouble == "starving")

    trouble, sev = ps.classify_trouble({"hp": 50, "max_hp": 50, "condition": 0, "hunger": HungerState.HUNGRY})
    check("HUNGRY is minor", sev == TroubleSeverity.MINOR and trouble == "hungry")

    # --- Test 13: should_pray integration ---
    print("\nTest 13: should_pray integration")
    ps = PrayerState()
    game = {"hp": 3, "max_hp": 50, "condition": 0, "hunger": 1}
    rec, reason = ps.should_pray(400, game)
    check("recommend prayer for critical HP at turn 400", rec, reason)

    ps = PrayerState()
    game = {"hp": 3, "max_hp": 50, "condition": 0, "hunger": 1}
    rec, reason = ps.should_pray(50, game)
    check("don't recommend prayer at turn 50 (timeout)", not rec, reason)

    # --- Test 14: Prayer then re-prayer timing ---
    print("\nTest 14: Prayer then re-prayer timing")
    ps = PrayerState()
    ps.update_prayed(400)
    check("last_prayer_turn updated", ps.last_prayer_turn == 400)

    safe, reason = ps.is_prayer_safe(500)
    check("unsafe 100 turns after prayer", not safe, reason)

    safe, reason = ps.is_prayer_safe(800)
    check("safe 400 turns after prayer", safe, reason)

    # --- Test 15: Sacrifice reduces timeout ---
    print("\nTest 15: Sacrifice reduces timeout")
    ps = PrayerState()
    ps.update_prayed(0)
    initial_timeout = ps._raw_timeout
    ps.update_sacrifice(15)  # difficulty 15 -> value 16
    check("sacrifice reduced timeout", ps._raw_timeout < initial_timeout,
          f"{initial_timeout} -> {ps._raw_timeout}")

    # --- Test 16: Minor trouble needs full timeout expiry ---
    print("\nTest 16: Minor trouble requires full timeout expiry")
    ps = PrayerState()
    game = {"hp": 50, "max_hp": 50, "condition": COND_CONF, "hunger": 1}
    rec, reason = ps.should_pray(250, game)
    check("don't recommend prayer for confusion at turn 250", not rec, reason)

    rec, reason = ps.should_pray(400, game)
    check("recommend prayer for confusion at turn 400", rec, reason)

    # --- Test 17: Minor trouble with food alternative ---
    print("\nTest 17: Minor trouble with better alternative")
    ps = PrayerState()
    game = {"hp": 50, "max_hp": 50, "condition": 0, "hunger": HungerState.HUNGRY, "has_food": True}
    rec, reason = ps.should_pray(400, game)
    check("don't recommend prayer when food available", not rec, reason)

    game["has_food"] = False
    rec, reason = ps.should_pray(400, game)
    check("recommend prayer for hunger without food", rec, reason)

    # --- Test 18: update_from_blstats ---
    print("\nTest 18: update_from_blstats")
    ps = PrayerState()
    bl = [0] * 27
    bl[BL_ALIGNMENT] = 12
    bl[BL_DEPTH] = 30
    ps.update_from_blstats(bl)
    check("alignment updated from blstats", ps.alignment_record == 12)
    check("gehennom detected at depth 30", ps.in_gehennom)

    bl[BL_DEPTH] = 5
    ps.update_from_blstats(bl)
    check("not gehennom at depth 5", not ps.in_gehennom)

    # --- Summary ---
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
