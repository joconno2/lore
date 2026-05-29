"""Tests for nhc.combat threat evaluation module."""
from __future__ import annotations

import pytest
from nhc.combat import ThreatDB, ThreatReport, CorpseReport, RoomThreatReport


@pytest.fixture(scope="module")
def db():
    return ThreatDB()


def _player(hp=50, max_hp=50, ac=5, level=10, speed=12,
            resistances=None, equipment=None, position=(40, 10),
            has_elbereth_source=True):
    return {
        "hp": hp,
        "max_hp": max_hp,
        "ac": ac,
        "level": level,
        "speed": speed,
        "resistances": resistances or set(),
        "equipment": equipment or {},
        "position": position,
        "has_elbereth_source": has_elbereth_source,
    }


# ============================================================
# ThreatDB loading
# ============================================================

class TestThreatDBLoading:
    def test_loads_monsters(self, db):
        assert db._get_monster("giant ant") is not None

    def test_loads_monsters_count(self, db):
        # 384 entries in JSON but 3 werebeasts appear twice (human + animal form),
        # so the name-keyed dict holds 381 unique monsters.
        assert len(db._monsters) == 381

    def test_loads_corpse_effects(self, db):
        assert "mechanics" in db.corpse_effects

    def test_unknown_monster_returns_none(self, db):
        assert db._get_monster("totally fake monster") is None


# ============================================================
# assess_threat
# ============================================================

class TestAssessThreat:
    def test_returns_threat_report(self, db):
        r = db.assess_threat("giant ant", _player())
        assert isinstance(r, ThreatReport)

    def test_cockatrice_instakill(self, db):
        r = db.assess_threat("cockatrice", _player())
        assert r.instakill_risk is True
        assert r.danger_level == 10
        assert "stoning touch" in r.special_attacks
        assert "stoning resistance" in r.required_resistances

    def test_chickatrice_instakill(self, db):
        r = db.assess_threat("chickatrice", _player())
        assert r.instakill_risk is True

    def test_floating_eye_ranged_preferred(self, db):
        r = db.assess_threat("floating eye", _player())
        assert r.ranged_preferred is True

    def test_floating_eye_paralysis_warning(self, db):
        r = db.assess_threat("floating eye", _player())
        assert any("paralysis" in s for s in r.special_attacks)

    def test_mind_flayer_brain_eating(self, db):
        r = db.assess_threat("mind flayer", _player())
        assert any("brain" in s for s in r.special_attacks)

    def test_death_touch_of_death(self, db):
        r = db.assess_threat("Death", _player())
        assert r.instakill_risk is True
        assert any("death" in s.lower() for s in r.special_attacks)

    def test_vampire_level_drain(self, db):
        r = db.assess_threat("vampire", _player())
        assert any("level drain" in s for s in r.special_attacks)
        assert "drain resistance" in r.required_resistances

    def test_green_slime_sliming(self, db):
        r = db.assess_threat("green slime", _player())
        assert r.instakill_risk is True
        assert any("sliming" in s for s in r.special_attacks)

    def test_rust_monster_ranged(self, db):
        r = db.assess_threat("rust monster", _player())
        assert r.ranged_preferred is True

    def test_low_hp_increases_danger(self, db):
        healthy = db.assess_threat("soldier ant", _player(hp=50, max_hp=50))
        wounded = db.assess_threat("soldier ant", _player(hp=10, max_hp=50))
        assert wounded.danger_level >= healthy.danger_level

    def test_danger_level_range(self, db):
        for name in ["newt", "giant ant", "arch-lich", "Death"]:
            r = db.assess_threat(name, _player())
            assert 1 <= r.danger_level <= 10

    def test_unknown_monster_conservative(self, db):
        r = db.assess_threat("xyzzy monster", _player())
        assert r.danger_level >= 5
        assert len(r.special_attacks) > 0  # warns about unknown

    def test_fire_ant_fire_resistance_needed(self, db):
        r = db.assess_threat("fire ant", _player())
        assert "fire resistance" in r.required_resistances

    def test_arch_lich_high_danger(self, db):
        r = db.assess_threat("arch-lich", _player(level=5))
        assert r.danger_level >= 7

    def test_newt_low_danger(self, db):
        r = db.assess_threat("newt", _player(level=10))
        assert r.danger_level <= 3

    def test_recommended_action_is_valid(self, db):
        valid = {"melee", "ranged", "elbereth", "flee", "use_item"}
        for name in ["newt", "cockatrice", "arch-lich", "floating eye"]:
            r = db.assess_threat(name, _player())
            assert r.recommended_action in valid

    def test_flee_from_instakill_without_resistance(self, db):
        r = db.assess_threat("cockatrice", _player(resistances=set()))
        assert r.recommended_action == "flee"

    def test_cockatrice_ranged_preferred(self, db):
        r = db.assess_threat("cockatrice", _player())
        assert r.ranged_preferred is True

    def test_purple_worm_engulf(self, db):
        r = db.assess_threat("purple worm", _player())
        assert any("engulf" in s or "digestion" in s.lower()
                    for s in r.special_attacks)

    def test_spell_caster_noted(self, db):
        r = db.assess_threat("arch-lich", _player())
        assert any("spell" in s for s in r.special_attacks)


# ============================================================
# respects_elbereth
# ============================================================

class TestRespectsElbereth:
    def test_giant_ant_respects(self, db):
        assert db.respects_elbereth("giant ant") is True

    def test_minotaur_ignores(self, db):
        assert db.respects_elbereth("minotaur") is False

    def test_death_ignores(self, db):
        assert db.respects_elbereth("Death") is False

    def test_pestilence_ignores(self, db):
        assert db.respects_elbereth("Pestilence") is False

    def test_famine_ignores(self, db):
        assert db.respects_elbereth("Famine") is False

    def test_shopkeeper_ignores(self, db):
        assert db.respects_elbereth("shopkeeper") is False

    def test_guard_ignores(self, db):
        assert db.respects_elbereth("guard") is False

    def test_aligned_priest_ignores(self, db):
        assert db.respects_elbereth("aligned priest") is False

    def test_archon_ignores(self, db):
        assert db.respects_elbereth("Archon") is False

    def test_angel_ignores(self, db):
        # A-class symbol
        assert db.respects_elbereth("Angel") is False

    def test_couatl_ignores(self, db):
        # Also A-class
        assert db.respects_elbereth("couatl") is False

    def test_all_at_sign_monsters_ignore(self, db):
        """Every @ monster should ignore Elbereth."""
        for name, mon in db._monsters.items():
            if mon["symbol"] == "@":
                assert db.respects_elbereth(name) is False, \
                    f"{name} (@ class) should ignore Elbereth"

    def test_all_a_class_ignore(self, db):
        """Every A monster should ignore Elbereth."""
        for name, mon in db._monsters.items():
            if mon["symbol"] == "A":
                assert db.respects_elbereth(name) is False, \
                    f"{name} (A class) should ignore Elbereth"

    def test_wizard_of_yendor_ignores(self, db):
        assert db.respects_elbereth("Wizard of Yendor") is False

    def test_demogorgon_ignores(self, db):
        assert db.respects_elbereth("Demogorgon") is False

    def test_unknown_monster_assumed_respects(self, db):
        assert db.respects_elbereth("nonexistent thing") is True

    def test_elbereth_in_threat_report_matches(self, db):
        """elbereth_effective in ThreatReport should match respects_elbereth."""
        for name in ["giant ant", "minotaur", "Death", "shopkeeper",
                      "Angel", "cockatrice"]:
            r = db.assess_threat(name, _player())
            assert r.elbereth_effective == db.respects_elbereth(name), \
                f"Mismatch for {name}"


# ============================================================
# corpse_value
# ============================================================

class TestCorpseValue:
    def test_returns_corpse_report(self, db):
        r = db.corpse_value("giant ant", set())
        assert isinstance(r, CorpseReport)

    def test_wraith_gain_level(self, db):
        r = db.corpse_value("wraith", set())
        assert r.safe_to_eat is True
        assert r.special_effect == "gain level"
        assert r.priority == 10

    def test_floating_eye_telepathy(self, db):
        r = db.corpse_value("floating eye", set())
        assert r.safe_to_eat is True
        assert r.beneficial_intrinsic == "telepathy"
        assert r.intrinsic_probability == 1.0

    def test_cockatrice_unsafe_without_resistance(self, db):
        r = db.corpse_value("cockatrice", set())
        assert r.safe_to_eat is False

    def test_cockatrice_safe_with_resistance(self, db):
        r = db.corpse_value("cockatrice", {"stoning resistance"})
        assert r.safe_to_eat is True

    def test_killer_bee_unsafe_without_poison_res(self, db):
        r = db.corpse_value("killer bee", set())
        assert r.safe_to_eat is False

    def test_killer_bee_safe_with_poison_res(self, db):
        r = db.corpse_value("killer bee", {"poison resistance"})
        assert r.safe_to_eat is True
        assert r.beneficial_intrinsic == "poison resistance"

    def test_green_slime_never_eat(self, db):
        r = db.corpse_value("green slime", {"poison resistance", "fire resistance"})
        assert r.safe_to_eat is False

    def test_death_never_eat(self, db):
        r = db.corpse_value("Death", set())
        assert r.safe_to_eat is False

    def test_lizard_safe_and_useful(self, db):
        r = db.corpse_value("lizard", set())
        assert r.safe_to_eat is True
        assert r.priority >= 7

    def test_lichen_safe(self, db):
        r = db.corpse_value("lichen", set())
        assert r.safe_to_eat is True
        assert r.priority >= 4

    def test_newt_mana(self, db):
        r = db.corpse_value("newt", set())
        assert r.safe_to_eat is True
        assert "mana" in (r.special_effect or "").lower()

    def test_red_dragon_fire_res(self, db):
        r = db.corpse_value("red dragon", set())
        assert r.beneficial_intrinsic == "fire resistance"
        assert r.safe_to_eat is True
        assert r.priority >= 7

    def test_already_have_intrinsic_low_priority(self, db):
        r = db.corpse_value("red dragon", {"fire resistance"})
        assert r.priority <= 4

    def test_intrinsic_probability_formula(self, db):
        # Level 2 monster: 2/15
        r = db.corpse_value("fire ant", set())
        assert abs(r.intrinsic_probability - 3 / 15.0) < 0.01  # fire ant is level 3

    def test_intrinsic_probability_capped(self, db):
        # High level monster should cap at 1.0
        r = db.corpse_value("red dragon", set())
        assert r.intrinsic_probability <= 1.0

    def test_unknown_corpse_unsafe(self, db):
        r = db.corpse_value("imaginary beast", set())
        assert r.safe_to_eat is False
        assert r.priority == 1

    def test_dog_aggravate(self, db):
        r = db.corpse_value("dog", set())
        assert r.safe_to_eat is False  # aggravate monster

    def test_acid_blob_unsafe_without_acid_res(self, db):
        r = db.corpse_value("acid blob", set())
        assert r.safe_to_eat is False

    def test_acid_blob_safe_with_acid_res(self, db):
        r = db.corpse_value("acid blob", {"acid resistance"})
        assert r.safe_to_eat is True

    def test_mind_flayer_telepathy(self, db):
        r = db.corpse_value("mind flayer", set())
        assert r.beneficial_intrinsic == "telepathy"
        assert "+1 INT" in (r.special_effect or "")

    def test_tengu_teleport_control(self, db):
        r = db.corpse_value("tengu", set())
        assert r.beneficial_intrinsic == "teleport control"

    def test_black_dragon_disint_res(self, db):
        r = db.corpse_value("black dragon", set())
        assert r.beneficial_intrinsic == "disintegration resistance"

    def test_chameleon_polymorph(self, db):
        r = db.corpse_value("chameleon", set())
        assert "polymorph" in (r.special_effect or "").lower()

    def test_werewolf_lycanthropy(self, db):
        r = db.corpse_value("werewolf", set())
        assert "lycanthropy" in (r.special_effect or "").lower()


# ============================================================
# assess_room
# ============================================================

class TestAssessRoom:
    def test_empty_room(self, db):
        r = db.assess_room([], _player())
        assert isinstance(r, RoomThreatReport)
        assert r.total_danger == 0.0
        assert r.target_list == []
        assert r.flee_recommended is False

    def test_single_weak_monster(self, db):
        r = db.assess_room(["newt"], _player())
        assert r.total_danger > 0
        assert len(r.target_list) == 1
        assert r.flee_recommended is False

    def test_instakill_prioritized_first(self, db):
        r = db.assess_room(["giant ant", "cockatrice", "newt"], _player())
        assert r.target_list[0] == "cockatrice"

    def test_multiple_threats_high_danger(self, db):
        r = db.assess_room(
            ["arch-lich", "master mind flayer", "minotaur"],
            _player(level=5))
        assert r.total_danger >= 20

    def test_flee_when_critical_hp(self, db):
        r = db.assess_room(
            ["soldier ant", "soldier ant"],
            _player(hp=5, max_hp=50))
        assert r.flee_recommended is True

    def test_corridor_retreat_when_surrounded(self, db):
        r = db.assess_room(
            ["hill giant", "hill giant", "hill giant", "hill giant"],
            _player(hp=30, max_hp=50, has_elbereth_source=False))
        assert "corridor" in r.recommended_approach.lower() or \
               "retreat" in r.recommended_approach.lower()

    def test_elbereth_suggested_when_available(self, db):
        r = db.assess_room(
            ["hill giant", "hill giant", "hill giant"],
            _player(hp=15, max_hp=50, has_elbereth_source=True))
        assert "elbereth" in r.recommended_approach.lower()

    def test_ranged_approach_for_contact_dangers(self, db):
        r = db.assess_room(["floating eye", "newt"], _player())
        assert "ranged" in r.recommended_approach.lower()

    def test_target_list_length_matches_input(self, db):
        monsters = ["giant ant", "killer bee", "soldier ant"]
        r = db.assess_room(monsters, _player())
        assert len(r.target_list) == len(monsters)

    def test_danger_higher_with_more_monsters(self, db):
        r1 = db.assess_room(["soldier ant"], _player())
        r2 = db.assess_room(["soldier ant", "soldier ant", "soldier ant"], _player())
        assert r2.total_danger > r1.total_danger


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_all_384_monsters_assess_without_error(self, db):
        ps = _player()
        for name in db._monsters:
            r = db.assess_threat(name, ps)
            assert 1 <= r.danger_level <= 10
            assert r.recommended_action in {
                "melee", "ranged", "elbereth", "flee", "use_item"}

    def test_all_384_corpses_evaluate_without_error(self, db):
        for name in db._monsters:
            r = db.corpse_value(name, set())
            assert isinstance(r, CorpseReport)
            assert 1 <= r.priority <= 10

    def test_all_384_elbereth_check_without_error(self, db):
        for name in db._monsters:
            result = db.respects_elbereth(name)
            assert isinstance(result, bool)

    def test_zero_hp_player(self, db):
        """Edge case: player at 0 HP."""
        r = db.assess_threat("newt", _player(hp=0, max_hp=50))
        assert r.danger_level >= 1

    def test_zero_max_hp_player(self, db):
        """Edge case: prevent div by zero."""
        r = db.assess_threat("newt", _player(hp=0, max_hp=0))
        assert isinstance(r, ThreatReport)

    def test_empty_resistances(self, db):
        r = db.assess_threat("fire ant", _player(resistances=set()))
        assert "fire resistance" in r.required_resistances

    def test_player_with_all_resistances(self, db):
        all_res = {
            "fire resistance", "cold resistance", "shock resistance",
            "poison resistance", "sleep resistance", "acid resistance",
            "stoning resistance", "disintegration resistance",
            "magic resistance", "drain resistance",
        }
        r = db.assess_threat("fire ant", _player(resistances=all_res))
        # Should still work, just lower effective danger.
        assert isinstance(r, ThreatReport)
