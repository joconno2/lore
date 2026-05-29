#!/usr/bin/env python3
"""Extract structured game mechanics from NetHack 3.6.7 source into JSON.

Reads pray.c, eat.c, prop.h, and artilist.h. Outputs:
  - data/parsed/prayer_mechanics.json
  - data/parsed/corpse_effects.json
  - data/parsed/artifacts.json
"""

import json
import re
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "source", "NetHack")
OUT = os.path.join(BASE, "data", "parsed")


def read(path):
    with open(os.path.join(SRC, path)) as f:
        return f.read()


# --------------------------------------------------------------------------
# prop.h: intrinsic/property definitions
# --------------------------------------------------------------------------

def parse_props():
    text = read("include/prop.h")
    props = {}
    for m in re.finditer(r"(\w+)\s*=\s*(\d+)", text):
        props[int(m.group(2))] = m.group(1)
    return props


# --------------------------------------------------------------------------
# artilist.h: artifact definitions
# --------------------------------------------------------------------------

ATTACK_TYPES = {
    "AD_PHYS": "physical",
    "AD_DRLI": "level_drain",
    "AD_COLD": "cold",
    "AD_FIRE": "fire",
    "AD_ELEC": "electricity",
    "AD_STUN": "stun",
    "AD_MAGM": "magic_missile",
    "AD_BLND": "blinding",
    "AD_WERE": "lycanthropy_resistance",
}

SPFX_FLAGS = {
    "SPFX_NOGEN":    "not_randomly_generated",
    "SPFX_RESTR":    "restricted",
    "SPFX_INTEL":    "intelligent",
    "SPFX_SEEK":     "searching",
    "SPFX_WARN":     "warning",
    "SPFX_ATTK":     "has_attack",
    "SPFX_DEFN":     "has_defense",
    "SPFX_DRLI":     "drain_life_resistance",
    "SPFX_SEARCH":   "auto_searching",
    "SPFX_BEHEAD":   "beheading",
    "SPFX_HALRES":   "hallucination_resistance",
    "SPFX_ESP":      "telepathy",
    "SPFX_STLTH":    "stealth",
    "SPFX_REGEN":    "regeneration",
    "SPFX_EREGEN":   "energy_regeneration",
    "SPFX_HSPDAM":   "half_spell_damage",
    "SPFX_HPHDAM":   "half_physical_damage",
    "SPFX_TCTRL":    "teleport_control",
    "SPFX_LUCK":     "luck",
    "SPFX_XRAY":     "xray_vision",
    "SPFX_REFLECT":  "reflection",
    "SPFX_PROTECT":  "protection",
    "SPFX_SPEAK":    "speaking",
    "SPFX_DFLAG2":   "double_damage_vs_flag",
    "SPFX_DCLAS":    "double_damage_vs_class",
    "SPFX_DALIGN":   "damage_vs_cross_aligned",
}

INVOKE_POWERS = {
    "INVIS":          "invisibility",
    "LEVITATION":     "levitation",
    "CONFLICT":       "conflict",
    "HEALING":        "healing",
    "ENLIGHTENING":   "enlightening",
    "ENERGY_BOOST":   "energy_boost",
    "CREATE_AMMO":    "create_ammo",
    "UNTRAP":         "untrap",
    "CHARGE_OBJ":     "charge_object",
    "LEV_TELE":       "level_teleport",
    "CREATE_PORTAL":  "create_portal",
    "TAMING":         "taming",
}

ALIGNMENTS = {
    "A_NONE":    "unaligned",
    "A_LAWFUL":  "lawful",
    "A_NEUTRAL": "neutral",
    "A_CHAOTIC": "chaotic",
}

ROLES = {
    "PM_ARCHEOLOGIST": "Archeologist",
    "PM_BARBARIAN":    "Barbarian",
    "PM_CAVEMAN":      "Caveman",
    "PM_HEALER":       "Healer",
    "PM_KNIGHT":       "Knight",
    "PM_MONK":         "Monk",
    "PM_PRIEST":       "Priest",
    "PM_RANGER":       "Ranger",
    "PM_ROGUE":        "Rogue",
    "PM_SAMURAI":      "Samurai",
    "PM_TOURIST":      "Tourist",
    "PM_VALKYRIE":     "Valkyrie",
    "PM_WIZARD":       "Wizard",
    "NON_PM":          None,
}

RACES = {
    "PM_ELF":    "Elf",
    "PM_ORC":    "Orc",
    "PM_DWARF":  "Dwarf",
    "PM_HUMAN":  "Human",
    "PM_GNOME":  "Gnome",
    "NON_PM":    None,
}

TARGET_FLAGS = {
    "M2_ELF":    "elves",
    "M2_ORC":    "orcs",
    "M2_DEMON":  "demons",
    "M2_WERE":   "werebeasts",
    "M2_GIANT":  "giants",
    "M2_UNDEAD": "undead",
    "S_DRAGON":  "dragons",
    "S_OGRE":    "ogres",
    "S_TROLL":   "trolls",
}

BASE_ITEMS = {
    "LONG_SWORD":        "long sword",
    "RUNESWORD":         "runesword",
    "WAR_HAMMER":        "war hammer",
    "BATTLE_AXE":        "battle-axe",
    "ORCISH_DAGGER":     "orcish dagger",
    "ELVEN_BROADSWORD":  "elven broadsword",
    "ELVEN_DAGGER":      "elven dagger",
    "ATHAME":            "athame",
    "BROADSWORD":        "broadsword",
    "SILVER_SABER":      "silver saber",
    "MORNING_STAR":      "morning star",
    "KATANA":            "katana",
    "TSURUGI":           "tsurugi",
    "CRYSTAL_BALL":      "crystal ball",
    "LUCKSTONE":         "luckstone",
    "MACE":              "mace",
    "QUARTERSTAFF":      "quarterstaff",
    "MIRROR":            "mirror",
    "LENSES":            "lenses",
    "HELM_OF_BRILLIANCE":"helm of brilliance",
    "BOW":               "bow",
    "SKELETON_KEY":      "skeleton key",
    "CREDIT_CARD":       "credit card",
    "AMULET_OF_ESP":     "amulet of ESP",
}


def parse_attack_tuple(s):
    """Parse {0,AD_PHYS,5,10} or NO_ATTK etc."""
    s = s.strip()
    if s in ("NO_ATTK", "NO_DFNS", "NO_CARY"):
        return None
    # Match macro invocations like PHYS(5,10), DRLI(0,0), DFNS(AD_MAGM), CARY(AD_FIRE)
    m = re.match(r"(\w+)\(([^)]*)\)", s)
    if m:
        macro = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        type_map = {
            "PHYS": "AD_PHYS", "DRLI": "AD_DRLI", "COLD": "AD_COLD",
            "FIRE": "AD_FIRE", "ELEC": "AD_ELEC", "STUN": "AD_STUN",
            "DFNS": None, "CARY": None,
        }
        if macro in ("DFNS", "CARY"):
            return {"type": ATTACK_TYPES.get(args[0], args[0])}
        atype = type_map.get(macro, "unknown")
        result = {"type": ATTACK_TYPES.get(atype, atype)}
        if len(args) >= 2:
            d1, d2 = int(args[0]), int(args[1])
            if d1 or d2:
                result["damage"] = f"+{d1}d{d2}" if d2 else f"+{d1}"
        return result
    return None


def parse_spfx(s):
    """Extract SPFX flags from a parenthesized expression."""
    flags = []
    for flag in SPFX_FLAGS:
        if flag in s:
            flags.append(SPFX_FLAGS[flag])
    return flags


def parse_artifacts():
    text = read("include/artilist.h")

    # Remove comments but keep content
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'#if 0.*?#endif', '', text, flags=re.DOTALL)

    artifacts = []

    # Find all A(...) invocations
    # Strategy: find A(" and then balance parens
    pattern = r'A\("([^"]+)"'
    for m in re.finditer(pattern, text):
        name = m.group(1)
        if not name:
            continue

        # Extract full A(...) call by balancing parens from the A(
        start = m.start()
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        call = text[start:end]
        # Tokenize: split by comma but respect nested parens
        tokens = []
        depth = 0
        current = ""
        # Skip "A(" prefix
        inner = call[2:-1]  # strip A( and )
        for ch in inner:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                tokens.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            tokens.append(current.strip())

        if len(tokens) < 14:
            continue

        # tokens: name, typ, s1, s2, mt, atk, dfn, cry, inv, al, cl, rac, cost, clr
        raw_name = tokens[0].strip('"')
        base_type = tokens[1].strip()
        s1 = tokens[2]  # special flags 1
        s2 = tokens[3]  # special flags 2
        mt = tokens[4]  # monster type target
        atk = tokens[5]
        dfn = tokens[6]
        cry = tokens[7]  # carry effect
        inv = tokens[8]  # invoke power
        al = tokens[9]
        cl = tokens[10]   # class/role restriction
        rac = tokens[11]  # race restriction
        cost = tokens[12]
        # clr = tokens[13]

        art = {"name": raw_name}
        art["base_item"] = BASE_ITEMS.get(base_type, base_type.lower().replace("_", " "))

        # Flags from s1 and s2
        all_flags = parse_spfx(s1) + parse_spfx(s2)
        if all_flags:
            art["properties"] = all_flags

        # Target type
        mt = mt.strip()
        if mt != "0" and mt in TARGET_FLAGS:
            art["effective_against"] = TARGET_FLAGS[mt]

        # Attack
        atk_parsed = parse_attack_tuple(atk)
        if atk_parsed:
            art["attack"] = atk_parsed

        # Defense
        dfn_parsed = parse_attack_tuple(dfn)
        if dfn_parsed:
            art["defense"] = dfn_parsed

        # Carry effect
        cry_parsed = parse_attack_tuple(cry)
        if cry_parsed:
            art["carry_effect"] = cry_parsed

        # Invoke power
        inv = inv.strip()
        if inv != "0":
            art["invoke"] = INVOKE_POWERS.get(inv, inv.lower())

        # Alignment
        al = al.strip()
        art["alignment"] = ALIGNMENTS.get(al, al)

        # Role restriction
        cl = cl.strip()
        art["role"] = ROLES.get(cl)

        # Race restriction
        rac = rac.strip()
        art["race"] = RACES.get(rac)

        # Cost
        cost_str = cost.strip().rstrip("L")
        try:
            art["cost"] = int(cost_str)
        except ValueError:
            pass

        # Is quest artifact?
        art["quest_artifact"] = "not_randomly_generated" in all_flags and "intelligent" in all_flags

        artifacts.append(art)

    return artifacts


# --------------------------------------------------------------------------
# pray.c: prayer mechanics
# --------------------------------------------------------------------------

def build_prayer_mechanics():
    return {
        "prayer_timeout": {
            "description": "Prayer timeout (ublesscnt) controls when you can safely pray again.",
            "initial_value": "Set by previous prayer or sacrifice outcomes.",
            "resets_to": {
                "after_pleased": "rnz(350) base, plus rnz(1000) per kick_on_butt factor (demigod status, crowned)",
                "after_angry": "rnz(300)",
                "after_too_soon": "current + rnz(250)",
                "after_sacrifice_gift": "rnz(300 + 50*nartifacts)"
            },
            "decreases_by": {
                "sacrifice": "value * (500 for chaotic, 300 for others) / MAXVALUE(24), subtracted from ublesscnt"
            },
            "notes": "ublesscnt decrements by 1 each turn naturally."
        },

        "prayer_safety_conditions": {
            "safe_prayer_requires_all": [
                "ublesscnt <= 0 (or <= 100 for minor trouble, <= 200 for major trouble)",
                "Luck >= 0",
                "ugangr == 0 (god not angry)",
                "alignment record >= 0",
                "On coaligned altar or no altar",
                "Not in Gehennom",
                "Not undead (unless chaotic, or in Gehennom)"
            ],
            "p_type_values": {
                "-1": "Undead praying to lawful/neutral god outside Gehennom. God turns you, deals rnd(20) damage, forces rehumanize.",
                "0": "Too soon. Timeout not expired. God gets upset, adds rnz(250) to timeout, -3 luck.",
                "1": "Naughty. Negative luck, god angry, or negative alignment. Triggers angrygods().",
                "2": "On non-coaligned altar. If water on altar, curses it and god gets upset. Otherwise pleased().",
                "3": "Safe. Coaligned altar or no altar, all conditions met. Grants invulnerability during prayer, calls pleased()."
            },
            "invulnerability": "p_type==3 and not in Gehennom grants invulnerability during the 3-turn prayer."
        },

        "major_troubles": {
            "description": "Checked in priority order. Positive trouble values. Guaranteed fix if Luck >= 0.",
            "list": [
                {"id": "TROUBLE_STONED",             "value": 14, "condition": "Stoned",                                "fix": "Cure stoning"},
                {"id": "TROUBLE_SLIMED",             "value": 13, "condition": "Slimed",                                "fix": "Remove slime"},
                {"id": "TROUBLE_STRANGLED",          "value": 12, "condition": "Strangled",                             "fix": "Remove strangulation (destroys amulet if worn)"},
                {"id": "TROUBLE_LAVA",               "value": 11, "condition": "Trapped in lava",                       "fix": "Teleport to safety"},
                {"id": "TROUBLE_SICK",               "value": 10, "condition": "Sick",                                  "fix": "Cure sickness"},
                {"id": "TROUBLE_STARVING",           "value":  9, "condition": "Hunger >= WEAK",                        "fix": "Set hunger to 900 (not hungry)"},
                {"id": "TROUBLE_REGION",             "value":  8, "condition": "In dangerous region (stinking cloud)",  "fix": "Teleport out of region"},
                {"id": "TROUBLE_HIT",                "value":  7, "condition": "Critically low HP (<=5 or HP/maxHP ratio)", "fix": "Full heal, possibly raise maxHP by rnd(5)"},
                {"id": "TROUBLE_LYCANTHROPE",        "value":  6, "condition": "Lycanthropy",                           "fix": "Cure lycanthropy"},
                {"id": "TROUBLE_COLLAPSING",         "value":  5, "condition": "Extreme encumbrance + STR loss > 3",    "fix": "Restore STR to max, uncurse sustain ability ring"},
                {"id": "TROUBLE_STUCK_IN_WALL",      "value":  4, "condition": "Surrounded by impassible rock",         "fix": "Teleport out, or grant temporary phasing"},
                {"id": "TROUBLE_CURSED_LEVITATION",  "value":  3, "condition": "Cursed levitation boots/ring",          "fix": "Uncurse the item"},
                {"id": "TROUBLE_UNUSEABLE_HANDS",    "value":  2, "condition": "Welded weapon or nohands polyform",     "fix": "Uncurse weapon, or rehumanize"},
                {"id": "TROUBLE_CURSED_BLINDFOLD",   "value":  1, "condition": "Cursed blindfold worn",                 "fix": "Uncurse blindfold"}
            ]
        },

        "minor_troubles": {
            "description": "Negative trouble values. Fixed only with higher prayer_luck or DEVOUT alignment.",
            "list": [
                {"id": "TROUBLE_PUNISHED",       "value": -1,  "condition": "Punished (ball and chain) or buried ball", "fix": "Remove chain, free from buried ball"},
                {"id": "TROUBLE_FUMBLING",       "value": -2,  "condition": "Cursed gauntlets of fumbling or fumble boots", "fix": "Uncurse the item"},
                {"id": "TROUBLE_CURSED_ITEMS",   "value": -3,  "condition": "Wearing/carrying cursed items (prioritized list)", "fix": "Uncurse worst cursed item"},
                {"id": "TROUBLE_SADDLE",         "value": -4,  "condition": "Cursed saddle on steed",                   "fix": "Uncurse saddle"},
                {"id": "TROUBLE_BLIND",          "value": -5,  "condition": "Blinded > 1 (not from engulfer) or timed deafness", "fix": "Cure blindness and deafness"},
                {"id": "TROUBLE_POISONED",       "value": -6,  "condition": "Any attribute below max",                  "fix": "Restore all attributes to max"},
                {"id": "TROUBLE_WOUNDED_LEGS",   "value": -7,  "condition": "Wounded legs (not riding)",                "fix": "Heal legs"},
                {"id": "TROUBLE_HUNGRY",         "value": -8,  "condition": "Hunger >= HUNGRY (but < WEAK)",            "fix": "Set hunger to 900"},
                {"id": "TROUBLE_STUNNED",        "value": -9,  "condition": "Timed stun",                               "fix": "Cure stun"},
                {"id": "TROUBLE_CONFUSED",       "value": -10, "condition": "Timed confusion",                          "fix": "Cure confusion"},
                {"id": "TROUBLE_HALLUCINATION",  "value": -11, "condition": "Timed hallucination",                      "fix": "Cure hallucination"}
            ]
        },

        "pleased_action_resolution": {
            "description": "When god is pleased, action level determines what gets fixed.",
            "action_formula": "action = rn1(prayer_luck + (on_altar ? 3 + on_shrine : 2), 1); capped at 3 if not on altar",
            "low_alignment_override": "If alignment < STRIDENT(4): action = 1 if record > 0 or 50% chance, else 0",
            "action_effects": {
                "0": "Nothing. God blows you off.",
                "1": "Fix worst trouble if major (positive value).",
                "2": "Fix all major troubles (up to 10 iterations).",
                "3": "Fix worst trouble even if minor.",
                "4": "Fix ALL troubles (major and minor).",
                "5": "Fix all troubles + gratuitous favor (pat on head)."
            }
        },

        "gratuitous_favors": {
            "description": "Pat-on-head effects, chosen by rn2((Luck+6)/2).",
            "effects": {
                "0": "Nothing extra.",
                "1": "Weapon: uncurse, bless, repair erosion.",
                "2": "Golden glow: restore lost levels (or +5 maxHP), full heal, restore STR, cure hunger, fix luck, cure blindness.",
                "3": "Castle tune hint (first time) or tune itself (second time). Falls through to 2 if already known.",
                "4": "Uncurse entire inventory (except helm of opposite alignment).",
                "5": "Grant intrinsic: telepathy > speed > stealth > protection (in that priority).",
                "6": "Receive a blessed spellbook.",
                "7-8": "Crowning (if PIOUS alignment and not already crowned). Otherwise spellbook."
            }
        },

        "crowning": {
            "description": "Highest prayer reward. Grants title, intrinsics, and possibly an artifact weapon.",
            "requirements": "alignment record >= PIOUS(20), not already crowned",
            "intrinsics_granted": ["see_invisible", "fire_resistance", "cold_resistance", "shock_resistance", "sleep_resistance", "poison_resistance"],
            "titles": {
                "lawful": "The Hand of Elbereth",
                "neutral": "Envoy of Balance",
                "chaotic": "Chosen (to steal souls / take lives)"
            },
            "gifts": {
                "lawful": "Excalibur (if wielding long sword) or spellbook of Finger of Death (wizard) / Restore Ability (monk)",
                "neutral": "Vorpal Blade (if no artifact weapon wielded)",
                "chaotic": "Stormbringer (if no artifact weapon wielded)"
            },
            "additional": "Weapon blessed, erodeproofed, spe raised to at least +1. Extra weapon skill slot granted."
        },

        "sacrifice_mechanics": {
            "corpse_value": {
                "formula": "mons[corpsenm].difficulty + 1",
                "freshness_required": "monstermoves <= corpse_age + 50 (or acid blob, which never rots for sacrifice purposes)",
                "eaten_penalty": "If partly eaten, value reduced proportionally via eaten_stat()",
                "MAXVALUE": 24
            },
            "modifiers": {
                "undead_corpse": "+1 value if you are not chaotic",
                "pet_corpse": "value = -1, aggravate monster, -3 alignment",
                "unicorn_same_alignment_as_altar": "Very bad. -1 WIS, value = -5.",
                "unicorn_your_alignment_on_your_altar": "Very good. +5 alignment, +3 value.",
                "unicorn_your_alignment_on_cross_altar": "Conversion trigger. alignment record set to -1.",
                "unicorn_other": "+3 value (ordinary bonus)."
            },
            "effects_when_value_positive": {
                "god_angry": "Reduce ugangr by value * (2 for chaotic, 3 for others) / MAXVALUE",
                "negative_alignment": "Add value to alignment record (partial absolution)",
                "timeout_remaining": "Reduce ublesscnt by value * (500 chaotic, 300 others) / MAXVALUE",
                "good_standing": "Chance of artifact gift: !rn2(10 + 2*ugifts*nartifacts). Otherwise luck boost: value*LUCKMAX/(MAXVALUE*2)."
            },
            "human_sacrifice": {
                "chaotic_on_chaotic_altar": "+5 alignment, +2 luck, may summon demon lord",
                "non_chaotic": "-5 alignment, +3 god anger, -1 WIS, -5 luck, angers own god",
                "demon_player": "Finds it satisfying, gains WIS exercise"
            },
            "conversion": {
                "condition": "Sacrificing on cross-aligned altar while your god is angry and you haven't previously converted",
                "effect": "Change alignment to altar's, -3 luck, +300 to prayer timeout"
            }
        },

        "god_anger": {
            "ugangr": "Counter for how angry your god is. Incremented by gods_upset(), decremented by sacrifice.",
            "effects_of_anger": {
                "maxanger_formula": {
                    "same_god": "3 * ugangr + luck_modifier",
                    "different_god": "alignment_record / 2 + luck_modifier",
                    "range": "1 to 15"
                },
                "rn2_maxanger_outcomes": {
                    "0-1": "God is displeased (message only).",
                    "2-3": "Lose 1 WIS, lose experience level.",
                    "4-5": "Random curse on inventory.",
                    "6": "Punished (ball and chain). Falls to 4-5 if already punished.",
                    "7-8": "Summon hostile minion.",
                    "9+": "God zaps you: lightning bolt then wide-angle disintegration beam. Can destroy armor."
                }
            },
            "prayer_too_soon_penalty": "+rnz(250) to ublesscnt, -3 luck, gods_upset()"
        },

        "alignment_thresholds": {
            "PIOUS":    20,
            "DEVOUT":   14,
            "FERVENT":   9,
            "STRIDENT":  4,
            "description": "Alignment record thresholds that affect prayer outcomes and god messages."
        },

        "gehennom": {
            "description": "Praying in Gehennom always fails. God can't help you. Likely angrygods() if low alignment or unlucky roll."
        },

        "water_prayer": {
            "description": "If praying on an altar with potions of water on the floor, they become holy (coaligned) or unholy (non-coaligned) water."
        }
    }


# --------------------------------------------------------------------------
# eat.c: corpse effects
# --------------------------------------------------------------------------

def build_corpse_effects():
    return {
        "intrinsic_gain_system": {
            "description": "After eating a corpse, the game picks one possible intrinsic at random from those the monster can convey, then rolls to see if you get it.",
            "possible_intrinsics": {
                "FIRE_RES":   {"check": "ptr->mconveys & MR_FIRE",   "chance_denominator": 15},
                "COLD_RES":   {"check": "ptr->mconveys & MR_COLD",   "chance_denominator": 15},
                "SLEEP_RES":  {"check": "ptr->mconveys & MR_SLEEP",  "chance_denominator": 15},
                "DISINT_RES": {"check": "ptr->mconveys & MR_DISINT", "chance_denominator": 15},
                "SHOCK_RES":  {"check": "ptr->mconveys & MR_ELEC",   "chance_denominator": 15},
                "POISON_RES": {"check": "ptr->mconveys & MR_POISON", "chance_denominator": 15,
                               "exception": "Killer bee and scorpion: 75% chance to use chance=1 instead of 15"},
                "TELEPORT":   {"check": "can_teleport(ptr)",          "chance_denominator": 10},
                "TELEPORT_CONTROL": {"check": "control_teleport(ptr)", "chance_denominator": 12},
                "TELEPAT":    {"check": "telepathic(ptr)",            "chance_denominator": 1}
            },
            "gain_formula": "if (ptr->mlevel <= rn2(chance)) return; // failed. So P(gain) = min(mlevel, chance) / chance for mlevel < chance, else guaranteed.",
            "selection": "If monster conveys multiple intrinsics, one is chosen uniformly at random (reservoir sampling). Strength from giants counted as an extra candidate.",
            "strength_from_giants": "Treated as an intrinsic candidate. If it's the only one, 50% chance of being skipped entirely. Uses gainstr()."
        },

        "special_corpse_effects": {
            "cprefx_effects": {
                "description": "Effects that happen when you START eating the corpse.",
                "petrifying_corpses": {
                    "monsters": ["cockatrice", "chickatrice", "Medusa"],
                    "effect": "Instant death by stoning unless Stone_resistance or can polymorph into stone golem."
                },
                "dogs_and_cats": {
                    "monsters": ["little dog", "dog", "large dog", "kitten", "housecat", "large cat"],
                    "effect": "Aggravate monster intrinsic granted (permanent). No penalty for cavemen/orcs."
                },
                "lizard": {
                    "effect": "Cures stoning (fix_petrification)."
                },
                "riders": {
                    "monsters": ["Death", "Pestilence", "Famine"],
                    "effect": "Instantly fatal. Corpse revives. Life-saving required to survive."
                },
                "green_slime": {
                    "effect": "Sliming begins (10 turns) unless already slimed, unchanging, or slimeproof."
                },
                "acidic_monsters": {
                    "effect": "If stoned, cures petrification (like lizard)."
                },
                "cannibalism": {
                    "condition": "Eating own race or polymorphed species (not cavemen/orcs).",
                    "effect": "Aggravate monster intrinsic, -2 to -5 luck."
                }
            },

            "cpostfx_effects": {
                "description": "Effects that happen AFTER completely eating the corpse.",
                "newt": {
                    "effect": "2/3 chance (or if energy <= 2/3 max): gain rnd(3) energy. 1/3 chance to increase max energy by 1."
                },
                "wraith": {
                    "effect": "Gain one experience level."
                },
                "human_werebeasts": {
                    "monsters": ["human wererat", "human werejackal", "human werewolf"],
                    "effect": "Contract lycanthropy of corresponding type."
                },
                "nurse": {
                    "effect": "Full HP heal, cure blindness. Also checks for intrinsic conveyance (poison resistance)."
                },
                "stalker": {
                    "effect": "If not already invisible: gain temporary invisibility (50-149 turns) and see_invisible. OR permanent invisibility + see_invisible if you have the intrinsic already."
                },
                "yellow_light": {
                    "effect": "Stun for 30 additional turns."
                },
                "bat_and_giant_bat": {
                    "effect": "Stun for 30 turns (bat) or 60 turns (giant bat, includes yellow_light fallthrough)."
                },
                "mimics": {
                    "monsters": ["small mimic (+20 turns)", "large mimic (+40 turns)", "giant mimic (+50 turns)"],
                    "effect": "Forces you to mimic a pile of gold (or orange if hallucinating) for duration. Cannot ride."
                },
                "quantum_mechanic": {
                    "effect": "Toggles intrinsic speed. If fast, become slow. If slow, become fast."
                },
                "lizard": {
                    "effect": "Reduce stun to 2 turns max, reduce confusion to 2 turns max."
                },
                "shapeshifters": {
                    "monsters": ["chameleon", "doppelganger", "sandestin"],
                    "effect": "Polymorph self (unless unchanging)."
                },
                "disenchanter": {
                    "effect": "Randomly strips one intrinsic (attrcurse)."
                },
                "mind_flayer": {
                    "monsters": ["mind flayer", "master mind flayer"],
                    "effect": "50% chance to gain +1 INT (if below max). If no INT gain, falls through to intrinsic check (telepathy possible)."
                },
                "hallucination_inducers": {
                    "condition": "Monster has AD_STUN or AD_HALU attack, or is violet fungus.",
                    "effect": "+200 turns of hallucination."
                }
            }
        },

        "corpse_freshness": {
            "formula": "rotted = (monstermoves - corpse_age) / (10 + rn2(20))",
            "cursed_modifier": "+2 to rotted value",
            "blessed_modifier": "-2 to rotted value",
            "nonrotting_corpses": ["lizard", "lichen", "Death", "Pestilence", "Famine"],
            "tainted_threshold": "rotted > 5: tainted, causes food poisoning (rn1(10,10) turns) unless Sick_resistance",
            "mildly_sick": "rotted > 5, or (rotted > 3 and 80% chance): lose rnd(8) HP",
            "rotten_food_chance": "If not tainted/poisonous/acidic and (orotten flag set or 1/7 chance): rotten food effects (confusion, blindness, or fainting)"
        },

        "poisonous_corpses": {
            "description": "Determined by poisonous() macro on the monster's permonst data. 80% chance of triggering poison effect.",
            "effect_without_resistance": "Lose rnd(4) STR, lose rnd(15) HP.",
            "effect_with_resistance": "Message only, no damage.",
            "common_examples": ["killer bee", "scorpion", "pit viper", "cobra", "water moccasin",
                                "black naga", "golden naga hatchling", "baby purple worm", "purple worm",
                                "quasit", "asp", "giant spider", "green slime"]
        },

        "acidic_corpses": {
            "description": "Determined by acidic() macro on monster data. Always triggers (no probability roll).",
            "effect_without_resistance": "Lose rnd(15) HP from stomach acid.",
            "effect_with_resistance": "No damage.",
            "cure_stoning": "Acidic corpses also cure stoning if eaten while turning to stone (same as lizard).",
            "common_examples": ["acid blob", "yellow light", "gelatinous cube", "blue jelly", "ochre jelly"]
        },

        "tin_effects": {
            "tin_varieties": [
                {"type": "rotten",      "nutrition": -50, "notes": "Causes vomiting (rn1(15,10) turns). Always rotten if cursed."},
                {"type": "homemade",    "nutrition":  50, "notes": "Health food. 1/7 chance of going rotten if not blessed."},
                {"type": "soup",        "nutrition":  20, "notes": "Health food."},
                {"type": "french fried","nutrition":  40, "notes": "Greasy. Makes hands slippery (5-15 turns)."},
                {"type": "pickled",     "nutrition":  40, "notes": "Health food."},
                {"type": "boiled",      "nutrition":  50, "notes": "Health food."},
                {"type": "smoked",      "nutrition":  50, "notes": "Health food."},
                {"type": "dried",       "nutrition":  55, "notes": "Health food."},
                {"type": "deep fried",  "nutrition":  60, "notes": "Greasy."},
                {"type": "szechuan",    "nutrition":  70, "notes": "Health food."},
                {"type": "broiled",     "nutrition":  80, "notes": ""},
                {"type": "stir fried",  "nutrition":  80, "notes": "Greasy."},
                {"type": "sauteed",     "nutrition":  95, "notes": ""},
                {"type": "candied",     "nutrition": 100, "notes": "Health food."},
                {"type": "pureed",      "nutrition": 500, "notes": "Health food."}
            ],
            "spinach_tin": {
                "nutrition": {
                    "blessed": 600,
                    "uncursed": "400 + rnd(200)",
                    "cursed": "200 + rnd(400)"
                },
                "effect": "Grants strength increase (gainstr). Described as feeling like Popeye."
            },
            "booby_trapped": "otrapped flag or (cursed and not homemade and 1/8 chance): explodes.",
            "corpse_effects_apply": "Eating tinned monster meat applies both cprefx and cpostfx, so intrinsics can be gained.",
            "nonrotting_tins": "Lizard and lichen tins are always at least homemade quality (never rotten).",
            "opening_time": {
                "blessed_tin": "0-1 turns",
                "tin_opener": "0-2 turns (depends on BUC of opener)",
                "dagger_or_knife": "3 turns",
                "pick_axe_or_axe": "6 turns",
                "bare_hands": "rn1(1 + 500/(DEX+STR), 10) turns"
            }
        },

        "nonrotting_food": {
            "description": "Lembas wafer and cram ration do not rot. However, if cursed, they behave as rotten.",
            "lembas_nutrition": {
                "base": 800,
                "elf_bonus": "+25% (1000 total)",
                "orc_penalty": "-25% (600 total)"
            },
            "cram_nutrition": {
                "base": 600,
                "dwarf_bonus": "+16.7% (700 total)"
            }
        }
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)

    # Artifacts
    artifacts = parse_artifacts()
    with open(os.path.join(OUT, "artifacts.json"), "w") as f:
        json.dump({"source": "NetHack 3.6.7 include/artilist.h",
                   "artifact_count": len(artifacts),
                   "artifacts": artifacts}, f, indent=2)
    print(f"Wrote {len(artifacts)} artifacts to artifacts.json")

    # Prayer mechanics
    prayer = build_prayer_mechanics()
    with open(os.path.join(OUT, "prayer_mechanics.json"), "w") as f:
        json.dump({"source": "NetHack 3.6.7 src/pray.c", "mechanics": prayer}, f, indent=2)
    print("Wrote prayer_mechanics.json")

    # Corpse effects
    corpse = build_corpse_effects()
    with open(os.path.join(OUT, "corpse_effects.json"), "w") as f:
        json.dump({"source": "NetHack 3.6.7 src/eat.c", "mechanics": corpse}, f, indent=2)
    print("Wrote corpse_effects.json")


if __name__ == "__main__":
    main()
