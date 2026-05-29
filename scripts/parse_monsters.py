#!/usr/bin/env python3
"""Parse NetHack 3.6.7 monster database from monst.c into structured JSON."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

SRCDIR = Path(__file__).resolve().parent.parent / "data" / "source" / "NetHack"
MONST_C = SRCDIR / "src" / "monst.c"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "parsed" / "monsters.json"

# ---------------------------------------------------------------------------
# Lookup tables built from the header files
# ---------------------------------------------------------------------------

SYMBOL_CLASSES = {
    "S_ANT": 1, "S_BLOB": 2, "S_COCKATRICE": 3, "S_DOG": 4, "S_EYE": 5,
    "S_FELINE": 6, "S_GREMLIN": 7, "S_HUMANOID": 8, "S_IMP": 9,
    "S_JELLY": 10, "S_KOBOLD": 11, "S_LEPRECHAUN": 12, "S_MIMIC": 13,
    "S_NYMPH": 14, "S_ORC": 15, "S_PIERCER": 16, "S_QUADRUPED": 17,
    "S_RODENT": 18, "S_SPIDER": 19, "S_TRAPPER": 20, "S_UNICORN": 21,
    "S_VORTEX": 22, "S_WORM": 23, "S_XAN": 24, "S_LIGHT": 25,
    "S_ZRUTY": 26, "S_ANGEL": 27, "S_BAT": 28, "S_CENTAUR": 29,
    "S_DRAGON": 30, "S_ELEMENTAL": 31, "S_FUNGUS": 32, "S_GNOME": 33,
    "S_GIANT": 34, "S_invisible": 35, "S_JABBERWOCK": 36, "S_KOP": 37,
    "S_LICH": 38, "S_MUMMY": 39, "S_NAGA": 40, "S_OGRE": 41,
    "S_PUDDING": 42, "S_QUANTMECH": 43, "S_RUSTMONST": 44, "S_SNAKE": 45,
    "S_TROLL": 46, "S_UMBER": 47, "S_VAMPIRE": 48, "S_WRAITH": 49,
    "S_XORN": 50, "S_YETI": 51, "S_ZOMBIE": 52, "S_HUMAN": 53,
    "S_GHOST": 54, "S_GOLEM": 55, "S_DEMON": 56, "S_EEL": 57,
    "S_LIZARD": 58,
    "S_WORM_TAIL": 59,
    "S_MIMIC_DEF": 60,
}

SYMBOL_CHARS = {
    1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h",
    9: "i", 10: "j", 11: "k", 12: "l", 13: "m", 14: "n", 15: "o",
    16: "p", 17: "q", 18: "r", 19: "s", 20: "t", 21: "u", 22: "v",
    23: "w", 24: "x", 25: "y", 26: "z", 27: "A", 28: "B", 29: "C",
    30: "D", 31: "E", 32: "F", 33: "G", 34: "H", 35: "I", 36: "J",
    37: "K", 38: "L", 39: "M", 40: "N", 41: "O", 42: "P", 43: "Q",
    44: "R", 45: "S", 46: "T", 47: "U", 48: "V", 49: "W", 50: "X",
    51: "Y", 52: "Z", 53: "@", 54: " ", 55: "'", 56: "&", 57: ";",
    58: ":",
    59: "~",
    60: "]",
}

SYMBOL_CLASS_NAMES = {v: k for k, v in SYMBOL_CLASSES.items()}

ATTACK_TYPES = {
    -1: "AT_ANY", 0: "AT_NONE", 1: "AT_CLAW", 2: "AT_BITE", 3: "AT_KICK",
    4: "AT_BUTT", 5: "AT_TUCH", 6: "AT_STNG", 7: "AT_HUGS",
    10: "AT_SPIT", 11: "AT_ENGL", 12: "AT_BREA", 13: "AT_EXPL",
    14: "AT_BOOM", 15: "AT_GAZE", 16: "AT_TENT", 254: "AT_WEAP",
    255: "AT_MAGC",
}

DAMAGE_TYPES = {
    -1: "AD_ANY", 0: "AD_PHYS", 1: "AD_MAGM", 2: "AD_FIRE", 3: "AD_COLD",
    4: "AD_SLEE", 5: "AD_DISN", 6: "AD_ELEC", 7: "AD_DRST", 8: "AD_ACID",
    9: "AD_SPC1", 10: "AD_SPC2", 11: "AD_BLND", 12: "AD_STUN",
    13: "AD_SLOW", 14: "AD_PLYS", 15: "AD_DRLI", 16: "AD_DREN",
    17: "AD_LEGS", 18: "AD_STON", 19: "AD_STCK", 20: "AD_SGLD",
    21: "AD_SITM", 22: "AD_SEDU", 23: "AD_TLPT", 24: "AD_RUST",
    25: "AD_CONF", 26: "AD_DGST", 27: "AD_HEAL", 28: "AD_WRAP",
    29: "AD_WERE", 30: "AD_DRDX", 31: "AD_DRCO", 32: "AD_DRIN",
    33: "AD_DISE", 34: "AD_DCAY", 35: "AD_SSEX", 36: "AD_HALU",
    37: "AD_DETH", 38: "AD_PEST", 39: "AD_FAMN", 40: "AD_SLIM",
    41: "AD_ENCH", 42: "AD_CORR", 240: "AD_CLRC", 241: "AD_SPEL",
    242: "AD_RBRE", 252: "AD_SAMU", 253: "AD_CURS",
}

SOUNDS = {
    0: "MS_SILENT", 1: "MS_BARK", 2: "MS_MEW", 3: "MS_ROAR", 4: "MS_GROWL",
    5: "MS_SQEEK", 6: "MS_SQAWK", 7: "MS_HISS", 8: "MS_BUZZ",
    9: "MS_GRUNT", 10: "MS_NEIGH", 11: "MS_WAIL", 12: "MS_GURGLE",
    13: "MS_BURBLE", 15: "MS_SHRIEK", 16: "MS_BONES", 17: "MS_LAUGH",
    18: "MS_MUMBLE", 19: "MS_IMITATE", 20: "MS_HUMANOID", 21: "MS_ARREST",
    22: "MS_SOLDIER", 23: "MS_GUARD", 24: "MS_DJINNI", 25: "MS_NURSE",
    26: "MS_SEDUCE", 27: "MS_VAMPIRE", 28: "MS_BRIBE", 29: "MS_CUSS",
    30: "MS_RIDER", 31: "MS_LEADER", 32: "MS_NEMESIS", 33: "MS_GUARDIAN",
    34: "MS_SELL", 35: "MS_ORACLE", 36: "MS_PRIEST", 37: "MS_SPELL",
    38: "MS_WERE", 39: "MS_BOAST",
}

SIZES = {
    0: "MZ_TINY", 1: "MZ_SMALL", 2: "MZ_MEDIUM", 3: "MZ_LARGE",
    4: "MZ_HUGE", 7: "MZ_GIGANTIC",
}

COLORS = {
    0: "CLR_BLACK", 1: "CLR_RED", 2: "CLR_GREEN", 3: "CLR_BROWN",
    4: "CLR_BLUE", 5: "CLR_MAGENTA", 6: "CLR_CYAN", 7: "CLR_GRAY",
    8: "NO_COLOR", 9: "CLR_ORANGE", 10: "CLR_BRIGHT_GREEN",
    11: "CLR_YELLOW", 12: "CLR_BRIGHT_BLUE", 13: "CLR_BRIGHT_MAGENTA",
    14: "CLR_BRIGHT_CYAN", 15: "CLR_WHITE",
}

MR_FLAGS = {
    0x01: "MR_FIRE", 0x02: "MR_COLD", 0x04: "MR_SLEEP", 0x08: "MR_DISINT",
    0x10: "MR_ELEC", 0x20: "MR_POISON", 0x40: "MR_ACID", 0x80: "MR_STONE",
}

M1_FLAGS = {
    0x00000001: "M1_FLY", 0x00000002: "M1_SWIM",
    0x00000004: "M1_AMORPHOUS", 0x00000008: "M1_WALLWALK",
    0x00000010: "M1_CLING", 0x00000020: "M1_TUNNEL",
    0x00000040: "M1_NEEDPICK", 0x00000080: "M1_CONCEAL",
    0x00000100: "M1_HIDE", 0x00000200: "M1_AMPHIBIOUS",
    0x00000400: "M1_BREATHLESS", 0x00000800: "M1_NOTAKE",
    0x00001000: "M1_NOEYES", 0x00002000: "M1_NOHANDS",
    0x00006000: "M1_NOLIMBS", 0x00008000: "M1_NOHEAD",
    0x00010000: "M1_MINDLESS", 0x00020000: "M1_HUMANOID",
    0x00040000: "M1_ANIMAL", 0x00080000: "M1_SLITHY",
    0x00100000: "M1_UNSOLID", 0x00200000: "M1_THICK_HIDE",
    0x00400000: "M1_OVIPAROUS", 0x00800000: "M1_REGEN",
    0x01000000: "M1_SEE_INVIS", 0x02000000: "M1_TPORT",
    0x04000000: "M1_TPORT_CNTRL", 0x08000000: "M1_ACID",
    0x10000000: "M1_POIS", 0x20000000: "M1_CARNIVORE",
    0x40000000: "M1_HERBIVORE", 0x80000000: "M1_METALLIVORE",
}

M2_FLAGS = {
    0x00000001: "M2_NOPOLY", 0x00000002: "M2_UNDEAD",
    0x00000004: "M2_WERE", 0x00000008: "M2_HUMAN",
    0x00000010: "M2_ELF", 0x00000020: "M2_DWARF",
    0x00000040: "M2_GNOME", 0x00000080: "M2_ORC",
    0x00000100: "M2_DEMON", 0x00000200: "M2_MERC",
    0x00000400: "M2_LORD", 0x00000800: "M2_PRINCE",
    0x00001000: "M2_MINION", 0x00002000: "M2_GIANT",
    0x00004000: "M2_SHAPESHIFTER", 0x00010000: "M2_MALE",
    0x00020000: "M2_FEMALE", 0x00040000: "M2_NEUTER",
    0x00080000: "M2_PNAME", 0x00100000: "M2_HOSTILE",
    0x00200000: "M2_PEACEFUL", 0x00400000: "M2_DOMESTIC",
    0x00800000: "M2_WANDER", 0x01000000: "M2_STALK",
    0x02000000: "M2_NASTY", 0x04000000: "M2_STRONG",
    0x08000000: "M2_ROCKTHROW", 0x10000000: "M2_GREEDY",
    0x20000000: "M2_JEWELS", 0x40000000: "M2_COLLECT",
    0x80000000: "M2_MAGIC",
}

M3_FLAGS = {
    0x0001: "M3_WANTSAMUL", 0x0002: "M3_WANTSBELL",
    0x0004: "M3_WANTSBOOK", 0x0008: "M3_WANTSCAND",
    0x0010: "M3_WANTSARTI", 0x0040: "M3_WAITFORU",
    0x0080: "M3_CLOSE", 0x0100: "M3_INFRAVISION",
    0x0200: "M3_INFRAVISIBLE", 0x0400: "M3_DISPLACES",
}

G_FLAGS = {
    0x1000: "G_UNIQ", 0x0800: "G_NOHELL", 0x0400: "G_HELL",
    0x0200: "G_NOGEN", 0x0080: "G_SGROUP", 0x0040: "G_LGROUP",
    0x0020: "G_GENO", 0x0010: "G_NOCORPSE",
}


def decode_flags(value, flag_table):
    """Return list of flag names that are set in value."""
    flags = []
    for mask, name in sorted(flag_table.items()):
        if value & mask == mask and mask != 0:
            flags.append(name)
    return flags


def decode_m1_flags(value):
    """Decode M1 flags, handling composite flags correctly.

    NOLIMBS (0x6000) is a superset of NOHANDS (0x2000).
    OMNIVORE (0x60000000) is CARNIVORE | HERBIVORE.
    """
    flags = []
    for mask in sorted(M1_FLAGS.keys()):
        name = M1_FLAGS[mask]
        if mask == 0x00002000:  # M1_NOHANDS
            if value & 0x6000 == 0x2000:
                flags.append("M1_NOHANDS")
        elif mask == 0x00006000:  # M1_NOLIMBS
            if value & 0x6000 == 0x6000:
                flags.append("M1_NOLIMBS")
        elif mask == 0x20000000:  # M1_CARNIVORE
            if value & 0x60000000 == 0x60000000:
                flags.append("M1_OMNIVORE")
            elif value & mask:
                flags.append("M1_CARNIVORE")
        elif mask == 0x40000000:  # M1_HERBIVORE
            # Already handled above as OMNIVORE if both set
            if value & 0x60000000 == 0x40000000:
                flags.append("M1_HERBIVORE")
        else:
            if value & mask:
                flags.append(name)
    return flags


# ---------------------------------------------------------------------------
# C expression evaluator for constant expressions in the MON() macros
# ---------------------------------------------------------------------------

# All #define constants that appear in monst.c field expressions.
CONST_TABLE = {}
# Attack types (name -> int)
for k, v in list(ATTACK_TYPES.items()):
    CONST_TABLE[v] = k
# Damage types (name -> int)
for k, v in list(DAMAGE_TYPES.items()):
    CONST_TABLE[v] = k
# Sounds (name -> int)
for k, v in list(SOUNDS.items()):
    CONST_TABLE[v] = k
# Sizes (name -> int)
for k, v in list(SIZES.items()):
    CONST_TABLE[v] = k
# Symbol classes
CONST_TABLE.update(SYMBOL_CLASSES)
# Resistance flags
for k, v in list(MR_FLAGS.items()):
    CONST_TABLE[v] = k
# M1 M2 M3 flags
for k, v in list(M1_FLAGS.items()):
    CONST_TABLE[v] = k
for k, v in list(M2_FLAGS.items()):
    CONST_TABLE[v] = k
for k, v in list(M3_FLAGS.items()):
    CONST_TABLE[v] = k
# Geno flags
for k, v in list(G_FLAGS.items()):
    CONST_TABLE[v] = k
# Colors
for k, v in list(COLORS.items()):
    CONST_TABLE[v] = k
# Composite M1 flags used in source
CONST_TABLE["M1_OMNIVORE"] = 0x60000000  # M1_CARNIVORE | M1_HERBIVORE
# Composite M3 flags
CONST_TABLE["M3_COVETOUS"] = 0x001f
CONST_TABLE["M3_WANTSALL"] = 0x001f
CONST_TABLE["M3_WAITMASK"] = 0x00c0
# Sound aliases
CONST_TABLE["MS_ORC"] = 9      # MS_GRUNT
CONST_TABLE["MS_ANIMAL"] = 13  # MS_BURBLE
CONST_TABLE["MS_FERRY"] = 40   # Charon only, never defined in vanilla
# Alignment
CONST_TABLE["A_NONE"] = -128
CONST_TABLE["A_NEUTRAL"] = 0
CONST_TABLE["A_CHAOTIC"] = -1
CONST_TABLE["A_LAWFUL"] = 1
# Color aliases from color.h and monst.c
CONST_TABLE["HI_DOMESTIC"] = 15   # CLR_WHITE
CONST_TABLE["HI_LORD"] = 5        # CLR_MAGENTA
CONST_TABLE["HI_ZAP"] = 12        # CLR_BRIGHT_BLUE
CONST_TABLE["HI_METAL"] = 6       # CLR_CYAN
CONST_TABLE["HI_GOLD"] = 11       # CLR_YELLOW
CONST_TABLE["HI_LEATHER"] = 3     # CLR_BROWN
CONST_TABLE["HI_PAPER"] = 15      # CLR_WHITE
CONST_TABLE["HI_WOOD"] = 3        # CLR_BROWN
CONST_TABLE["HI_ORGANIC"] = 3     # CLR_BROWN
CONST_TABLE["DRAGON_SILVER"] = 14  # CLR_BRIGHT_CYAN
# Weight macros
CONST_TABLE["WT_HUMAN"] = 1450
CONST_TABLE["WT_ELF"] = 800
CONST_TABLE["WT_DRAGON"] = 4500
# Size aliases
CONST_TABLE["MZ_HUMAN"] = 2  # MZ_MEDIUM
# Extra symbol classes used in source
CONST_TABLE["S_WORM_TAIL"] = 59


def eval_c_expr(expr):
    """Evaluate a C constant expression (ints, |, +, -, ~, parens, #define names)."""
    expr = expr.strip()
    if not expr:
        return 0

    # Replace all known constants with their integer values.
    # Sort by length descending so longer names match first.
    tokens = sorted(CONST_TABLE.keys(), key=len, reverse=True)
    result = expr
    for tok in tokens:
        result = re.sub(r'\b' + re.escape(tok) + r'\b', str(CONST_TABLE[tok]), result)

    # Strip C long/unsigned suffixes
    result = re.sub(r'(\d)(?:UL|ULL|L|LL|U)\b', r'\1', result, flags=re.IGNORECASE)

    # Now evaluate. Should only contain digits, |, +, -, ~, parens, spaces.
    try:
        val = eval(result, {"__builtins__": {}}, {})
        return int(val)
    except Exception:
        print(f"WARNING: could not evaluate: {expr!r} -> {result!r}", file=sys.stderr)
        return 0


# ---------------------------------------------------------------------------
# Preprocessor: strip dead code (#if 0), keep conditional code
# ---------------------------------------------------------------------------

def strip_dead_code(text):
    """Remove #if 0 ... #endif blocks, keep all other conditional code.

    Also removes #define lines, #include lines, and other preprocessor
    directives that aren't relevant to data extraction. Keeps content
    inside #ifdef CHARON, #ifdef MAIL, SPLITMON blocks, etc.
    """
    lines = text.split('\n')
    output = []
    # Stack tracks nesting of #if/#ifdef/#ifndef. Each entry is True if
    # we're in a dead (#if 0) block at that level.
    dead_stack = []

    for line in lines:
        stripped = line.strip()

        # Track #if 0 blocks
        if re.match(r'#\s*if\s+0\b', stripped):
            dead_stack.append(True)
            continue
        elif re.match(r'#\s*(?:ifdef|ifndef|if\b)', stripped):
            dead_stack.append(False)
            continue
        elif re.match(r'#\s*else\b', stripped):
            if dead_stack:
                # Invert: if we were dead, we're now live and vice versa
                dead_stack[-1] = not dead_stack[-1]
            continue
        elif re.match(r'#\s*endif\b', stripped):
            if dead_stack:
                dead_stack.pop()
            continue

        # If inside any dead block, skip this line
        if any(dead_stack):
            continue

        # Skip other preprocessor directives (defines, includes, etc.)
        if re.match(r'#\s*(define|include|undef)\b', stripped):
            continue

        output.append(line)

    return '\n'.join(output)


# ---------------------------------------------------------------------------
# Parser: extract MON() calls from monst.c
# ---------------------------------------------------------------------------

def preprocess(text):
    """Strip comments, join continuation lines, collapse whitespace."""
    # Remove block comments
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
    # Remove line comments
    text = re.sub(r'//.*', '', text)
    # Join backslash-continued lines
    text = re.sub(r'\\\n', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def find_matching_paren(text, start):
    """Find the index of the closing paren matching the opening paren at `start`."""
    depth = 0
    i = start
    in_string = False
    escape_next = False
    while i < len(text):
        ch = text[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if ch == '\\' and in_string:
            escape_next = True
            i += 1
            continue
        if ch == '"' and not in_string:
            in_string = True
            i += 1
            continue
        if ch == '"' and in_string:
            in_string = False
            i += 1
            continue
        if in_string:
            i += 1
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level_args(text):
    """Split text on commas at top-level (depth 0) of parens/braces."""
    args = []
    depth = 0
    current = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            current.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            current.append(ch)
            escape_next = True
            continue
        if ch == '"' and not in_string:
            in_string = True
            current.append(ch)
            continue
        if ch == '"' and in_string:
            in_string = False
            current.append(ch)
            continue
        if in_string:
            current.append(ch)
            continue
        if ch in '({':
            depth += 1
            current.append(ch)
        elif ch in ')}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args


def extract_mon_calls(text):
    """Find all MON(...) calls in the text, return their inner argument strings."""
    calls = []
    idx = 0
    while True:
        pos = text.find('MON(', idx)
        if pos == -1:
            break
        # Skip if preceded by an alphanumeric (part of another identifier)
        if pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] == '_'):
            idx = pos + 1
            continue
        paren_start = pos + 3  # index of '('
        paren_end = find_matching_paren(text, paren_start)
        if paren_end == -1:
            idx = pos + 1
            continue
        inner = text[paren_start + 1:paren_end]
        calls.append(inner)
        idx = paren_end + 1
    return calls


def parse_attack_tuple(expr):
    """Parse an ATTK(at, ad, n, d) or NO_ATTK or {0,0,0,0}-style expression."""
    expr = expr.strip()
    if expr in ("NO_ATTK", "{ 0, 0, 0, 0 }", "{0, 0, 0, 0}", "{0,0,0,0}"):
        return {"attack_type": "AT_NONE", "damage_type": "AD_PHYS",
                "dice": 0, "sides": 0}

    m = re.match(r'(?:ATTK\s*\(|{\s*)(.*?)(?:\)|})$', expr, re.DOTALL)
    if m:
        parts = [p.strip() for p in m.group(1).split(',')]
    else:
        parts = [p.strip() for p in expr.split(',')]

    if len(parts) != 4:
        print(f"WARNING: unexpected attack format: {expr!r}", file=sys.stderr)
        return {"attack_type": "AT_NONE", "damage_type": "AD_PHYS",
                "dice": 0, "sides": 0}

    at_val = eval_c_expr(parts[0])
    ad_val = eval_c_expr(parts[1])
    return {
        "attack_type": ATTACK_TYPES.get(at_val, f"UNKNOWN({at_val})"),
        "damage_type": DAMAGE_TYPES.get(ad_val, f"UNKNOWN({ad_val})"),
        "dice": eval_c_expr(parts[2]),
        "sides": eval_c_expr(parts[3]),
    }


# Hardcoded macro expansions for attack macros defined inside monst.c
ATTACK_MACROS = {
    "SEDUCTION_ATTACKS_YES": (
        "A(ATTK(AT_BITE, AD_SSEX, 0, 0), ATTK(AT_CLAW, AD_PHYS, 1, 3),"
        " ATTK(AT_CLAW, AD_PHYS, 1, 3), NO_ATTK, NO_ATTK, NO_ATTK)"
    ),
    "SEDUCTION_ATTACKS_NO": (
        "A(ATTK(AT_CLAW, AD_PHYS, 1, 3), ATTK(AT_CLAW, AD_PHYS, 1, 3),"
        " ATTK(AT_BITE, AD_DRLI, 2, 6), NO_ATTK, NO_ATTK, NO_ATTK)"
    ),
}


def parse_attacks(expr):
    """Parse A(atk1, atk2, ..., atk6) expression."""
    expr = expr.strip()

    # Expand known attack macros
    if expr in ATTACK_MACROS:
        expr = ATTACK_MACROS[expr]

    m = re.match(r'A\s*\((.*)\)$', expr, re.DOTALL)
    if not m:
        m = re.match(r'\{(.*)\}$', expr, re.DOTALL)
    if not m:
        print(f"WARNING: unexpected attack array format: {expr!r}", file=sys.stderr)
        return []

    inner = m.group(1).strip()
    parts = split_top_level_args(inner)
    return [parse_attack_tuple(p) for p in parts]


def parse_mon(call_text):
    """Parse a single MON() call's argument text into a monster dict."""
    # MON(nam, sym, LVL(...), gen, A(...), SIZ(...), mr1, mr2, flg1, flg2, flg3, d, col)
    args = split_top_level_args(call_text)

    if len(args) != 13:
        print(f"WARNING: expected 13 args, got {len(args)}: {call_text[:80]!r}",
              file=sys.stderr)
        return None

    name = args[0].strip().strip('"')
    if name == "":
        return None

    sym_val = eval_c_expr(args[1])

    # LVL(lvl, mov, ac, mr, aln)
    lvl_m = re.match(r'LVL\s*\((.*)\)$', args[2].strip(), re.DOTALL)
    if lvl_m:
        lvl_parts = [p.strip() for p in lvl_m.group(1).split(',')]
    else:
        print(f"WARNING: couldn't parse LVL: {args[2]!r}", file=sys.stderr)
        lvl_parts = ["0", "0", "0", "0", "0"]

    level = eval_c_expr(lvl_parts[0])
    speed = eval_c_expr(lvl_parts[1])
    ac = eval_c_expr(lvl_parts[2])
    magic_resistance = eval_c_expr(lvl_parts[3])
    alignment = eval_c_expr(lvl_parts[4])

    geno = eval_c_expr(args[3])
    frequency = geno & 0x0007
    geno_flags = decode_flags(geno, G_FLAGS)

    attacks = parse_attacks(args[4])

    # SIZ(wt, nut, snd, siz)
    siz_m = re.match(r'SIZ\s*\((.*)\)$', args[5].strip(), re.DOTALL)
    if siz_m:
        siz_parts = [p.strip() for p in siz_m.group(1).split(',')]
    else:
        print(f"WARNING: couldn't parse SIZ: {args[5]!r}", file=sys.stderr)
        siz_parts = ["0", "0", "0", "0"]

    weight = eval_c_expr(siz_parts[0])
    nutrition = eval_c_expr(siz_parts[1])
    sound_val = eval_c_expr(siz_parts[2])
    size_val = eval_c_expr(siz_parts[3])

    mr1 = eval_c_expr(args[6])
    mr2 = eval_c_expr(args[7])
    flg1 = eval_c_expr(args[8])
    flg2 = eval_c_expr(args[9])
    flg3 = eval_c_expr(args[10])
    difficulty = eval_c_expr(args[11])
    color_val = eval_c_expr(args[12])

    return {
        "name": name,
        "symbol_class": SYMBOL_CLASS_NAMES.get(sym_val, f"S_UNKNOWN({sym_val})"),
        "symbol_class_id": sym_val,
        "symbol": SYMBOL_CHARS.get(sym_val, "?"),
        "level": level,
        "speed": speed,
        "armor_class": ac,
        "magic_resistance": magic_resistance,
        "alignment": alignment,
        "generation_flags": geno_flags,
        "generation_flags_raw": geno & ~0x0007,
        "generation_frequency": frequency,
        "attacks": attacks,
        "weight": weight,
        "nutrition": nutrition,
        "sound": SOUNDS.get(sound_val, f"MS_UNKNOWN({sound_val})"),
        "sound_id": sound_val,
        "size": SIZES.get(size_val, f"MZ_UNKNOWN({size_val})"),
        "size_id": size_val,
        "resistances_raw": mr1,
        "resistances": decode_flags(mr1, MR_FLAGS),
        "resistances_conveyed_raw": mr2,
        "resistances_conveyed": decode_flags(mr2, MR_FLAGS),
        "flags1_raw": flg1,
        "flags1": decode_m1_flags(flg1),
        "flags2_raw": flg2,
        "flags2": decode_flags(flg2, M2_FLAGS),
        "flags3_raw": flg3,
        "flags3": decode_flags(flg3, M3_FLAGS),
        "difficulty": difficulty,
        "color": COLORS.get(color_val, f"CLR_UNKNOWN({color_val})"),
        "color_id": color_val,
    }


def main():
    raw = MONST_C.read_text()

    # Phase 1: strip #if 0 dead code, keep all other conditionally compiled code
    text = strip_dead_code(raw)

    # Phase 2: strip C comments, collapse whitespace
    text = preprocess(text)

    # Phase 3: find all MON() calls anywhere in the processed text.
    # This handles the SPLITMON array split naturally.
    mon_calls = extract_mon_calls(text)

    monsters = []
    for call in mon_calls:
        m = parse_mon(call)
        if m is not None:
            monsters.append(m)

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(monsters, f, indent=2)

    # Summary
    print(f"Total monsters parsed: {len(monsters)}")
    print()
    class_counts = Counter(m["symbol_class"] for m in monsters)
    print("Breakdown by symbol class:")
    for cls, count in sorted(class_counts.items()):
        sym = SYMBOL_CHARS.get(SYMBOL_CLASSES.get(cls, -1), "?")
        print(f"  {cls:20s} ({sym}): {count}")

    # Print 3 sample monsters
    print()
    print("=" * 72)
    print("Sample monsters:")
    print("=" * 72)
    samples = [
        next((m for m in monsters if m["name"] == "giant ant"), None),
        next((m for m in monsters if m["name"] == "mind flayer"), None),
        next((m for m in monsters if m["name"] == "Demogorgon"), None),
    ]
    for s in samples:
        if s:
            print(json.dumps(s, indent=2))
            print()

    print(f"Output written to: {OUTPUT}")


if __name__ == "__main__":
    main()
