#!/usr/bin/env python3
"""Parse NetHack 3.6.7 objects.c into structured JSON.

Reads macro invocations (WEAPON, ARMOR, SCROLL, POTION, RING, WAND, AMULET,
GEM, TOOL, FOOD, SPELL, etc.) and extracts all fields into a flat dict per
item. Outputs data/parsed/items.json grouped by object class, and prints
price tables to stdout for verification.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

SRC = Path(__file__).resolve().parent.parent / "data/source/NetHack/src/objects.c"
OUT = Path(__file__).resolve().parent.parent / "data/parsed/items.json"

MATERIALS = {
    0: "none", 1: "liquid", 2: "wax", 3: "veggy", 4: "flesh", 5: "paper",
    6: "cloth", 7: "leather", 8: "wood", 9: "bone", 10: "dragon_hide",
    11: "iron", 12: "metal", 13: "copper", 14: "silver", 15: "gold",
    16: "platinum", 17: "mithril", 18: "plastic", 19: "glass",
    20: "gemstone", 21: "mineral",
}

MATERIAL_NAMES = {
    "LIQUID": 1, "WAX": 2, "VEGGY": 3, "FLESH": 4, "PAPER": 5,
    "CLOTH": 6, "LEATHER": 7, "WOOD": 8, "BONE": 9, "DRAGON_HIDE": 10,
    "IRON": 11, "METAL": 12, "COPPER": 13, "SILVER": 14, "GOLD": 15,
    "PLATINUM": 16, "MITHRIL": 17, "PLASTIC": 18, "GLASS": 19,
    "GEMSTONE": 20, "MINERAL": 21,
}

CLASS_NAMES = {
    "ILLOBJ_CLASS": "illobj", "WEAPON_CLASS": "weapon", "ARMOR_CLASS": "armor",
    "RING_CLASS": "ring", "AMULET_CLASS": "amulet", "TOOL_CLASS": "tool",
    "FOOD_CLASS": "food", "POTION_CLASS": "potion", "SCROLL_CLASS": "scroll",
    "SPBOOK_CLASS": "spellbook", "WAND_CLASS": "wand", "COIN_CLASS": "coin",
    "GEM_CLASS": "gem", "ROCK_CLASS": "rock", "BALL_CLASS": "ball",
    "CHAIN_CLASS": "chain", "VENOM_CLASS": "venom",
}


def read_source():
    return SRC.read_text()


def strip_comments(text):
    """Remove C block and line comments."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)
    return text


def strip_defines(text):
    """Remove #define macro definitions (including continuation lines)."""
    lines = text.split('\n')
    result = []
    in_define = False
    for line in lines:
        stripped = line.rstrip()
        if stripped.lstrip().startswith('#define'):
            in_define = True
        if in_define:
            if not stripped.endswith('\\'):
                in_define = False
            continue
        result.append(line)
    return '\n'.join(result)


def extract_macro_calls(text, macro_name):
    """Find all top-level macro invocations and return their raw arg strings.

    Handles nested parentheses so we capture the full argument list.
    """
    results = []
    pattern = re.compile(r'\b' + re.escape(macro_name) + r'\s*\(')
    for m in pattern.finditer(text):
        start = m.end() - 1  # position of opening '('
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    inner = text[start + 1:i]
                    results.append(inner)
                    break
            i += 1
    return results


def split_args(s):
    """Split a C macro argument string on commas, respecting parens and strings."""
    args = []
    depth = 0
    current = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == '\\' and in_string:
            current.append(ch)
            escape = True
            continue
        if ch == '"' and depth == 0:
            in_string = not in_string
            current.append(ch)
            continue
        if in_string:
            current.append(ch)
            continue
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
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


def parse_string(s):
    """Parse a C string literal or None/NULL."""
    s = s.strip()
    if s in ('None', '(char *) 0', 'NULL', '0'):
        return None
    m = re.match(r'^"(.*)"$', s)
    if m:
        return m.group(1)
    return s


def parse_int(s):
    """Parse an integer, handling simple C expressions."""
    s = s.strip()
    # Handle bitwise OR for damage types like P|S
    if '|' in s and not s.startswith('"'):
        return s  # keep as string for damage type combos
    # Handle negation
    if s.startswith('-'):
        inner = parse_int(s[1:])
        if isinstance(inner, int):
            return -inner
        return s
    # Handle some known constants
    try:
        return int(s)
    except ValueError:
        return s


def resolve_material(s):
    """Convert material name to string."""
    s = s.strip()
    if s in MATERIAL_NAMES:
        return MATERIALS[MATERIAL_NAMES[s]]
    try:
        return MATERIALS[int(s)]
    except (ValueError, KeyError):
        return s.lower()


def parse_weapons(text):
    items = []

    # PROJECTILE(name,desc,kn,prob,wt,cost,sdam,ldam,hitbon,metal,sub,color)
    for raw in extract_macro_calls(text, 'PROJECTILE'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "weapon",
            "subclass": "projectile",
            "name_known": bool(parse_int(a[2])),
            "prob": parse_int(a[3]),
            "weight": parse_int(a[4]),
            "cost": parse_int(a[5]),
            "sdam": parse_int(a[6]),
            "ldam": parse_int(a[7]),
            "hitbon": parse_int(a[8]),
            "material": resolve_material(a[9]),
            "skill": a[10].strip(),
            "nutrition": parse_int(a[4]),  # nutrition = weight for weapons
        })

    # BOW(name,desc,kn,prob,wt,cost,hitbon,metal,sub,color)
    for raw in extract_macro_calls(text, 'BOW'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "weapon",
            "subclass": "bow",
            "name_known": bool(parse_int(a[2])),
            "prob": parse_int(a[3]),
            "weight": parse_int(a[4]),
            "cost": parse_int(a[5]),
            "sdam": 2,
            "ldam": 2,
            "hitbon": parse_int(a[6]),
            "material": resolve_material(a[7]),
            "skill": a[8].strip(),
            "nutrition": parse_int(a[4]),
        })

    # WEAPON(name,desc,kn,mg,bi,prob,wt,cost,sdam,ldam,hitbon,typ,sub,metal,color)
    for raw in extract_macro_calls(text, 'WEAPON'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "weapon",
            "subclass": "weapon",
            "name_known": bool(parse_int(a[2])),
            "merge": bool(parse_int(a[3])),
            "bimanual": bool(parse_int(a[4])),
            "prob": parse_int(a[5]),
            "weight": parse_int(a[6]),
            "cost": parse_int(a[7]),
            "sdam": parse_int(a[8]),
            "ldam": parse_int(a[9]),
            "hitbon": parse_int(a[10]),
            "damage_type": a[11].strip(),
            "skill": a[12].strip(),
            "material": resolve_material(a[13]),
            "nutrition": parse_int(a[6]),
        })

    return items


def parse_armor(text):
    items = []

    # ARMOR(name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,sub,metal,c)
    for raw in extract_macro_calls(text, 'ARMOR'):
        a = split_args(raw)
        sub_val = a[12].strip()
        subclass_map = {
            "ARM_SUIT": "suit", "ARM_SHIELD": "shield", "ARM_HELM": "helm",
            "ARM_GLOVES": "gloves", "ARM_BOOTS": "boots", "ARM_CLOAK": "cloak",
            "ARM_SHIRT": "shirt",
        }
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "armor",
            "subclass": subclass_map.get(sub_val, sub_val),
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bulky": bool(parse_int(a[4])),
            "prob": parse_int(a[6]),
            "delay": parse_int(a[7]),
            "weight": parse_int(a[8]),
            "cost": parse_int(a[9]),
            "ac": parse_int(a[10]),
            "mc": parse_int(a[11]),
            "material": resolve_material(a[13]),
        })

    # HELM, CLOAK, SHIELD, GLOVES, BOOTS all expand to ARMOR, but we
    # also parse them directly since they appear before ARMOR #undef.
    # Actually they use the ARMOR macro internally. The extract_macro_calls
    # for ARMOR will catch the inner ARMOR() call in HELM() etc.
    # Let me check...

    # No. HELM etc. are #define macros that expand to ARMOR(). In the source,
    # HELM(...) appears, not ARMOR(...). extract_macro_calls looks at the
    # literal text, so we need to parse HELM, CLOAK, SHIELD, GLOVES, BOOTS
    # separately.

    # HELM(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)
    for raw in extract_macro_calls(text, 'HELM'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "armor",
            "subclass": "helm",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bulky": False,
            "prob": parse_int(a[5]),
            "delay": parse_int(a[6]),
            "weight": parse_int(a[7]),
            "cost": parse_int(a[8]),
            "ac": parse_int(a[9]),
            "mc": parse_int(a[10]),
            "material": resolve_material(a[11]),
        })

    # CLOAK(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)
    for raw in extract_macro_calls(text, 'CLOAK'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "armor",
            "subclass": "cloak",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bulky": False,
            "prob": parse_int(a[5]),
            "delay": parse_int(a[6]),
            "weight": parse_int(a[7]),
            "cost": parse_int(a[8]),
            "ac": parse_int(a[9]),
            "mc": parse_int(a[10]),
            "material": resolve_material(a[11]),
        })

    # SHIELD(name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,metal,c)
    for raw in extract_macro_calls(text, 'SHIELD'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "armor",
            "subclass": "shield",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bulky": bool(parse_int(a[4])),
            "prob": parse_int(a[6]),
            "delay": parse_int(a[7]),
            "weight": parse_int(a[8]),
            "cost": parse_int(a[9]),
            "ac": parse_int(a[10]),
            "mc": parse_int(a[11]),
            "material": resolve_material(a[12]),
        })

    # GLOVES(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)
    for raw in extract_macro_calls(text, 'GLOVES'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "armor",
            "subclass": "gloves",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bulky": False,
            "prob": parse_int(a[5]),
            "delay": parse_int(a[6]),
            "weight": parse_int(a[7]),
            "cost": parse_int(a[8]),
            "ac": parse_int(a[9]),
            "mc": parse_int(a[10]),
            "material": resolve_material(a[11]),
        })

    # BOOTS(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)
    for raw in extract_macro_calls(text, 'BOOTS'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "armor",
            "subclass": "boots",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bulky": False,
            "prob": parse_int(a[5]),
            "delay": parse_int(a[6]),
            "weight": parse_int(a[7]),
            "cost": parse_int(a[8]),
            "ac": parse_int(a[9]),
            "mc": parse_int(a[10]),
            "material": resolve_material(a[11]),
        })

    # DRGN_ARMR(name,mgc,power,cost,ac,color)
    for raw in extract_macro_calls(text, 'DRGN_ARMR'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": None,
            "class": "armor",
            "subclass": "suit",
            "name_known": True,
            "magic": bool(parse_int(a[1])),
            "bulky": True,
            "prob": 0,
            "delay": 5,
            "weight": 40,
            "cost": parse_int(a[3]),
            "ac": parse_int(a[4]),
            "mc": 0,
            "material": "dragon_hide",
        })

    return items


def parse_rings(text):
    items = []
    # RING(name,stone,power,cost,mgc,spec,mohs,metal,color)
    for raw in extract_macro_calls(text, 'RING'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "ring",
            "cost": parse_int(a[3]),
            "magic": bool(parse_int(a[4])),
            "material": resolve_material(a[7]),
            "weight": 3,
        })
    return items


def parse_amulets(text):
    items = []
    # AMULET(name,desc,power,prob)
    # All amulets have cost=150, weight=20, material=iron
    for raw in extract_macro_calls(text, 'AMULET'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "amulet",
            "prob": parse_int(a[3]),
            "cost": 150,
            "weight": 20,
            "material": "iron",
        })
    return items


def parse_tools(text):
    items = []

    # CONTAINER(name,desc,kn,mgc,chg,prob,wt,cost,mat,color)
    for raw in extract_macro_calls(text, 'CONTAINER'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "tool",
            "subclass": "container",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "prob": parse_int(a[5]),
            "weight": parse_int(a[6]),
            "cost": parse_int(a[7]),
            "material": resolve_material(a[8]),
        })

    # TOOL(name,desc,kn,mrg,mgc,chg,prob,wt,cost,mat,color)
    for raw in extract_macro_calls(text, 'TOOL'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "tool",
            "subclass": "tool",
            "name_known": bool(parse_int(a[2])),
            "merge": bool(parse_int(a[3])),
            "magic": bool(parse_int(a[4])),
            "prob": parse_int(a[6]),
            "weight": parse_int(a[7]),
            "cost": parse_int(a[8]),
            "material": resolve_material(a[9]),
        })

    # WEPTOOL(name,desc,kn,mgc,bi,prob,wt,cost,sdam,ldam,hitbon,sub,mat,clr)
    for raw in extract_macro_calls(text, 'WEPTOOL'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "tool",
            "subclass": "weptool",
            "name_known": bool(parse_int(a[2])),
            "magic": bool(parse_int(a[3])),
            "bimanual": bool(parse_int(a[4])),
            "prob": parse_int(a[5]),
            "weight": parse_int(a[6]),
            "cost": parse_int(a[7]),
            "sdam": parse_int(a[8]),
            "ldam": parse_int(a[9]),
            "hitbon": parse_int(a[10]),
            "skill": a[11].strip(),
            "material": resolve_material(a[12]),
        })

    return items


def parse_food(text):
    items = []
    # FOOD(name, prob, delay, wt, unk, tin, nutrition, color)
    # cost = nutrition / 20 + 5
    for raw in extract_macro_calls(text, 'FOOD'):
        a = split_args(raw)
        nutrition = parse_int(a[6])
        cost = nutrition // 20 + 5 if isinstance(nutrition, int) else nutrition
        items.append({
            "name": parse_string(a[0]),
            "description": None,
            "class": "food",
            "prob": parse_int(a[1]),
            "delay": parse_int(a[2]),
            "weight": parse_int(a[3]),
            "cost": cost,
            "nutrition": nutrition,
            "material": resolve_material(a[5]),
        })
    return items


def parse_potions(text):
    items = []
    # POTION(name,desc,mgc,power,prob,cost,color)
    for raw in extract_macro_calls(text, 'POTION'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "potion",
            "magic": bool(parse_int(a[2])),
            "prob": parse_int(a[4]),
            "cost": parse_int(a[5]),
            "weight": 20,
            "material": "glass",
        })
    return items


def parse_scrolls(text):
    items = []
    # SCROLL(name,text,mgc,prob,cost)
    for raw in extract_macro_calls(text, 'SCROLL'):
        a = split_args(raw)
        name = parse_string(a[0])
        items.append({
            "name": name,
            "description": parse_string(a[1]),
            "class": "scroll",
            "magic": bool(parse_int(a[2])),
            "prob": parse_int(a[3]),
            "cost": parse_int(a[4]),
            "weight": 5,
            "material": "paper",
        })
    return items


def parse_spellbooks(text):
    items = []
    # SPELL(name,desc,sub,prob,delay,level,mgc,dir,color)
    # cost = level * 100
    for raw in extract_macro_calls(text, 'SPELL'):
        a = split_args(raw)
        level = parse_int(a[5])
        cost = level * 100 if isinstance(level, int) else level
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "spellbook",
            "skill": a[2].strip(),
            "prob": parse_int(a[3]),
            "delay": parse_int(a[4]),
            "level": level,
            "cost": cost,
            "weight": 50,
            "material": "paper",
        })
    return items


def parse_wands(text):
    items = []
    # WAND(name,typ,prob,cost,mgc,dir,metal,color)
    for raw in extract_macro_calls(text, 'WAND'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "wand",
            "prob": parse_int(a[2]),
            "cost": parse_int(a[3]),
            "magic": bool(parse_int(a[4])),
            "material": resolve_material(a[6]),
            "weight": 7,
        })
    return items


def parse_gems(text):
    items = []
    # GEM(name,desc,prob,wt,gval,nutr,mohs,glass,color)
    for raw in extract_macro_calls(text, 'GEM'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "gem",
            "prob": parse_int(a[2]),
            "weight": parse_int(a[3]),
            "cost": parse_int(a[4]),
            "nutrition": parse_int(a[5]),
            "hardness": parse_int(a[6]),
            "material": resolve_material(a[7]),
        })

    # ROCK(name,desc,kn,prob,wt,gval,sdam,ldam,mgc,nutr,mohs,glass,color)
    for raw in extract_macro_calls(text, 'ROCK'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "description": parse_string(a[1]),
            "class": "gem",
            "subclass": "rock",
            "name_known": bool(parse_int(a[2])),
            "prob": parse_int(a[3]),
            "weight": parse_int(a[4]),
            "cost": parse_int(a[5]),
            "sdam": parse_int(a[6]),
            "ldam": parse_int(a[7]),
            "magic": bool(parse_int(a[8])),
            "nutrition": parse_int(a[9]),
            "hardness": parse_int(a[10]),
            "material": resolve_material(a[11]),
        })

    return items


def parse_coins(text):
    items = []
    # COIN(name,prob,metal,worth)
    for raw in extract_macro_calls(text, 'COIN'):
        a = split_args(raw)
        items.append({
            "name": parse_string(a[0]),
            "class": "coin",
            "prob": parse_int(a[1]),
            "material": resolve_material(a[2]),
            "cost": parse_int(a[3]),
        })
    return items


def parse_raw_objects(text):
    """Parse remaining OBJECT() calls not captured by specific macros.

    These are special items like Amulet of Yendor, Candelabrum, etc.
    """
    items = []

    # Find all OBJECT() calls that are at top level (not inside a #define)
    # We need to be selective: only those that appear as actual item entries,
    # not inside macro definitions.

    # Strategy: find OBJECT( calls that are NOT preceded by a backslash
    # (continuation) or #define on the same logical line.
    lines = text.split('\n')
    in_define = False
    in_deferred = False
    object_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Track #if 0 blocks
        if stripped.startswith('#if 0'):
            in_deferred = True
        if in_deferred:
            if stripped.startswith('#endif'):
                in_deferred = False
            continue

        if stripped.startswith('#define'):
            in_define = True
        if in_define:
            if not stripped.endswith('\\'):
                in_define = False
            continue
        if 'OBJECT(OBJ(' in stripped and not stripped.startswith('//'):
            object_lines.append(i)

    # Now extract full calls starting from those lines
    full_text_lines = text.split('\n')
    for start_line in object_lines:
        # Accumulate until parens balance
        acc = ''
        depth = 0
        started = False
        for li in range(start_line, min(start_line + 20, len(full_text_lines))):
            acc += full_text_lines[li] + ' '
            for ch in full_text_lines[li]:
                if ch == '(' and not started:
                    # Find the OBJECT( start
                    pass
                if ch == '(':
                    depth += 1
                    started = True
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and started:
                        break
            if depth == 0 and started:
                break

        # Now parse this OBJECT call
        # Find the OBJECT( prefix
        m = re.search(r'OBJECT\s*\(', acc)
        if not m:
            continue
        inner_start = m.end()
        # Find balanced end
        depth = 1
        pos = inner_start
        while pos < len(acc) and depth > 0:
            if acc[pos] == '(':
                depth += 1
            elif acc[pos] == ')':
                depth -= 1
            pos += 1
        inner = acc[inner_start:pos - 1]

        # The OBJECT macro signature:
        # OBJECT(obj,bits,prp,sym,prob,dly,wt,cost,sdam,ldam,oc1,oc2,nut,color)
        # where obj = OBJ(name, desc)
        # and bits = BITS(...) or literal numbers

        # Extract OBJ(name, desc)
        obj_m = re.search(r'OBJ\s*\(', inner)
        if not obj_m:
            continue
        obj_start = obj_m.end()
        d = 1
        p = obj_start
        while p < len(inner) and d > 0:
            if inner[p] == '(':
                d += 1
            elif inner[p] == ')':
                d -= 1
            p += 1
        obj_inner = inner[obj_start:p - 1]
        obj_args = split_args(obj_inner)
        name = parse_string(obj_args[0])
        desc = parse_string(obj_args[1]) if len(obj_args) > 1 else None

        # Now get the rest of the args after OBJ(...)
        # Skip past the OBJ(...) and find the next comma
        rest = inner[p:].strip()
        if rest.startswith(','):
            rest = rest[1:]

        # Split remaining args: bits,prp,sym,prob,dly,wt,cost,sdam,ldam,oc1,oc2,nut,color
        rest_args = split_args(rest)

        # bits is the first arg (could be BITS(...) or raw)
        # We need: sym(class), prob, dly, wt, cost, sdam, ldam, oc1, oc2, nut
        # After bits: prp(idx1), sym(idx2), prob(idx3), dly(idx4), wt(idx5),
        #             cost(idx6), sdam(idx7), ldam(idx8), oc1(idx9), oc2(idx10),
        #             nut(idx11), color(idx12)

        if len(rest_args) < 12:
            continue

        sym = rest_args[2].strip()
        obj_class = CLASS_NAMES.get(sym, sym.lower())
        cost = parse_int(rest_args[6])
        weight = parse_int(rest_args[5])
        prob = parse_int(rest_args[3])

        # Extract material from BITS if possible
        bits_raw = rest_args[0]
        material = "unknown"
        bits_m = re.search(r'BITS\s*\(', bits_raw)
        if bits_m:
            bs = bits_raw[bits_m.end():]
            # Find closing paren
            d2 = 1
            p2 = 0
            while p2 < len(bs) and d2 > 0:
                if bs[p2] == '(':
                    d2 += 1
                elif bs[p2] == ')':
                    d2 -= 1
                p2 += 1
            bits_inner = bs[:p2 - 1]
            bits_args = split_args(bits_inner)
            # BITS(nmkn,mrg,uskn,ctnr,mgc,chrg,uniq,nwsh,big,tuf,dir,sub,mtrl)
            if len(bits_args) >= 13:
                material = resolve_material(bits_args[12])
        else:
            # Raw comma-separated bits
            # The bits expand to: nmkn,mrg,uskn,0,mgc,chrg,uniq,nwsh,big,tuf,dir,mtrl,sub
            # which is 13 values
            # In this case rest_args[0] through rest_args[12] are the bits,
            # and the rest shifts. This is trickier. Let's just handle BITS() form.
            pass

        item = {
            "name": name,
            "description": desc,
            "class": obj_class,
            "prob": prob,
            "weight": weight,
            "cost": cost,
        }
        if material != "unknown":
            item["material"] = material

        items.append(item)

    return items


def print_price_table(items, obj_class, label):
    """Print price grouping for a class."""
    filtered = [i for i in items if i.get("class") == obj_class and i.get("name")]
    if not filtered:
        return

    by_price = defaultdict(list)
    for item in filtered:
        by_price[item["cost"]].append(item["name"])

    print(f"\n{'='*60}")
    print(f"  {label} - Price Table")
    print(f"{'='*60}")
    for price in sorted(by_price.keys()):
        names = sorted(by_price[price])
        print(f"  {price:>6} zm : {', '.join(names)}")


def main():
    text = read_source()

    # Strip #if 0 blocks
    text_clean = re.sub(r'#if\s+0\s*/\*\s*DEFERRED\s*\*/.*?#endif', '', text, flags=re.DOTALL)

    # Strip comments and macro definitions
    text_clean = strip_comments(text_clean)
    text_clean = strip_defines(text_clean)

    all_items = []
    all_items.extend(parse_weapons(text_clean))
    all_items.extend(parse_armor(text_clean))
    all_items.extend(parse_rings(text_clean))
    all_items.extend(parse_amulets(text_clean))
    all_items.extend(parse_tools(text_clean))
    all_items.extend(parse_food(text_clean))
    all_items.extend(parse_potions(text_clean))
    all_items.extend(parse_scrolls(text_clean))
    all_items.extend(parse_spellbooks(text_clean))
    all_items.extend(parse_wands(text_clean))
    all_items.extend(parse_gems(text_clean))
    all_items.extend(parse_coins(text_clean))
    all_items.extend(parse_raw_objects(text_clean))

    # Group by class
    grouped = defaultdict(list)
    for item in all_items:
        grouped[item["class"]].append(item)

    # Summary
    print(f"Total items parsed: {len(all_items)}")
    for cls in sorted(grouped.keys()):
        print(f"  {cls}: {len(grouped[cls])}")

    # Write JSON
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(dict(grouped), f, indent=2)
    print(f"\nWritten to {OUT}")

    # Print price tables for verification
    print_price_table(all_items, "scroll", "SCROLLS")
    print_price_table(all_items, "potion", "POTIONS")
    print_price_table(all_items, "ring", "RINGS")
    print_price_table(all_items, "wand", "WANDS")
    print_price_table(all_items, "amulet", "AMULETS")
    print_price_table(all_items, "spellbook", "SPELLBOOKS")


if __name__ == "__main__":
    main()
