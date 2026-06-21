"""Extract a rich, oracle-ready game state from a live AutoAscend agent.

Both the descent and threat oracles were bottlenecked on thin state (XL/HP/depth
only). The LLM can only reason about what it's told -- gear, AC, hunger, and the
monsters actually on screen. This pulls all of it from blstats + inventory +
visible monsters into a compact dict.
"""

_HUNGER = {0: "satiated", 1: "normal", 2: "hungry", 3: "weak", 4: "fainting"}
_DUNGEON = {0: "DoD", 2: "Gnomish Mines", 3: "Quest", 4: "Sokoban", 5: "Mines End"}


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def extract(agent, threat_monster=None):
    bl = agent.blstats
    hp = int(bl.hitpoints); mhp = max(1, int(bl.max_hitpoints))
    st = {
        "role": _safe(lambda: str(agent.character).split()[0], "?"),
        "xl": int(bl.experience_level),
        "hp": hp, "max_hp": mhp, "hp_pct": round(100 * hp / mhp),
        "ac": int(bl.armor_class),                # lower is better
        "depth": int(bl.depth),
        "dungeon": _DUNGEON.get(int(bl.dungeon_number), str(int(bl.dungeon_number))),
        "hunger": _HUNGER.get(int(bl.hunger_state), str(int(bl.hunger_state))),
        "str": int(bl.strength), "gold": int(bl.gold),
    }
    # gear
    st["weapon"] = _safe(lambda: _wielded_weapon(agent), "none")
    st["has_ranged"] = _safe(lambda: _has_ranged(agent), False)
    st["has_gloves"] = _safe(lambda: _has_gloves(agent), False)
    # visible threats
    st["monsters_near"] = _safe(lambda: _nearby_monsters(agent), [])
    if threat_monster is not None:
        try:
            _, y, x, mon, _ = threat_monster
            st["threat_name"] = mon.mname
            st["threat_dist"] = max(abs(y - bl.y), abs(x - bl.x))
        except Exception:
            pass
    return st


def _items(agent):
    return list(agent.inventory.items)


def _wielded_weapon(agent):
    for it in _items(agent):
        if getattr(it, "equipped", False) and it.is_weapon():
            return it.text
    return "none"


def _has_ranged(agent):
    for it in _items(agent):
        t = (getattr(it, "text", "") or "").lower()
        if any(w in t for w in ("bow", "sling", "crossbow", "dart", "arrow", "dagger")):
            return True
    return False


def _has_gloves(agent):
    for it in _items(agent):
        t = (getattr(it, "text", "") or "").lower()
        if "gloves" in t or "gauntlets" in t:
            if getattr(it, "equipped", False):
                return True
    return False


def _nearby_monsters(agent, radius=8, limit=6):
    bl = agent.blstats
    out = []
    for m in agent.get_visible_monsters():
        try:
            _, y, x, mon, _ = m
            d = max(abs(y - bl.y), abs(x - bl.x))
            if d <= radius:
                out.append({"name": mon.mname, "dist": int(d)})
        except Exception:
            continue
    out.sort(key=lambda r: r["dist"])
    return out[:limit]
