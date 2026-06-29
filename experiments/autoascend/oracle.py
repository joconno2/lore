"""LORE oracle: queries an LLM for a tactical decision at a dangerous game state.

Model-agnostic (OpenAI-compatible API, served by vLLM). The intervention layer
calls query_threat() when AutoAscend faces a knowledge-dependent death risk
(cockatrice -> petrification, fast hard-hitter -> being overrun). The oracle
returns a constrained action the intervention layer can execute.

The constrained vocabulary keeps the oracle's output executable and the EC search
space small:
  FIGHT   - melee is safe, engage
  RANGED  - attack from range only, never melee (e.g. cockatrice, floating eye)
  AVOID   - do not approach; route around
  FLEE    - move away toward known safe tiles / upstairs
  ELBERETH- engrave Elbereth to scare it (works on most non-@, non-mindless)
  PRAY    - emergency prayer (low HP, prayer timeout ok)
"""
import json
import os
import urllib.request

ACTIONS = ["FIGHT", "RANGED", "AVOID", "FLEE", "ELBERETH", "PRAY", "EAT"]

SYSTEM = (
    "You are a NetHack 3.6 tactical advisor for an expert bot. Given a threat, "
    "choose the single safest correct action from this set: "
    + ", ".join(ACTIONS) + ". "
    "Hard rules you must never violate:\n"
    "- Cockatrice/chickatrice ('c'): NEVER melee barehanded and NEVER touch the "
    "corpse (instant petrification death). Use RANGED, or AVOID if no ranged.\n"
    "- Floating eye ('e'): NEVER melee (paralysis death). RANGED or AVOID.\n"
    "- Fast hard-hitters that outpace you (unicorns 'u', etc.) at low HP: do not "
    "trade blows you can lose. ELBERETH to scare, or PRAY if HP critical.\n"
    "- Hunger: at WEAK or FAINTING, act NOW before disorientation -- EAT if food "
    "is available, else PRAY (the god feeds a starving supplicant). Waiting until "
    "fainting is often fatal (you become too disoriented to act).\n"
    "Reply ONLY with compact JSON: {\"action\": <one of the set>, "
    "\"reason\": <short>}. No prose."
)

DESCENT_SYSTEM = (
    "You are a NetHack 3.6 strategic advisor for an expert bot (Valkyrie). The "
    "bot is standing on a down-stair and about to descend. Decide whether to "
    "DESCEND now or BUILD (stay on this level/branch to gain XP, gear, and HP "
    "before going deeper). Core principle: dungeon level danger scales with "
    "depth; descending underleveled or underequipped gets you killed by fast "
    "hard-hitters (unicorns, ants) a few levels down. A rough safe heuristic is "
    "experience level >= ~2x dungeon depth in the early game, healthy HP, and "
    "some armor/weapon. But weigh the whole state. If already strong for the "
    "depth, DESCEND (don't dawdle; score needs depth).\n"
    "Reply ONLY with compact JSON: {\"decision\": \"DESCEND\"|\"BUILD\", "
    "\"reason\": <short>}. No prose."
)


def query_descent(state, base_url=None, model=None, mock=False):
    """state: role, xl, hp, max_hp, depth, hunger, ac, has_weapon, monsters_near."""
    if mock:
        return _mock_descent(state)
    base_url = base_url or os.environ.get("LORE_ORACLE_URL", "http://localhost:8000/v1")
    model = model or os.environ.get("LORE_ORACLE_MODEL", "served-model")
    resp = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": DESCENT_SYSTEM},
                     {"role": "user", "content": "State:\n" + json.dumps(state, indent=2)}],
        "temperature": 0.0, "max_tokens": 80,
    })
    return _parse_descent(resp["choices"][0]["message"]["content"].strip())


def _parse_descent(text):
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1:
        try:
            obj = json.loads(text[s:e + 1])
            dec = str(obj.get("decision", "")).upper()
            if dec in ("DESCEND", "BUILD"):
                return {"decision": dec, "reason": obj.get("reason", ""), "raw": text}
        except Exception:
            pass
    return {"decision": "DESCEND", "reason": "parse_fail_default_descend", "raw": text}


def _mock_descent(state):
    # Conservative policy: only descend with a real XP cushion for the depth and
    # healthy HP. Tunable to test whether cautious descent timing helps at all.
    xl = state.get("xl", 1); depth = max(1, state.get("depth", 1))
    hp_frac = state.get("hp", 1) / max(1, state.get("max_hp", 1))
    if xl >= 2.5 * depth + 1 and hp_frac >= 0.7:
        return {"decision": "DESCEND", "reason": "strong enough for depth"}
    return {"decision": "BUILD", "reason": "underleveled/hurt for depth"}


def build_user_prompt(state):
    """state: dict with keys like role, xl, hp, max_hp, depth, threat_glyph,
    threat_name, threat_dist, threat_speed, has_ranged, has_gloves, can_elbereth."""
    return "Threat state:\n" + json.dumps(state, indent=2) + "\nChoose one action."


def _post(url, payload, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def query_threat(state, base_url=None, model=None, mock=False):
    if mock:
        return _mock(state)
    base_url = base_url or os.environ.get("LORE_ORACLE_URL", "http://localhost:8000/v1")
    model = model or os.environ.get("LORE_ORACLE_MODEL", "served-model")
    resp = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_user_prompt(state)}],
        "temperature": 0.0, "max_tokens": 120,
    })
    text = resp["choices"][0]["message"]["content"].strip()
    return _parse(text)


def _parse(text):
    # tolerate code fences / stray text around the JSON
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1:
        try:
            obj = json.loads(text[s:e + 1])
            act = str(obj.get("action", "")).upper()
            if act in ACTIONS:
                return {"action": act, "reason": obj.get("reason", ""), "raw": text}
        except Exception:
            pass
    return {"action": None, "reason": "parse_fail", "raw": text}


def _mock(state):
    """Deterministic 'perfect knowledge' oracle for mechanism testing."""
    # Hunger emergency takes priority -- act at weak/fainting before disorientation.
    hunger = str(state.get("hunger", "")).lower()
    if hunger in ("weak", "fainting"):
        return {"action": "EAT", "reason": "hunger emergency: eat or pray before fainting"}
    g = (state.get("threat_name") or "").lower()
    if "cockatrice" in g or "chickatrice" in g:
        return {"action": "RANGED" if state.get("has_ranged") else "AVOID", "reason": "petrification risk"}
    if "floating eye" in g:
        return {"action": "RANGED" if state.get("has_ranged") else "AVOID", "reason": "paralysis risk"}
    if "unicorn" in g:
        hp_frac = state.get("hp", 1) / max(1, state.get("max_hp", 1))
        if hp_frac < 0.4:
            return {"action": "PRAY", "reason": "hp critical vs fast hitter"}
        return {"action": "ELBERETH", "reason": "scare fast hard-hitter"}
    return {"action": "FIGHT", "reason": "default"}


# ---------------------------------------------------------------------------
# Survival oracle: the "pro who wouldn't die" intervention. AutoAscend dies to
# WINNABLE situations -- fights ponies/dwarves to death, walks into cockatrice
# melee, lets food hit zero. At a near-death moment the LLM (pro judgment) picks
# the survival action AA's heuristic misses: usually DISENGAGE rather than trade
# blows. Fires only when death is plausible (HP low / starving / instant-threat).
# ---------------------------------------------------------------------------
SURVIVAL_ACTIONS = ["FLEE", "HEAL", "ELBERETH", "PRAY", "EAT", "FIGHT"]

SURVIVAL_SYSTEM = (
    "You advise an EXPERT NetHack 3.6 bot (AutoAscend) that is a strong, capable "
    "fighter -- it WINS the large majority of its fights and you should TRUST it. "
    "You are consulted only at moments flagged as risky, and your job is to catch "
    "the RARE case where it would die in a winnable situation. Overriding a fight "
    "the bot would have won WASTES the game (fleeing exposes it, engraving wastes "
    "turns) -- this is the most common mistake, so do NOT do it.\n"
    "Default to FIGHT. Only override when death is genuinely imminent:\n"
    "  FIGHT    - DEFAULT. HP not critical, or the threat is ordinary. Trust the bot.\n"
    "  HEAL     - HP critical (<~15%) AND you hold a healing potion.\n"
    "  PRAY     - HP critical OR starving (Weak/Fainting), AND prayer is safe.\n"
    "  ELBERETH - HP critical, no heal/prayer, and the attacker is a normal monster "
    "(NOT @/human, minotaur, or mindless/blind).\n"
    "  EAT      - Weak/Fainting with food/corpse available.\n"
    "  FLEE     - last resort: about to die and none of the above apply.\n"
    "Pick FIGHT unless the state clearly shows imminent death. Reply ONLY compact "
    "JSON: {\"action\": <one of the set>, \"reason\": <short>}."
)


def query_survival(state, base_url=None, model=None, mock=False):
    """state: hp, max_hp, hp_frac, hunger, threat_name, threat_adjacent,
    has_healing, prayer_safe, has_elbereth_ok, depth, xl."""
    if mock:
        return _mock_survival(state)
    base_url = base_url or os.environ.get("LORE_ORACLE_URL", "http://localhost:8000/v1")
    model = model or os.environ.get("LORE_ORACLE_MODEL", "served-model")
    resp = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": SURVIVAL_SYSTEM},
                     {"role": "user", "content": "State:\n" + json.dumps(state, indent=2)}],
        "temperature": 0.0, "max_tokens": 80,
    })
    t = resp["choices"][0]["message"]["content"].strip()
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1:
        try:
            o = json.loads(t[s:e + 1]); a = str(o.get("action", "")).upper()
            if a in SURVIVAL_ACTIONS:
                return {"action": a, "reason": o.get("reason", ""), "raw": t}
        except Exception:
            pass
    return {"action": "FLEE", "reason": "parse_fail_default_flee", "raw": t}


def _mock_survival(state):
    """Perfect-knowledge pro: disengage early, heal/pray at the brink."""
    hp = state.get("hp_frac", 1.0)
    if state.get("hunger") in ("weak", "fainting"):
        if state.get("prayer_safe") and hp < 0.3:
            return {"action": "PRAY", "reason": "starving + critical"}
        return {"action": "EAT", "reason": "starving"}
    if hp < 0.15:
        if state.get("has_healing"):
            return {"action": "HEAL", "reason": "critical, heal"}
        if state.get("prayer_safe"):
            return {"action": "PRAY", "reason": "critical, no heal"}
        return {"action": "ELBERETH", "reason": "critical, scare + buy time"}
    if hp < 0.45 and state.get("threat_adjacent"):
        return {"action": "ELBERETH" if state.get("has_elbereth_ok", True) else "FLEE",
                "reason": "losing the trade, disengage"}
    return {"action": "FIGHT", "reason": "ok to fight"}


# ---------------------------------------------------------------------------
# Food oracle: the LLM manages the food economy -- AutoAscend's #1 real-game
# death (35% starve; it runs food to ZERO and over-relies on prayer 5-23x/game).
# The LLM decides, from wiki knowledge, when to proactively eat safe corpses vs
# reserve/spend prayer vs continue. This is the knowledge AA's heuristic lacks.
# ---------------------------------------------------------------------------
FOOD_ACTIONS = ["EAT", "PRAY", "CONTINUE"]

FOOD_SYSTEM = (
    "You manage the food economy of an expert NetHack 3.6 bot. Starvation is its "
    "#1 death: it lets food drop to zero and band-aids with repeated prayer until "
    "the prayer cooldown can't keep up, then faints and dies. Your job: prevent "
    "that by banking nutrition EARLY. Given the state choose one action:\n"
    "  EAT      - go eat a safe corpse on this level now (or inventory food). Do "
    "this PROACTIVELY once Hungry, before you're desperate -- most fresh corpses "
    "are safe nutrition. Build a buffer.\n"
    "  PRAY     - emergency only: Weak/Fainting, no food/corpse, and prayer is "
    "safe (cooldown elapsed). The god feeds a starving supplicant, but tires of "
    "frequent prayer -- do not waste it.\n"
    "  CONTINUE - food is fine; keep doing what the bot was doing.\n"
    "Wiki safety: avoid OLD/rotten corpses; never cockatrice/chickatrice/floating-"
    "eye corpses; avoid kobold/zombie/were and acidic/poisonous corpses without "
    "resistance. When in doubt, fresh non-listed corpses are safe.\n"
    "Reply ONLY compact JSON: {\"action\": <EAT|PRAY|CONTINUE>, \"reason\": <short>}."
)


def query_food(state, base_url=None, model=None, mock=False):
    """state: hunger (str), hp, max_hp, has_inv_food, corpse_on_level,
    prayer_safe, depth, turn."""
    if mock:
        return _mock_food(state)
    base_url = base_url or os.environ.get("LORE_ORACLE_URL", "http://localhost:8000/v1")
    model = model or os.environ.get("LORE_ORACLE_MODEL", "served-model")
    resp = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": FOOD_SYSTEM},
                     {"role": "user", "content": "State:\n" + json.dumps(state, indent=2)}],
        "temperature": 0.0, "max_tokens": 80,
    })
    t = resp["choices"][0]["message"]["content"].strip()
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1:
        try:
            o = json.loads(t[s:e + 1]); a = str(o.get("action", "")).upper()
            if a in FOOD_ACTIONS:
                return {"action": a, "reason": o.get("reason", ""), "raw": t}
        except Exception:
            pass
    return {"action": "CONTINUE", "reason": "parse_fail", "raw": t}


def _mock_food(state):
    """Perfect-knowledge food policy (upper bound)."""
    h = str(state.get("hunger", "")).lower()
    if h in ("weak", "fainting"):
        if state.get("has_inv_food") or state.get("corpse_on_level"):
            return {"action": "EAT", "reason": "weak: eat now"}
        if state.get("prayer_safe"):
            return {"action": "PRAY", "reason": "weak, no food, prayer safe"}
        return {"action": "CONTINUE", "reason": "weak but no recourse"}
    if h == "hungry" and state.get("corpse_on_level"):
        return {"action": "EAT", "reason": "hungry: bank nutrition proactively"}
    return {"action": "CONTINUE", "reason": "food ok"}


# ---------------------------------------------------------------------------
# Endgame oracle: the LLM DRIVES the endgame descent. AutoAscend's GO_DOWN is an
# unimplemented TODO; the LORE primitives (dig, stairs, fight, flee, ...) are the
# action SPACE, and this query -- fed the wiki strategy corpus -- chooses the
# action at each step. The decision is the LLM's, not a hardcoded heuristic.
# ---------------------------------------------------------------------------
ENDGAME_ACTIONS = ["DIG_DOWN", "DESCEND_STAIRS", "EXPLORE", "FIGHT", "FLEE", "PRAY", "ELBERETH"]

ENDGAME_SYSTEM = (
    "You are the strategic controller of an expert NetHack 3.6 bot whose GOAL is "
    "to descend as deep as possible toward the endgame: the bottom of the Dungeons "
    "of Doom (the Castle), then Gehennom (entered past the Castle), down to the "
    "Vibrating Square for the invocation, the Sanctum for the Amulet, then up the "
    "Elemental and Astral planes to ascend. The bot is already strong (high XL, "
    "good armor, key resistances). Your job at each step is to pick the SINGLE best "
    "action to make safe downward progress.\n"
    "Action set:\n"
    "  DIG_DOWN       - zap a wand of digging downward to fall a level (fastest "
    "descent; only if a wand is held AND the level is diggable -- special levels "
    "like the Castle and some Gehennom levels are no-dig).\n"
    "  DESCEND_STAIRS - travel to the down-stair and take it (use when digging is "
    "blocked, or to use a known stair).\n"
    "  EXPLORE        - explore the current level to reveal stairs/items/threats.\n"
    "  FIGHT          - kill an adjacent threat blocking progress.\n"
    "  FLEE           - retreat from a deadly situation (surrounded, low HP).\n"
    "  PRAY           - emergency prayer (HP critical / starving).\n"
    "  ELBERETH       - engrave Elbereth to scare attackers and buy time.\n"
    "Use the provided KNOWLEDGE (NetHack wiki strategy) to decide. Prefer fast "
    "descent (DIG_DOWN) when safe and possible; switch to DESCEND_STAIRS when "
    "digging is blocked; handle threats only when they actually endanger progress.\n"
    "Reply ONLY with compact JSON: {\"action\": <one of the set>, \"reason\": <short>}."
)


def query_endgame(state, knowledge="", base_url=None, model=None, mock=False):
    """LLM picks the next endgame-descent action. state: depth, dungeon, xl, hp,
    max_hp, ac, hunger, has_dig_wand, on_stair, last_dig_result, adjacent_threats,
    have_downstair. knowledge: relevant wiki-corpus text (retrieval; EC-tunable)."""
    if mock:
        return _mock_endgame(state)
    base_url = base_url or os.environ.get("LORE_ORACLE_URL", "http://localhost:8000/v1")
    model = model or os.environ.get("LORE_ORACLE_MODEL", "served-model")
    user = ""
    if knowledge:
        user += "KNOWLEDGE (NetHack wiki strategy):\n" + knowledge.strip() + "\n\n"
    user += "Current state:\n" + json.dumps(state, indent=2) + "\nChoose one action."
    resp = _post(base_url.rstrip("/") + "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": ENDGAME_SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.0, "max_tokens": 100,
    })
    return _parse_endgame(resp["choices"][0]["message"]["content"].strip())


def _parse_endgame(text):
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1:
        try:
            obj = json.loads(text[s:e + 1])
            act = str(obj.get("action", "")).upper()
            if act in ENDGAME_ACTIONS:
                return {"action": act, "reason": obj.get("reason", ""), "raw": text}
        except Exception:
            pass
    return {"action": None, "reason": "parse_fail", "raw": text}


def _mock_endgame(state):
    """Perfect-knowledge endgame controller (upper bound for mechanism testing)."""
    hp_frac = state.get("hp", 1) / max(1, state.get("max_hp", 1))
    if hp_frac < 0.25:
        return {"action": "PRAY", "reason": "hp critical"}
    if state.get("adjacent_threats"):
        return {"action": "FIGHT" if hp_frac > 0.5 else "FLEE", "reason": "threat adjacent"}
    if state.get("has_dig_wand") and not state.get("level_no_dig"):
        return {"action": "DIG_DOWN", "reason": "fast descent"}
    if state.get("have_downstair"):
        return {"action": "DESCEND_STAIRS", "reason": "dig blocked, use stair"}
    return {"action": "EXPLORE", "reason": "find the downstair"}


def retrieve_knowledge(state, corpus_path=None, max_chars=3500):
    """Naive retrieval: return corpus lines relevant to the current situation.
    This is the seam the EC search will optimize (which chunks, how ranked). For
    now: keyword-match on depth/dungeon/threat against the strategy corpus."""
    path = corpus_path or os.environ.get(
        "LORE_CORPUS", "/workspace/ASCENSION_STRATEGY.md")
    try:
        text = open(path).read()
    except Exception:
        return ""
    terms = set()
    dungeon = str(state.get("dungeon", "")).lower()
    if state.get("depth", 0) >= 25 or "gehennom" in dungeon:
        terms |= {"gehennom", "castle", "vibrating", "invocation", "amulet",
                  "demon", "dig", "valley", "sanctum", "plane"}
    else:
        terms |= {"descend", "dig", "depth", "stair", "dungeon"}
    for t in (state.get("adjacent_threats") or []):
        terms.add(str(t).lower())
    # paragraph-level keyword match
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for p in paras:
        pl = p.lower()
        score = sum(1 for t in terms if t in pl)
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    out, total = [], 0
    for _, p in scored:
        if total + len(p) > max_chars:
            break
        out.append(p)
        total += len(p)
    return "\n\n".join(out)


# synthetic death-states drawn from the gap analysis
TEST_STATES = [
    {"name": "cockatrice_DL14", "role": "Val", "xl": 10, "hp": 80, "max_hp": 110,
     "depth": 14, "threat_glyph": "c", "threat_name": "cockatrice", "threat_dist": 1,
     "has_ranged": True, "has_gloves": False, "can_elbereth": True},
    {"name": "white_unicorn_lowhp", "role": "Val", "xl": 9, "hp": 18, "max_hp": 75,
     "depth": 4, "threat_glyph": "u", "threat_name": "white unicorn", "threat_speed": 24,
     "threat_dist": 1, "has_ranged": False, "has_gloves": True, "can_elbereth": True},
    {"name": "white_unicorn_okhp", "role": "Val", "xl": 9, "hp": 55, "max_hp": 75,
     "depth": 4, "threat_glyph": "u", "threat_name": "white unicorn", "threat_speed": 24,
     "threat_dist": 2, "has_ranged": False, "has_gloves": True, "can_elbereth": True},
]


if __name__ == "__main__":
    import sys
    mock = "--mock" in sys.argv
    for st in TEST_STATES:
        d = query_threat(st, mock=mock)
        print(f"{st['name']:>22}  ->  {d['action']:<9} ({d.get('reason','')})")
