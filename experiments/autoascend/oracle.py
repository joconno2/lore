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

ACTIONS = ["FIGHT", "RANGED", "AVOID", "FLEE", "ELBERETH", "PRAY"]

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
    xl = state.get("xl", 1); depth = max(1, state.get("depth", 1))
    hp_frac = state.get("hp", 1) / max(1, state.get("max_hp", 1))
    if xl >= 2 * depth and hp_frac >= 0.6:
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
