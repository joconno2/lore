"""LORE planning layer over the frozen AutoAscend base.

Not single-action patches (those failed -- the base's reflexes are already good).
A persistent, prioritized GOAL STACK that shapes behavior over many turns and
targets the structural gap AutoAscend has: no long-horizon, knowledge-grounded
planning. See docs/PLANNING_LAYER.md.

Step 1 (this file): the plan/goal model + an executor preempt + one hardcoded,
KB-grounded goal (stockpile_food) to prove a persistent goal drives multi-turn
behavior and beats the base on starvation. No LLM yet; the planner and retrieval
come once the mechanism is validated.
"""
import json
import os

import lore_patches  # reuse COUNTERS / _bump

# --- structured KB: corpse safety ----------------------------------------
_KB_DIR = os.environ.get("LORE_KB", "/workspace/data/parsed")


def _load_safe_corpses():
    """Monster names whose fresh corpse is safe + nutritious to eat (no poison,
    petrify, polymorph, lycanthropy, acid, aggravate, intrinsic-risk). Grounded
    in corpse_effects.json; conservative -- unknown -> unsafe."""
    safe = set()
    try:
        data = json.load(open(os.path.join(_KB_DIR, "corpse_effects.json")))
        rows = data.values() if isinstance(data, dict) else data
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = (r.get("name") or r.get("monster") or "").lower()
            if not name:
                continue
            blob = json.dumps(r).lower()
            bad = any(k in blob for k in ("poison", "petrif", "stone", "polymorph",
                      "lycanthrop", "were", "acid", "aggravate", "hallucinat",
                      "stun", "teleport", "invisible", "speed toggle", "cure"))
            if not bad:
                safe.add(name)
    except Exception:
        pass
    # always-safe staples (fast fallback if the file shape differs)
    safe.update({"lichen", "newt", "lizard", "gnome", "jackal", "sewer rat",
                 "giant rat", "kobold", "hill orc", "goblin", "dwarf", "floating eye"})
    safe.discard("floating eye")  # corpse safe to eat but skip (telepathy nuance)
    return safe


_SAFE_CORPSES = None


def safe_corpses():
    global _SAFE_CORPSES
    if _SAFE_CORPSES is None:
        _SAFE_CORPSES = _load_safe_corpses()
    return _SAFE_CORPSES


# --- plan / goal model ----------------------------------------------------
class Goal:
    """A persistent objective. entry/exit are predicates over the agent.
    act(agent) performs one turn-advancing step toward the goal, or returns
    False if it can't act this turn (executor falls through to the base)."""
    id = "goal"
    priority = 0

    def entry(self, agent):  # should this goal be active now?
        return False

    def exit(self, agent):   # is it satisfied / should drop?
        return True

    def act(self, agent):    # one step toward it; True if acted (turn advanced)
        return False


class Plan:
    def __init__(self, goals):
        self.goals = sorted(goals, key=lambda g: -g.priority)

    def active(self, agent):
        for g in self.goals:
            try:
                if not g.exit(agent) and g.entry(agent):
                    return g
            except Exception:
                continue
        return None


# --- a first executable goal (KB-grounded) -------------------------------
class StockpileFood(Goal):
    """Bank nutrition while safe corpses are available, up to satiated, so the
    agent enters deep food-poor stretches with a reserve. Base eats at HUNGRY;
    this banks earlier and preserves inventory rations. Real long-horizon food
    economy is the LLM planner's job; this proves the executor + grounding."""
    id = "stockpile_food"
    priority = 40

    def entry(self, agent):
        from autoascend.glyph import Hunger
        h = int(agent.blstats.hunger_state)
        if h >= int(Hunger.SATIATED) + 1 and h < int(Hunger.HUNGRY):
            # not hungry yet, but bank if a safe corpse is right here
            return self._safe_corpse_below(agent)
        return int(Hunger.HUNGRY) <= h < int(Hunger.FAINTING)

    def exit(self, agent):
        from autoascend.glyph import Hunger
        return int(agent.blstats.hunger_state) <= int(Hunger.SATIATED)

    def _safe_corpse_below(self, agent):
        try:
            below = agent.inventory.items_below_me or []
            sc = safe_corpses()
            for it in below:
                t = (getattr(it, "text", "") or "").lower()
                if "corpse" in t and any(n in t for n in sc):
                    return True
        except Exception:
            pass
        return False

    def act(self, agent):
        try:
            agent.eat_corpses_from_ground(only_below_me=True).run()
            return True
        except Exception:
            try:
                agent.eat_corpses_from_ground(only_below_me=False).run()
                return True
            except Exception:
                return False


DEFAULT_GOALS = [StockpileFood()]


# --- executor: install the plan as a top-priority preempt ----------------
def install(plan=None):
    from autoascend.strategy import Strategy
    from autoascend import global_logic as _gl

    plan = plan or Plan(DEFAULT_GOALS)
    orig_global = _gl.GlobalLogic.global_strategy

    def make_exec(self):
        agent = self.agent

        def factory():
            g = plan.active(agent)
            if g is None:
                yield False
                return
            now = int(agent.blstats.time)
            if agent.__dict__.get("_lore_plan_turn") == now:
                yield False
                return
            yield True
            agent.__dict__["_lore_plan_turn"] = now
            lore_patches._bump("plan_goal_" + g.id)
            try:
                g.act(agent)
            except Exception:
                pass

        return Strategy(factory)

    def patched_global(self):
        return orig_global(self).preempt(self.agent, [make_exec(self)])

    _gl.GlobalLogic.global_strategy = patched_global
    return ["lore_planner.install (goals=%s)" % [g.id for g in plan.goals]]


# =========================================================================
# Goal library (structural gaps the base lacks) + LLM planner + re-planning
# executor. The LLM PLANS (which goals, what priority) from state + retrieved
# corpus knowledge; goals are executable code. EC later tunes the interface.
# =========================================================================

from autoascend.glyph import Hunger


class FoodReserve(Goal):
    """Long-horizon food: keep lizard/lichen corpses as a never-rot reserve and
    bank safe corpses up to (not past) satiated. The real starvation fix -- the
    base eats when hungry but keeps no deliberate reserve. (corpus sec.3)"""
    id = "food_reserve"
    priority = 50

    def entry(self, agent):
        # Rare, bounded interrupt: only when actually hungry AND a safe corpse is
        # underfoot -> eat in place. Never wander (that hijacked the base and
        # stranded the agent on DL1). The base keeps control otherwise.
        h = int(agent.blstats.hunger_state)
        return int(Hunger.HUNGRY) <= h < int(Hunger.FAINTING) and self._corpse_here(agent)

    def exit(self, agent):
        return int(agent.blstats.hunger_state) < int(Hunger.HUNGRY)

    def _corpse_here(self, agent):
        try:
            sc = safe_corpses()
            for it in (agent.inventory.items_below_me or []):
                t = (getattr(it, "text", "") or "").lower()
                if "corpse" in t and any(n in t for n in sc):
                    return True
        except Exception:
            pass
        return False

    def act(self, agent):
        try:
            agent.eat_corpses_from_ground(only_below_me=True).run()
            return True
        except Exception:
            return False


class EmergencyPrayer(Goal):
    """HP/starvation backstop -- prayer ONLY as a major-trouble rescue, never
    routine (routine pray-for-hunger angers the god; corpus sec.1)."""
    id = "emergency_prayer"
    priority = 95

    def entry(self, agent):
        bl = agent.blstats
        major_hp = bl.hitpoints <= 5 or bl.hitpoints * 7 <= bl.max_hitpoints
        starving = int(bl.hunger_state) >= int(Hunger.FAINTING)
        try:
            safe = agent.is_safe_to_pray(400)
        except Exception:
            safe = False
        return safe and (major_hp or starving)

    def exit(self, agent):
        return False  # entry already gates it

    def act(self, agent):
        try:
            agent.pray()
            return True
        except Exception:
            return False


# Excluded as duplicative-and-harmful to the base (a strong tactical agent):
# - StockpileFood: broad entry + wandering eat hijacked the base (stranded DL1).
# - EmergencyPrayer: the base already emergency-prays; ours over-prayed (27x),
#   angering the god -> starvation. Per-turn tactical goals fight the base.
# Kept: only a bounded eat-in-place. The real value is STRATEGIC route-direction
# (branch order, prep-gate, endgame) -- a separate director layer, not per-turn.
GOAL_LIBRARY = {g.id: g for g in [FoodReserve()]}


# --- corpus retrieval (phase-relevant strategy text) ---------------------
_CORPUS_PATH = os.environ.get("LORE_CORPUS", "/workspace/data/strategy/ASCENSION_STRATEGY.md")
_CORPUS = None


def corpus_text():
    global _CORPUS
    if _CORPUS is None:
        try:
            _CORPUS = open(_CORPUS_PATH).read()
        except Exception:
            _CORPUS = ""
    return _CORPUS


def _phase(agent):
    d = int(agent.blstats.depth)
    if d <= 3:
        return "early"
    if d <= 10:
        return "mid"
    return "deep"


# --- LLM planner: orders/activates goals from state + knowledge ----------
_PLAN_SYSTEM = (
    "You are the strategic planner for a NetHack Valkyrie bot built on a strong "
    "symbolic base. You do NOT pick single actions -- the base handles tactics. "
    "You choose which long-horizon GOALS are active and their priority, given the "
    "state and the strategy notes. Output ONLY compact JSON: "
    "{\"goals\": [<goal_id>, ...]} ordered most-important first, a subset of the "
    "provided goal ids. No prose."
)


def llm_plan(agent, base_url=None, model=None, mock=False):
    ids = list(GOAL_LIBRARY.keys())
    if mock:
        return ids  # mock: all goals active by static priority
    try:
        import oracle as _oracle
        bl = agent.blstats
        try:
            import game_state as _gs
            st = _gs.extract(agent)
        except Exception:
            st = {"xl": int(bl.experience_level), "hp": int(bl.hitpoints),
                  "depth": int(bl.depth), "hunger": int(bl.hunger_state)}
        phase = _phase(agent)
        notes = corpus_text()
        # crude phase-relevant slice to bound the prompt
        sec = {"early": "## 1.", "mid": "## 2.", "deep": "## 4."}.get(phase, "## 1.")
        idx = notes.find(sec)
        knowledge = notes[idx:idx + 1800] if idx >= 0 else notes[:1800]
        goal_desc = {gid: (GOAL_LIBRARY[gid].__doc__ or "").split("\n")[0] for gid in ids}
        user = (f"State: {json.dumps(st)}\nPhase: {phase}\n"
                f"Goals available: {json.dumps(goal_desc)}\n"
                f"Strategy notes:\n{knowledge}\n"
                "Which goals are active now, ordered by priority?")
        url = (base_url or os.environ.get("LORE_ORACLE_URL", "http://localhost:8000/v1")).rstrip("/") + "/chat/completions"
        resp = _oracle._post(url,
                             {"model": model or os.environ.get("LORE_ORACLE_MODEL", "served-model"),
                              "messages": [{"role": "system", "content": _PLAN_SYSTEM},
                                           {"role": "user", "content": user}],
                              "temperature": 0.0, "max_tokens": 80})
        txt = resp["choices"][0]["message"]["content"]
        s, e = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[s:e + 1])
        chosen = [g for g in obj.get("goals", []) if g in GOAL_LIBRARY]
        return chosen or ids
    except Exception:
        return ids


def install_planner(base_url=None, model=None, mock=False, replan_every=400):
    """Re-planning executor: the LLM refreshes the active goal ordering every
    `replan_every` turns (or on milestone change); each turn the top active goal
    runs. Bounded queries -> cheap on long runs and the EC cadence knob."""
    from autoascend.strategy import Strategy
    from autoascend import global_logic as _gl

    orig_global = _gl.GlobalLogic.global_strategy
    state = {"order": list(GOAL_LIBRARY.keys()), "last_replan": -10 ** 9, "last_ms": None}

    def make_exec(self):
        agent = self.agent

        def factory():
            now = int(agent.blstats.time)
            ms = getattr(self, "milestone", None)
            if now - state["last_replan"] >= replan_every or ms != state["last_ms"]:
                state["order"] = llm_plan(agent, base_url=base_url, model=model, mock=mock)
                state["last_replan"] = now
                state["last_ms"] = ms
                lore_patches._bump("replan")
            g = None
            for gid in state["order"]:
                cand = GOAL_LIBRARY.get(gid)
                try:
                    if cand and not cand.exit(agent) and cand.entry(agent):
                        g = cand
                        break
                except Exception:
                    continue
            if g is None:
                yield False
                return
            if agent.__dict__.get("_lore_plan_turn") == now:
                yield False
                return
            yield True
            agent.__dict__["_lore_plan_turn"] = now
            lore_patches._bump("plan_goal_" + g.id)
            try:
                g.act(agent)
            except Exception:
                pass

        return Strategy(factory)

    def patched_global(self):
        return orig_global(self).preempt(self.agent, [make_exec(self)])

    _gl.GlobalLogic.global_strategy = patched_global
    return ["lore_planner.install_planner (mock=%s, goals=%s, replan_every=%d)"
            % (mock, list(GOAL_LIBRARY.keys()), replan_every)]


# --- Strategic goal: priest protection donation (confirmed base gap) ------
# AutoAscend never buys protection. Donating ~400*XL gold to a co-aligned
# Minetown priest grants intrinsic protection (AC), the one big survival lever
# the base lacks. Fires only when adjacent to a priest with gold -> rare, clean,
# additive in the proven-transparent seam. Survival -> depth -> endgame surface.
class ProtectionDonation(Goal):
    id = "protection_donation"
    priority = 70

    def __init__(self):
        self._done = False
        self._tries = 0

    def entry(self, agent):
        if self._done or self._tries >= 3:
            return False
        try:
            if int(agent.blstats.gold) < 400:
                return False
            return self._priest(agent) is not None
        except Exception:
            return False

    def exit(self, agent):
        return self._done or self._tries >= 3

    def _priest(self, agent):
        bl = agent.blstats
        for m in agent.get_visible_monsters():
            try:
                _, y, x, mon, _ = m
                if "priest" in (mon.mname or "").lower() and max(abs(y - bl.y), abs(x - bl.x)) == 1:
                    return (y, x)
            except Exception:
                continue
        return None

    def act(self, agent):
        import nle.nethack as nh
        p = self._priest(agent)
        if p is None:
            return False
        self._tries += 1
        bl = agent.blstats
        amount = min(int(bl.gold), 400 * max(1, int(bl.experience_level)))
        try:
            with agent.atom_operation():
                agent.step(nh.Command.CHAT)
                msg = (agent.single_message or "").lower()
                if "direction" in msg or "whom" in msg:
                    agent.direction(p[0], p[1])
                    msg = (agent.single_message or "").lower()
                # priest asks for a contribution -> answer the getlin
                if "contribut" in msg or "how much" in msg or "gold piece" in msg:
                    agent.type_text(str(amount))
                    agent.step("\r")
                    self._done = True
                    lore_patches._bump("protection_donated")
                    return True
        except Exception:
            pass
        return False
