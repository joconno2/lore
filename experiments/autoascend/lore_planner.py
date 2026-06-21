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
