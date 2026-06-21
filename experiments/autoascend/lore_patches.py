"""LORE intervention layer: runtime patches over the frozen AutoAscend base.

We do not edit AutoAscend's source. We wrap it from outside. First intervention:
the Sokoban solver crashes with a bare AssertionError when the real board
diverges from its precomputed solution (boulder destroyed/displaced). That kills
~10% of runs, disproportionately the deepest ones. AutoAscend already has a
graceful abandon path (log 'sokoban_dropped' -> advance milestone -> raise
AgentPanic). We route the desync into that path instead of letting it crash.
"""
import functools
from autoascend.strategy import Strategy
from autoascend import global_logic as _gl
from autoascend.exceptions import AgentPanic

_Milestone = _gl.Milestone

# Reliable telemetry: AutoAscend's StatsLogger.log_event raises KeyError on
# unknown names (fixed dict), so we count intervention firings here instead.
COUNTERS = {}


def _bump(name):
    COUNTERS[name] = COUNTERS.get(name, 0) + 1


def apply():
    # Operate on the Strategy object the original method returns, not its
    # internals -- robust to however many decorators are stacked on it.
    orig_method = _gl.GlobalLogic.solve_sokoban_strategy

    def patched(self, *a, **k):
        orig_strat = orig_method(self, *a, **k)        # a Strategy
        orig_factory = orig_strat.strategy             # callable -> generator

        def safe_factory():
            gen = orig_factory()
            try:
                cond = next(gen)                       # condition yield
            except StopIteration as e:
                return
            yield cond
            if not cond:
                return
            try:
                next(gen)                              # run body
            except StopIteration as e:
                return e.value
            except AssertionError:
                # Sokoban solver desynced from the real board. Abandon via
                # AutoAscend's own recovery: advance milestone, panic.
                try:
                    self.agent.stats_logger.log_event("sokoban_desync_lore")
                except Exception:
                    pass
                try:
                    self.milestone = _Milestone(int(self.milestone) + 1)
                except Exception:
                    pass
                raise AgentPanic("sokoban solver desync (LORE patch)")

        return Strategy(safe_factory, getattr(orig_strat, "config", None))

    _gl.GlobalLogic.solve_sokoban_strategy = patched
    return ["safe_solve_sokoban (Strategy-wrap)"]


# --- Intervention #2: oracle melee veto on knowledge-dependent instadeaths ---
# AutoAscend melees cockatrices (not in its never-melee set) -> petrification
# deaths, including its deepest runs. We intercept melee_monster_priority: when
# the target is a flagged-dangerous monster, query the oracle; if it says not to
# melee, return a strongly negative priority so AutoAscend falls back to ranged
# or avoidance. Only flagged monsters trigger a query, so queries stay rare.

_PETRIFIERS = ("cockatrice", "chickatrice")
_NON_MELEE_ACTIONS = {"RANGED", "AVOID", "FLEE", "ELBERETH", "PRAY"}


def apply_oracle_veto(mock=True, base_url=None, model=None):
    import oracle as _oracle
    from autoascend.combat import fight_heur as _fh

    orig = _fh.melee_monster_priority

    def patched_melee_priority(agent, monsters, monster):
        try:
            _, y, x, mon, _ = monster
            name = (mon.mname or "").lower()
            if any(p in name for p in _PETRIFIERS):
                bl = agent.blstats
                state = {
                    "role": str(getattr(agent, "character", ""))[:3],
                    "xl": int(bl.experience_level), "hp": int(bl.hitpoints),
                    "max_hp": int(bl.max_hitpoints), "depth": int(bl.depth),
                    "threat_name": name, "threat_dist": 1,
                    "has_ranged": False, "has_gloves": False, "can_elbereth": True,
                }
                d = _oracle.query_threat(state, base_url=base_url, model=model, mock=mock)
                _bump("veto_query_" + name.replace(" ", "_"))
                if d.get("action") in _NON_MELEE_ACTIONS:
                    _bump("veto_fired_" + name.replace(" ", "_"))
                    return -1000  # veto melee; AutoAscend falls back to ranged/avoid
        except Exception:
            pass
        return orig(agent, monsters, monster)

    _fh.melee_monster_priority = patched_melee_priority
    return ["oracle_melee_veto (mock=%s)" % mock]


# --- Intervention #3: oracle-gated descent timing -------------------------
# AutoAscend leaves DL1 at XL8 then rushes to DL4-5 and dies to fast hitters
# (unicorns: all XL8-9 on DL4-5). move() is the single choke point for stair
# traversal. On a '>' descent we ask the oracle DESCEND vs BUILD. BUILD holds
# the descent (the agent levels/explores instead). Re-queried per (level, XL)
# so rising XP eventually flips to DESCEND; a per-level hold cap guarantees no
# deadlock. Queries are at most ~once per level per XP gain, so cheap.

def apply_descent_gate(mock=True, base_url=None, model=None, max_holds_per_level=6):
    import oracle as _oracle
    from autoascend.agent import Agent
    from autoascend.exceptions import AgentPanic

    orig_move = Agent.move

    def gated_move(self, y, x=None):
        try:
            d = self.calc_direction(self.blstats.y, self.blstats.x, y, x) if x is not None else y
        except Exception:
            d = y
        if d == ">":
            try:
                bl = self.blstats
                key = self.current_level().key()
                cache = self.__dict__.setdefault("_lore_descent_cache", {})
                holds = self.__dict__.setdefault("_lore_descent_holds", {})
                ck = (key, int(bl.experience_level))
                dec = cache.get(ck)
                if dec is None:
                    state = {"role": "Val", "xl": int(bl.experience_level),
                             "hp": int(bl.hitpoints), "max_hp": int(bl.max_hitpoints),
                             "depth": int(bl.depth),
                             "hunger": int(getattr(bl, "hunger_state", 1))}
                    dec = _oracle.query_descent(state, base_url=base_url, model=model, mock=mock).get("decision", "DESCEND")
                    cache[ck] = dec
                    _bump("descent_query")
                if dec == "BUILD":
                    holds[key] = holds.get(key, 0) + 1
                    if holds[key] <= max_holds_per_level:
                        _bump("descent_hold")
                        raise AgentPanic("lore: build before descending DL%d (XL%d)"
                                         % (int(bl.depth), int(bl.experience_level)))
                    _bump("descent_cap_reached")
                else:
                    _bump("descent_go")
            except AgentPanic:
                raise
            except Exception:
                pass
        return orig_move(self, y, x)

    Agent.move = gated_move
    return ["oracle_descent_gate (mock=%s, cap=%d)" % (mock, max_holds_per_level)]
