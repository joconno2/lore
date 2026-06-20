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
