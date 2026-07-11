"""Diagnose WHY AutoAscend's Sokoban solver fails 100% here (structural bug to
fix, NOT an LLM task -- Sokoban is deterministic/solvable). Reach Sokoban via the
macro director on a known Sokoban-reaching seed, then on the FIRST solver
invocation dump: the live board (walls/boulders/stairs as ASCII), whether any of
AA's precomputed sokomaps match, and the full traceback of the failure."""
import sys, json, gym, nle, lore_patches, traceback, os
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_patches.apply_macro_director(mock=True)  # to reach Sokoban
if os.environ.get("LORE_SOKOFIX") == "1":
    lore_patches.apply_sokoban_fix()

from autoascend import global_logic as _gl
from autoascend import soko_solver
from autoascend.glyph import G
from autoascend import utils
import numpy as np

DUMP = {"entered": 0, "board": None, "match": None, "traceback": None, "boulders": None}
orig = _gl.GlobalLogic.solve_sokoban_strategy

def _ascii(agent):
    obj = agent.current_level().objects
    wall = utils.isin(obj, G.WALL)
    boul = utils.isin(agent.glyphs, G.BOULDER)
    down = utils.isin(obj, G.STAIR_DOWN); up = utils.isin(obj, G.STAIR_UP)
    ys, xs = wall.nonzero()
    if len(ys) == 0: return "NO WALLS", None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    rows = []
    for y in range(y0, y1 + 1):
        r = ""
        for x in range(x0, x1 + 1):
            if boul[y, x]: r += "0"
            elif wall[y, x]: r += "|"
            elif down[y, x]: r += ">"
            elif up[y, x]: r += "<"
            else: r += "."
        rows.append(r)
    return "\n".join(rows), [(int(y), int(x)) for y, x in zip(*boul.nonzero())]

def _match(agent):
    wall_map = utils.isin(agent.current_level().objects, G.WALL)
    for i, (smap, ans) in enumerate(soko_solver.maps.items()):
        sk = soko_solver.convert_map(smap)
        try:
            off = np.array(min(zip(*wall_map.nonzero()))) - np.array(min(zip(*(sk.sokomap == soko_solver.WALL).nonzero())))
            mask = wall_map[off[0]:off[0]+sk.sokomap.shape[0], off[1]:off[1]+sk.sokomap.shape[1]]
            if (mask & (sk.sokomap == soko_solver.WALL) == mask).all():
                return i
        except Exception:
            pass
    return -1

def patched(self, *a, **k):
    strat = orig(self, *a, **k)
    of = strat.strategy
    def sf():
        gen = of()
        cond = next(gen)
        yield cond
        if not cond: return
        if not DUMP["entered"]:
            DUMP["entered"] = 1
            try:
                DUMP["board"], DUMP["boulders"] = _ascii(self.agent)
                DUMP["match"] = _match(self.agent)
            except Exception as e:
                DUMP["board"] = "dump err %r" % e
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        except BaseException as e:
            DUMP["traceback"] = traceback.format_exc()[-1500:]
            DUMP["exc"] = repr(e)
            raise
    from autoascend.strategy import Strategy
    return Strategy(sf, getattr(strat, "config", None))

_gl.GlobalLogic.solve_sokoban_strategy = patched
env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: DUMP["end"] = repr(e)[:80]
DUMP["n_maps"] = len(soko_solver.maps)
try:
    s = w.get_summary()
    DUMP["final_milestone"] = str(s.get("milestone")); DUMP["score"] = s.get("score")
    DUMP["depth"] = s.get("level_num"); DUMP["xl"] = s.get("experience_level")
    DUMP["sokoban_dropped"] = s.get("sokoban_dropped")
    DUMP["skips"] = lore_patches.COUNTERS.get("sokoban_skip_done_move")
    from autoascend.item import flatten_items as _fi
    DUMP["kit"] = [str(it) for it in _fi(w.agent.inventory.items)
                   if any(k in str(it).lower() for k in ("bag of holding", "reflection", "amulet"))]
except Exception:
    pass
json.dump(DUMP, open(OUT, "w"), default=str, indent=1)
print("DONE", flush=True)
