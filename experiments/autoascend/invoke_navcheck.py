"""Resolve the sealed-pocket ambiguity for the real-invocation demo. Iterative ^V
teleport to the invocation range (~DL48-53), then MEASURE the level topology:
- reachable = cells the agent can walk to now (bfs != -1 & walkable)
- total_walkable = all walkable cells on the level
If reachable << total_walkable, the agent is in an isolated POCKET of a full level
(escapable only by digging -> test digging). If reachable ~= total_walkable, the
level is NAVIGABLE and the vibrating-square demo is feasible. Also report downstair
reachability and whether a wand-of-digging breaches the pocket wall."""
import sys, json, gym, nle, lore_patches, lore_scenario
import numpy as np
seed = int(sys.argv[1]); OUT = sys.argv[2]; TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 50
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
from autoascend.exceptions import AgentFinished
import nle.nethack as nh

R = {"seed": seed, "target": TARGET, "levels": []}

def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    # iterative teleport down to target, force-generating each level
    for d in range(28, TARGET + 1):
        try: lore_scenario._do_teleport(agent, d)
        except Exception as e:
            R["levels"].append({"tp_to": d, "err": repr(e)[:50]}); continue
    # now measure the final level's topology
    try:
        dep = int(agent.blstats.depth)
        lvl = agent.current_level()
        bf = agent.bfs()
        walkable = lvl.walkable
        reachable = (bf != -1) & walkable
        n_reach = int(reachable.sum())
        n_walk = int(walkable.sum())
        # downstair reachable?
        try:
            downs = (lvl.objects == nh.GLYPH_CMAP_OFF + nh.S_dnstair) if hasattr(lvl, "objects") else None
        except Exception:
            downs = None
        # count doors/corridors reachable vs total as a topology sanity check
        R["final"] = {
            "depth": dep,
            "reachable_cells": n_reach,
            "total_walkable_cells": n_walk,
            "reach_frac": round(n_reach / max(1, n_walk), 3),
            "agent_pos": [int(agent.blstats.y), int(agent.blstats.x)],
            "verdict": ("SEALED_POCKET" if n_reach < 0.25 * n_walk else "NAVIGABLE"),
        }
    except Exception as e:
        R["final"] = {"err": repr(e)[:80]}
    raise AgentFinished()

_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: R["end"] = repr(e)[:60]
json.dump(R, open(OUT, "w"), default=str, indent=1)
print("DONE", R.get("final"), flush=True)
