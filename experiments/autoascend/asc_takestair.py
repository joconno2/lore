"""Is stair-taking actually broken, or does dig-down just handle every level first?
stair_descents=0 across all descent runs -- but DIG_DOWN is tried before
DESCEND_STAIRS, so on diggable levels stairs are never needed. This isolates the
stair-take mechanism: teleport, reveal (^F), find the downstair, report whether it
is bfs-reachable, then go_to it and press '>' and report whether depth increased.
Answers: (a) after reveal, is the downstair reachable? (b) does go_to reach it?
(c) does '>' descend? -- the crux for passing no-dig Gehennom levels.

Usage: python asc_takestair.py <seed> <out>
"""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop(); lore_scenario.patch_ring_parse()
from autoascend import global_logic as _gl
from autoascend.glyph import G
from autoascend import utils as _u
from autoascend.exceptions import AgentFinished
import autoascend.agent as _ag

R = {"seed": seed}


def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    low = agent.env.env.unwrapped.env
    try:
        lore_scenario._do_teleport(agent, 28)
    except Exception as e:
        R["tp_err"] = repr(e)[:60]
    R["depth"] = int(agent.blstats.depth)
    R["dungeon"] = int(agent.current_level().dungeon_number)
    # reveal
    try:
        low.step(27); low.step(6); low.step(27)
        agent.step(_ag.A.Command.ESC); agent.inventory.update()
    except Exception as e:
        R["reveal_err"] = repr(e)[:60]
    # locate downstair
    try:
        lvl = agent.current_level(); bf = agent.bfs()
        dmask = _u.isin(lvl.objects, G.STAIR_DOWN)
        downs = list(zip(*dmask.nonzero()))
        R["n_downstairs"] = len(downs)
        if not downs:
            R["result"] = "NO_DOWNSTAIR_REVEALED"
            raise AgentFinished()
        reach = [(y, x) for y, x in downs if bf[y, x] != -1]
        R["reachable"] = len(reach)
        R["my_pos"] = [int(agent.blstats.y), int(agent.blstats.x)]
        R["down_pos"] = [[int(y), int(x)] for y, x in downs][:5]
        if not reach:
            R["result"] = "DOWNSTAIR_WALLED (not bfs-reachable after reveal)"
            raise AgentFinished()
        # ROBUST single-step nav to the stair (path+move, tolerate AA errors)
        ty, tx = int(reach[0][0]), int(reach[0][1])
        before = int(agent.blstats.depth)
        errs = 0
        for _ in range(80):
            y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
            if (y0, x0) == (ty, tx):
                break
            try:
                path = agent.path(y0, x0, ty, tx)
                if not path or len(path) < 2:
                    break
                ny, nx = int(path[1][0]), int(path[1][1])
                if agent.current_level().walkable[ny, nx]:
                    agent.move(ny, nx)
                else:
                    break
            except Exception as e:
                errs += 1
                R.setdefault("nav_err", repr(e)[:50])
                try:
                    agent.go_to(ty, tx, max_steps=1)
                except Exception:
                    break
        R["nav_errs"] = errs
        R["reached_stair"] = (int(agent.blstats.y), int(agent.blstats.x)) == (ty, tx)
        # press '>'
        try:
            low.step(ord('>')); low.step(13); low.step(13)
            agent.step(_ag.A.Command.ESC); agent.inventory.update()
        except Exception as e:
            R["descend_err"] = repr(e)[:60]
        after = int(agent.blstats.depth)
        R["depth_before"] = before; R["depth_after"] = after
        R["result"] = "DESCENDED" if after > before else "AT_STAIR_BUT_NO_DESCEND"
    except AgentFinished:
        raise
    except Exception as e:
        import traceback
        R["scan_err"] = repr(e)[:80]; R["tb"] = traceback.format_exc()[-300:]
    raise AgentFinished()


_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: R["end"] = repr(e)[:60]
json.dump(R, open(OUT, "w"), default=str, indent=1)
print("DONE", R.get("result"), "reach=%s" % R.get("reachable"), flush=True)
