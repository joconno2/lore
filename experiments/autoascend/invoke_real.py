"""Real invocation demo (rung 4, AA-0%). Iterative-teleport to deep Gehennom with
the invocation kit; on each candidate level (target-6 .. target, reached by DOWN
teleport) systematically go_to every reachable cell watching for the vibrating
square's 'strange vibration' message. Reports the level+pos when found."""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); target = int(sys.argv[2]); OUT = sys.argv[3]
lore_scenario.patch_enhance_noop()
lore_scenario.patch_ring_parse()   # tolerate worn rings + candelabrum state strings
from autoascend import global_logic as _gl
import autoascend.agent as _A
from autoascend.exceptions import AgentFinished
import nle.nethack as _nh
R = {"levels": [], "found": None}
def _msg(agent):
    try: return bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip().lower()
    except Exception: return ""
def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False): raise AgentFinished()
    agent.__dict__["_done"] = True
    # SEARCH PHASE: survival kit ONLY -- the Candelabrum isn't in AA's item DB, so
    # having it in inventory makes every inventory.update() assert and corrupts the
    # agent (blocks go_to). Wish the invocation items LATER, once on the square.
    for it in ["12 blessed potions of gain level","12 blessed potions of gain level",
               "blessed +3 gray dragon scale mail","blessed ring of free action"]:
        lore_scenario._do_wish(agent, it)
    try:
        agent.step(_A.A.Command.ESC); agent.inventory.update()
        for _ in range(24):
            pot=None
            for x in _A.flatten_items(agent.inventory.items):
                if getattr(x,"category",None)==_nh.POTION_CLASS: pot=x;break
            if pot is None: break
            try: agent.inventory.quaff(pot)
            except Exception: break
        s=agent.inventory.wear_best_stuff()
        if s.check_condition(): s.run()
        agent.step(_A.A.Command.ESC); agent.inventory.update()
    except Exception as e: R["setup_err"]=repr(e)[:50]
    # search each candidate level, reached by DOWN teleport (43->..->target)
    for lvl in range(max(target-6, 28), target+1):
        try: lore_scenario._do_teleport(agent, lvl)
        except Exception as e: R["levels"].append([lvl,"tp_err %r"%e]); continue
        try: cur=int(agent.blstats.depth)
        except Exception: cur=lvl
        if "vibrat" in _msg(agent):
            R["found"]={"level":cur,"pos":[int(agent.blstats.y),int(agent.blstats.x)],"how":"on-arrival"}; break
        found=False; visited=0
        try:
            for _ in range(150):
                if "vibrat" in _msg(agent):
                    found=True; R["found"]={"level":cur,"pos":[int(agent.blstats.y),int(agent.blstats.x)],"how":"walked","visited":visited}; break
                # AA's explore1 populates + covers the level model (raw go_to fails
                # on the empty post-teleport level). Stop when fully explored.
                e=agent.exploration.explore1(0)
                if e.check_condition():
                    e.run(); visited+=1
                else:
                    break
        except AgentFinished:
            R["levels"].append([cur,"died",visited]); raise
        except Exception as _we:
            R.setdefault("walk_err", repr(_we)[:50])
        R["levels"].append([cur,"FOUND" if found else "no-vib", visited])
        if found: break
    raise AgentFinished()
_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: R["end"]=repr(e)[:50]
json.dump(R, open(OUT,"w"), default=str, indent=1); print("DONE", flush=True)
