"""Real invocation demo (rung 4). Iterative-teleport to deep Gehennom with the
invocation kit; on each level DL(target..target-6) SINGLE-STEP-walk watching for
the 'strange vibration' message (the vibrating square). Robust: catches death per
level and moves on. If found, records level+pos (the real invocation is then the
proven ritual on that square)."""
import sys, json, gym, nle, lore_patches, lore_scenario, random
seed = int(sys.argv[1]); target = int(sys.argv[2]); OUT = sys.argv[3]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
import autoascend.agent as _A
from autoascend.exceptions import AgentFinished, AgentPanic
import nle.nethack as _nh
DIRS = [ord(c) for c in "hjklyubn"]
R = {"levels": [], "found": None}
def _msg(agent):
    try: return bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip().lower()
    except Exception: return ""
def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    low = agent.env.env.unwrapped.env
    for it in ["The Candelabrum of Invocation","7 wax candles","The Bell of Opening",
               "The Book of the Dead","12 blessed potions of gain level",
               "12 blessed potions of gain level","blessed +3 gray dragon scale mail",
               "blessed ring of free action"]:
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
    for lvl in range(target, max(target-7, 27), -1):
        try: lore_scenario._do_teleport(agent, lvl)
        except Exception as e: R["levels"].append([lvl,"tp_err"]); continue
        try: cur=int(agent.blstats.depth)
        except Exception: cur=lvl
        found=False; steps=0
        try:
            for steps in range(80):
                if "vibrat" in _msg(agent):
                    found=True; R["found"]={"level":cur,"pos":[int(agent.blstats.y),int(agent.blstats.x)],"msg":_msg(agent)[:60]}; break
                low.step(random.choice(DIRS)); low.step(13)  # single step + clear
        except AgentFinished:
            R["levels"].append([cur,"died",steps]); 
            raise
        except Exception:
            pass
        R["levels"].append([cur,"FOUND" if found else "no-vib",steps])
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
