"""Test: does genociding deadly monster classes (a real in-game technique, read on
the safe start level -> global effect) let the agent SURVIVE at DL49 to explore for
the vibrating square? Wish genocide scrolls + survival kit, genocide the deadly
Gehennom classes, iterative-teleport to DL49, explore, report survival."""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
import autoascend.agent as _A
import nle.nethack as _nh
from autoascend.exceptions import AgentFinished
# deadly Gehennom monster classes (genocide clears them globally): & major demons,
# L liches, U umber hulks, m mind flayers(actually 'h'), V vamps, M mummies, ; eels
CLASSES = ["&", "L", "U", "h", "V", "M", ";", "n"]
R = {"genocided": [], "vibration_found": None}
def _msg(agent):
    try: return bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip().lower()
    except Exception: return ""
def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False): raise AgentFinished()
    agent.__dict__["_done"] = True
    low = agent.env.env.unwrapped.env
    # wish survival kit + one blessed genocide scroll per class
    for it in ["12 blessed potions of gain level","12 blessed potions of gain level",
               "blessed +3 gray dragon scale mail","blessed ring of free action"] + \
              ["blessed scroll of genocide" for _ in CLASSES]:
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
    # read genocide scrolls: 'r' -> scroll letter -> "What class...?" -> class -> enter
    for cls in CLASSES:
        try:
            slt=None
            for nm,oc,lt in zip(agent.last_observation['inv_strs'],agent.last_observation['inv_oclasses'],agent.last_observation['inv_letters']):
                if int(oc)==_nh.SCROLL_CLASS and int(lt)!=0 and 'genocide' in bytes(nm).decode('latin1').lower():
                    slt=chr(int(lt));break
            if slt is None: break
            low.step(ord('r')); low.step(ord(slt))
            for ch in cls: low.step(ord(ch))
            low.step(13); low.step(13); low.step(13)
            R["genocided"].append(cls)
            agent.step(_A.A.Command.ESC); agent.inventory.update()
        except Exception as e: R["geno_err"]=repr(e)[:40]; break
    # teleport to DL49, explore, measure survival
    try: lore_scenario._do_teleport(agent, 49)
    except Exception as e: R["tp_err"]=repr(e)[:40]
    try: R["tp_depth"]=int(agent.blstats.depth)
    except Exception: pass
    steps=0
    try:
        for steps in range(300):
            if "vibrat" in _msg(agent): R["vibration_found"]=1; break
            e=agent.exploration.explore1(0)
            if e.check_condition(): e.run()
            else: break
    except AgentFinished:
        R["survived_explore_steps"]=steps; R["died_exploring"]=1; raise
    except Exception as ex: R["explore_err"]=repr(ex)[:40]
    R["survived_explore_steps"]=steps
    try: R["explored"]=int(agent.current_level().was_on.sum())
    except Exception: pass
    raise AgentFinished()
_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: R["end"]=repr(e)[:50]
json.dump(R, open(OUT,"w"), default=str, indent=1); print("DONE", flush=True)
