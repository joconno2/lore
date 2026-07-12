"""Why does the descent heal reflex never fire? Dump what the char actually holds
after the tank setup: teleport-less, just wish the survival kit, then print every
inventory line (letter, oclass, name) and run the exact heal-detection logic the
reflex uses. Isolates: are the full-healing potions present + detected as 'healing'?
"""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
lore_scenario.patch_ring_parse()
from autoascend import global_logic as _gl
from autoascend.exceptions import AgentFinished
import autoascend.agent as _ag
import nle.nethack as _nh

KIT = ["8 blessed potions of full healing",
       "blessed +3 gray dragon scale mail", "blessed ring of free action",
       "blessed wand of digging (0:8)", "5 blessed food rations"]
R = {"seed": seed}

def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    for it in KIT:
        try: lore_scenario._do_wish(agent, it)
        except Exception as e: R.setdefault("wish_err", []).append(repr(e)[:50])
    try:
        agent.step(_ag.A.Command.ESC); agent.inventory.update()
    except Exception: pass
    inv = []
    try:
        for nm, oc, lt in zip(agent.last_observation['inv_strs'],
                              agent.last_observation['inv_oclasses'],
                              agent.last_observation['inv_letters']):
            if int(lt) == 0: continue
            s = bytes(nm).decode('latin1').strip('\x00').strip()
            if not s: continue
            inv.append({"lt": chr(int(lt)), "oc": int(oc), "name": s,
                        "is_potion": int(oc) == _nh.POTION_CLASS,
                        "has_healing": 'healing' in s.lower()})
    except Exception as e:
        R["inv_err"] = repr(e)[:80]
    R["inv"] = inv
    R["n_healing_potions"] = sum(1 for i in inv if i["is_potion"] and i["has_healing"])
    R["hp"] = int(agent.blstats.hitpoints)
    R["max_hp"] = int(agent.blstats.max_hitpoints)
    R["xl"] = int(agent.blstats.experience_level)
    R["POTION_CLASS"] = int(_nh.POTION_CLASS)
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
print("DONE n_healing=%s hp=%s/%s" % (R.get("n_healing_potions"), R.get("hp"), R.get("max_hp")), flush=True)
