"""Minimal repro: wish 2 rings, wear via low.step, then a normal agent.step ->
capture the exact traceback of the 'on right hand' AssertionError."""
import sys, json, gym, nle, lore_patches, lore_scenario, traceback
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
import autoascend.agent as _A
import nle.nethack as _nh
TB = {}
def probe(self):
    agent = self.agent
    if not getattr(agent, "_done", False):
        agent.__dict__["_done"] = True
        low = agent.env.env.unwrapped.env
        for it in ["blessed ring of free action", "blessed ring of fire resistance"]:
            lore_scenario._do_wish(agent, it)
        try: agent.step(_A.A.Command.ESC); agent.inventory.update()
        except Exception: pass
        # wear rings via low.step
        rings = [chr(int(lt)) for oc, lt in zip(agent.last_observation['inv_oclasses'],
                 agent.last_observation['inv_letters']) if int(oc)==_nh.RING_CLASS and int(lt)!=0]
        TB["rings"] = rings
        for l, f in zip(rings[:2], ['r','l']):
            low.step(ord('P')); low.step(ord(l)); low.step(ord(f)); low.step(13)
        # now a NORMAL agent.step -> does it assert?
        try:
            agent.step(_A.A.Command.ESC); agent.inventory.update()
            TB["result"] = "no assert"
        except BaseException as e:
            TB["exc"] = repr(e)[:100]; TB["tb"] = traceback.format_exc()[-1200:]
    raise __import__("autoascend.exceptions", fromlist=["AgentFinished"]).AgentFinished()
_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: TB.setdefault("end", repr(e)[:80])
json.dump(TB, open(OUT,"w"), default=str, indent=1)
print("DONE", flush=True)
