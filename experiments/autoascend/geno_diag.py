"""Diagnose the genocide scroll-read desync: wish 1 genocide scroll on DL1, read it
step-by-step, capturing the game message after EVERY keypress -> see where the
prompt sequence diverges from assumption."""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
import autoascend.agent as _A
import nle.nethack as _nh
from autoascend.exceptions import AgentFinished
TR = []
def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False): raise AgentFinished()
    agent.__dict__["_done"] = True
    low = agent.env.env.unwrapped.env
    lore_scenario._do_wish(agent, "blessed scroll of genocide")
    try: agent.step(_A.A.Command.ESC); agent.inventory.update()
    except Exception: pass
    def m(r):
        try:
            o = r[0] if isinstance(r, tuple) else r
            return bytes(o['message']).decode('latin1').strip('\x00').strip()[:70]
        except Exception: return "?"
    # find the genocide scroll letter
    slt = None
    for nm, oc, lt in zip(agent.last_observation['inv_strs'], agent.last_observation['inv_oclasses'], agent.last_observation['inv_letters']):
        if int(oc) == _nh.SCROLL_CLASS and int(lt) != 0:
            slt = chr(int(lt)); break
    inv=[]
    for nm,oc,lt in zip(agent.last_observation['inv_strs'],agent.last_observation['inv_oclasses'],agent.last_observation['inv_letters']):
        t=bytes(nm).decode('latin1').strip('\x00').strip()
        if t: inv.append("%s|oc=%d"%(t,int(oc)))
    TR.append(("full_inv", inv))
    TR.append(("scroll_letter", slt))
    if slt:
        TR.append(("after 'r'", m(low.step(ord('r')))))
        TR.append((f"after '{slt}'", m(low.step(ord(slt)))))
        TR.append(("after '&'", m(low.step(ord('&')))))
        TR.append(("after enter", m(low.step(13))))
        TR.append(("after enter2", m(low.step(13))))
    raise AgentFinished()
_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: TR.append(("end", repr(e)[:50]))
json.dump({"trace": TR}, open(OUT,"w"), default=str, indent=1); print("DONE", flush=True)
