"""Probe: are the invocation items wishable, and what do they look like in
AutoAscend's inventory model? Foundation for the endgame-sequence (option b):
Candelabrum of Invocation + 7 wax candles + Bell of Opening + Book of the Dead.
No teleport, no combat -- just wish on DL1 and dump inventory."""
import sys, json, gym, nle, lore_patches, lore_scenario, time
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()

from autoascend import global_logic as _gl

WISHES = ["The Candelabrum of Invocation", "7 wax candles",
          "The Bell of Opening", "The Book of the Dead"]

def probe_global(self):
    agent = self.agent
    if not getattr(agent, "_done", False):
        agent.__dict__["_done"] = True
        import autoascend.agent as _ag
        for it in WISHES:
            lore_scenario._do_wish(agent, it)
        try:
            agent.step(_ag.A.Command.ESC); agent.inventory.update()
        except Exception:
            pass
        from autoascend.agent import flatten_items
        names = []
        for it in flatten_items(agent.inventory.items):
            names.append("%s | cat=%s" % (str(it), getattr(it, "category", "?")))
        lore_patches.COUNTERS["inv"] = names
        # also raw obs inventory (ground truth)
        raw = []
        for nm, oc, lt in zip(agent.last_observation['inv_strs'],
                              agent.last_observation['inv_oclasses'],
                              agent.last_observation['inv_letters']):
            s = bytes(nm).decode('latin1').strip('\x00').strip()
            if s:
                raw.append("%s [%s] cat=%d" % (s, chr(int(lt)), int(oc)))
        lore_patches.COUNTERS["raw_inv"] = raw
    raise __import__("autoascend.exceptions", fromlist=["AgentFinished"]).AgentFinished()

_gl.GlobalLogic.global_strategy = probe_global
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:60]
json.dump({"seed": seed, "raw_inv": lore_patches.COUNTERS.get("raw_inv"),
           "inv": lore_patches.COUNTERS.get("inv")}, open(OUT, "w"), default=str, indent=1)
print("DONE", flush=True)
