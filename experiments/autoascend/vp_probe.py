"""Valley-placement probe: ^V-by-depth to 26 lands in the MAIN dungeon (dnum=0),
not the Gehennom Valley of the Dead (dnum=1, Gehennom L1). Find which target
reaches the Valley. Teleport to each target, report dungeon_number + level_number
(within-branch) + depth + any 'Valley'/Gehennom message. Quick (teleport + read,
no game play). Unique vp_ prefix so it can't collide with Jim's v26 runs.

Usage: python vp_probe.py <seed> <out>
"""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
from autoascend.exceptions import AgentFinished

R = {"seed": seed, "targets": []}


def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    for tgt in range(25, 33):
        try:
            lore_scenario._do_teleport(agent, tgt)
        except Exception as e:
            R["targets"].append({"tgt": tgt, "err": repr(e)[:40]}); continue
        try:
            bl = agent.blstats
            msg = bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip()
            R["targets"].append({
                "tgt": tgt,
                "depth": int(bl.depth),
                "dnum": int(getattr(bl, "dungeon_number", -1)),
                "lnum": int(getattr(bl, "level_number", -1)),
                "msg": msg[:50],
            })
        except Exception as e:
            R["targets"].append({"tgt": tgt, "read_err": repr(e)[:40]})
    raise AgentFinished()


_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: R["end"] = repr(e)[:50]
json.dump(R, open(OUT, "w"), default=str, indent=1)
print("DONE", flush=True)
