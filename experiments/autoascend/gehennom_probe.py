"""Probe: can wizard level-teleport place the agent deep in Gehennom, and what
does the level look like? Foundation for the real invocation (option b/ascension).
Teleport to a target depth, dump depth/dungeon/features/message."""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); target = int(sys.argv[2]); OUT = sys.argv[3]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
from autoascend.glyph import G
from autoascend import utils

DUMP = {}
def probe(self):
    agent = self.agent
    if not getattr(agent, "_done", False):
        agent.__dict__["_done"] = True
        try:
            lore_scenario._do_teleport(agent, target)
        except Exception as e:
            DUMP["tp_err"] = repr(e)[:80]
        bl = agent.blstats
        DUMP["depth"] = int(bl.depth)
        DUMP["dungeon_number"] = int(getattr(bl, "dungeon_number", -1))
        DUMP["level_number"] = int(getattr(bl, "level_number", -1))
        try:
            DUMP["msg"] = bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip()
        except Exception: pass
        # count stairs/features visible
        try:
            obj = agent.current_level().objects
            DUMP["has_downstair"] = bool(utils.isin(obj, G.STAIR_DOWN).any())
            DUMP["has_upstair"] = bool(utils.isin(obj, G.STAIR_UP).any())
        except Exception as e:
            DUMP["feat_err"] = repr(e)[:60]
    raise __import__("autoascend.exceptions", fromlist=["AgentFinished"]).AgentFinished()

_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: DUMP["end"] = repr(e)[:60]
json.dump({"seed": seed, "target": target, **DUMP}, open(OUT, "w"), default=str, indent=1)
print("DONE", flush=True)
