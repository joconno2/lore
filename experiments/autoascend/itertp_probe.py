"""Probe: does ITERATIVE wizard ^V teleport (to deepest+1 repeatedly) force-generate
deeper Gehennom levels, reaching the invocation level (~DL45-53) without maze nav?
A single ^V clamps at the deepest generated level (~DL29); this tests stepping down."""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
TRAJ = []
def probe(self):
    agent = self.agent
    if not getattr(agent, "_done", False):
        agent.__dict__["_done"] = True
        try:
            for d in range(28, 60):
                lore_scenario._do_teleport(agent, d)
                try:
                    dep = int(agent.blstats.depth); dn = int(getattr(agent.blstats, "dungeon_number", -1))
                except Exception:
                    dep, dn = -1, -1
                TRAJ.append((d, dep, dn))
                if dep <= (TRAJ[-2][1] if len(TRAJ) > 1 else 0) and len(TRAJ) > 3:
                    # depth stopped increasing across a couple tries -> clamped
                    if len(TRAJ) >= 3 and TRAJ[-1][1] == TRAJ[-3][1]:
                        break
        except Exception as e:
            TRAJ.append(("err", repr(e)[:60], 0))
    raise __import__("autoascend.exceptions", fromlist=["AgentFinished"]).AgentFinished()
_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: TRAJ.append(("end", repr(e)[:50], 0))
json.dump({"seed": seed, "traj": TRAJ, "max_depth_reached": max((t[1] for t in TRAJ if isinstance(t[1], int)), default=-1)}, open(OUT, "w"), default=str, indent=1)
print("DONE", flush=True)
