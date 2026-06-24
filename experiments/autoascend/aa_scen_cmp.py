"""Paired deep-scenario: teleport to target depth, same seed, base vs +LORE
late-game layer (oracle veto + crash recovery). Measures survival from depth.
Args: target seed lore(0/1) out."""
import sys, json
import gym, nle
import lore_patches, lore_scenario
from autoascend.env_wrapper import EnvWrapper
target=int(sys.argv[1]); seed=int(sys.argv[2]); lore=sys.argv[3]=="1"; OUT=sys.argv[4]
lore_patches.apply()                      # sokoban (both arms)
if lore:
    lore_patches.apply_crash_recovery()
    lore_patches.apply_oracle_veto(mock=True)   # late-game instadeath avoidance (cockatrice/eye)
lore_scenario.patch_enhance_noop()
lore_scenario.install_teleport(target)
env_raw = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env_raw.seed(seed, seed)
except Exception: pass
w = EnvWrapper(env_raw, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: w.end_reason=f"exc:{repr(e)[:90]}"
s=w.get_summary()
res={"seed":seed,"lore":lore,"target":target,"depth":s.get("level_num"),"xl":s.get("experience_level"),
     "score":s.get("score"),"turns":s.get("turns"),"end":str(s.get("end_reason"))[:60],
     "counters":dict(lore_patches.COUNTERS)}
json.dump(res, open(OUT,"w"), default=str)
print("DONE", res, flush=True)
