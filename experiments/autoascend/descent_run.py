import sys, json, gym, nle, lore_patches, lore_scenario, time
seed=int(sys.argv[1]); lore=sys.argv[2]=="1"; target=int(sys.argv[3]); OUT=sys.argv[4]
lore_patches.apply()
if lore:
    lore_patches.apply_crash_recovery(); lore_patches.apply_oracle_veto(mock=True)
lore_scenario.patch_enhance_noop()
KIT=["12 blessed potions of gain level","12 blessed potions of gain level",
     "blessed +3 gray dragon scale mail","blessed +2 long sword",
     "blessed ring of free action","blessed amulet of reflection"]
if lore:
    lore_scenario.install_descent(target, wishes=KIT)
else:
    # base = AutoAscend in same body/depth but NO endgame descent planner: bounded
    # local play (its actual behavior in Gehennom -- it has no model to descend).
    lore_scenario.install_scenario(target, wishes=KIT)
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
t0=time.time()
try: w.main()
except BaseException as e: w.end_reason=repr(e)[:80]
s=w.get_summary()
C=lore_patches.COUNTERS
json.dump({"seed":seed,"lore":lore,"target":target,"score":s.get("score"),"turns":s.get("turns"),
          "xl":s.get("experience_level"),"end":str(w.__dict__.get("end_reason"))[:60],
          "tp_depth":C.get("tp_depth"),"max_depth":C.get("max_depth"),"descents":C.get("descents"),
          "descend_iters":C.get("descend_iters"),"stairs_seen":C.get("stairs_seen"),"descend_err":C.get("descend_err"),
          "t":round(time.time()-t0),"xl_after":C.get("xl_after"),"wishes":C.get("wishes")},
          open(OUT,"w"), default=str)
print("DONE", flush=True)
