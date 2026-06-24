import sys, json, gym, nle, lore_patches, lore_scenario, time
seed=int(sys.argv[1]); lore=sys.argv[2]=="1"; target=int(sys.argv[3]); OUT=sys.argv[4]
lore_patches.apply()
if lore:
    lore_patches.apply_crash_recovery(); lore_patches.apply_oracle_veto(mock=True)
lore_scenario.patch_enhance_noop(); lore_scenario.install_scenario(target, wishes=["12 blessed potions of gain level","blessed +3 gray dragon scale mail","blessed +2 long sword","blessed ring of free action"])
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = EnvWrapper = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
t0=time.time()
try: w.main()
except BaseException as e: w.end_reason=repr(e)[:80]
s=w.get_summary()
json.dump({"seed":seed,"lore":lore,"target":target,"score":s.get("score"),"turns":s.get("turns"),
          "xl":s.get("experience_level"),"end":str(w.__dict__.get("end_reason"))[:60],
          "tp":lore_patches.COUNTERS.get("scenario_teleport",0),"tp_depth":lore_patches.COUNTERS.get("tp_depth"),
          "t":round(time.time()-t0),"ac":int(getattr(w.agent.blstats,"armor_class",99)) if getattr(w,"agent",None) and getattr(w.agent,"blstats",None) else None,"xl_after":lore_patches.COUNTERS.get("xl_after"),"wishes":lore_patches.COUNTERS.get("wishes")}, open(OUT,"w"), default=str)
print("DONE", flush=True)
