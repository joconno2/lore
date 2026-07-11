import sys, json, gym, nle, lore_patches, lore_scenario, time
seed=int(sys.argv[1]); lore=sys.argv[2]=="1"; target=int(sys.argv[3]); OUT=sys.argv[4]
lore_patches.apply()
if lore:
    lore_patches.apply_crash_recovery(); lore_patches.apply_oracle_veto(mock=True)
lore_scenario.patch_enhance_noop()
KIT=["12 blessed potions of gain level","12 blessed potions of gain level",
     "12 blessed potions of gain level",   # ~36 XL -> big HP to tank Gehennom swarms
     "blessed +3 gray dragon scale mail","blessed +2 long sword",
     "blessed +3 helmet","blessed +3 pair of high boots","blessed +3 pair of leather gloves",
     "blessed +3 cloak of protection",     # full armor -> AC-15 (survives the swarm)
     "8 blessed potions of full healing",  # kept for the in-descent heal reflex
     "blessed ring of free action","blessed ring of fire resistance",
     "blessed amulet of reflection",
     "4 blessed killer bee corpses","4 blessed kobold corpses",
     "5 blessed food rations","5 blessed food rations",  # extra: descent eats too
     "blessed wand of digging (0:8)","blessed wand of digging (0:8)",
     "blessed wand of digging (0:8)","blessed wand of digging (0:8)"]
if lore:
    lore_scenario.install_descent(target, wishes=KIT)
else:
    # base = AutoAscend in same body/depth but NO endgame descent planner: bounded
    # local play (its actual behavior in Gehennom -- it has no model to descend).
    lore_scenario.install_scenario(target, wishes=KIT)
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
# Capture EVERY tty frame so the real death screen survives (the agent's own
# last_observation is stale at death). Stash the last 3 frames' bottom lines.
_LAST=[]
_orig_step=env.step
def _hook_step(a):
    r=_orig_step(a)
    try:
        obs=r[0]
        tc=obs["tty_chars"] if isinstance(obs,dict) else obs[0]
        import numpy as _np
        txt="\n".join(bytes(row).decode("latin1").rstrip() for row in tc)
        _LAST.append(txt)
        if len(_LAST)>12: _LAST.pop(0)
    except Exception: pass
    return r
env.step=_hook_step
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(env, agent_args=dict(panic_on_errors=False, verbose=False))
t0=time.time()
try: w.main()
except BaseException as e: w.end_reason=repr(e)[:80]
# capture the raw final tty (wizard-mode death screen breaks AA's parser, so
# end_reason is often empty) -- read the actual death cause off the screen.
death=""
try:
    # death cause is on an earlier frame than the wizard-mode score screen.
    # scan recent frames for the cause line.
    KW=("killed by","you die","turn to stone","stone","petrif","drown","disintegr",
        "you feel","level teleport","genocid","starv","Killed","engulf","swallow","bit")
    hits=[]
    for fr in reversed(_LAST):
        for l in fr.split("\n"):
            ls=l.strip()
            if ls and any(k.lower() in ls.lower() for k in KW):
                hits.append(ls)
    death=" || ".join(dict.fromkeys(hits))[:400] or (_LAST[-1].split("\n")[0] if _LAST else "")
except Exception as _e:
    death="(tty capture failed: %r)"%_e
s=w.get_summary()
C=lore_patches.COUNTERS
json.dump({"seed":seed,"lore":lore,"target":target,"score":s.get("score"),"turns":s.get("turns"),
          "xl":s.get("experience_level"),"end":str(w.__dict__.get("end_reason"))[:60],
          "tp_depth":C.get("tp_depth"),"max_depth":C.get("max_depth"),"descents":C.get("descents"),
          "descend_iters":C.get("descend_iters"),"stairs_seen":C.get("stairs_seen"),"descend_err":C.get("descend_err"),
          "jewelry":C.get("equipped_jewelry"),"ac_equip":C.get("ac_after_equip"),"eaten":C.get("corpses_eaten"),
          "explore1_ran":C.get("explore1_ran"),"search_ran":C.get("search_ran"),
          "pre_wear":C.get("pre_wear"),"pre_eat":C.get("pre_eat"),"pre_fight2":C.get("pre_fight2"),
          "pre_engulf":C.get("pre_engulf"),"pre_emergency":C.get("pre_emergency"),
          "downstair_glyphs":C.get("downstair_glyphs"),"explored_cells":C.get("explored_cells"),"dungeon_num":C.get("dungeon_num"),
          "digs":C.get("digs"),"dig_fail":C.get("dig_fail"),"dig_panic":C.get("dig_panic"),"wands_seen":C.get("wands_seen"),
          "zap_msg":C.get("zap_msg"),"oracle_err":C.get("oracle_err"),"policy":C.get("policy"),
          "oracle_actions":{k:C[k] for k in C if k.startswith("oracle_") and k!="oracle_err"},
          "first_reach":{int(k.split("_")[-1]):C[k] for k in C if k.startswith("firstreach_")},
          "survived_depth":{K:(max([int(k.split("_")[-1]) for k in C if k.startswith("firstreach_") and (int(s.get("turns") or 0)-C[k])>=K], default=0)) for K in (20,50,100)},
          "death":death,"end_reason":str(getattr(w,"end_reason",""))[:120],
          "t_after_wishes":C.get("t_after_wishes"),"t_after_quaff":C.get("t_after_quaff"),
          "t_after_eat":C.get("t_after_eat"),"t_after_equip":C.get("t_after_equip"),
          "vibration_found":C.get("vibration_found"),"vibration_pos":C.get("vibration_pos"),"healing_kept":C.get("healing_kept"),"descent_heals":C.get("descent_heals"),
          "af_action":C.get("agentfinished_action"),"af_iter":C.get("agentfinished_iter"),
          "af_tb":C.get("agentfinished_tb"),
          "down_diag":C.get("down_diag"),"level_no_dig":C.get("level_no_dig"),
          "stair_descents":C.get("stair_descents"),
          "t":round(time.time()-t0),"xl_after":C.get("xl_after"),"wishes":C.get("wishes")},
          open(OUT,"w"), default=str)
print("DONE", flush=True)
