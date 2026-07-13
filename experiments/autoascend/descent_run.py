import sys, json, gym, nle, lore_patches, lore_scenario, time
seed=int(sys.argv[1]); lore=sys.argv[2]=="1"; target=int(sys.argv[3]); OUT=sys.argv[4]
lore_patches.apply()
if lore:
    lore_patches.apply_crash_recovery(); lore_patches.apply_oracle_veto(mock=True)
lore_scenario.patch_enhance_noop()
# PROPER ASCENSION KIT (per docs/GEHENNOM_DESCENT_PLAYBOOK.md). Prayer AND Elbereth
# are dead in Gehennom, so survival is entirely gear + tactics. No genocide shortcut.
KIT=[
    # XP/HP tank -- quaffed first in setup (before healing exists), then rest wished
    "12 blessed potions of gain level","12 blessed potions of gain level","12 blessed potions of gain level",
    # RESISTANCES + protection (worn): the hard gates
    "blessed +7 gray dragon scale mail",   # MAGIC RESISTANCE (blocks finger of death etc.)
    "blessed +7 shield of reflection",     # REFLECTION (bounces death rays from Orcus/demons)
    "blessed +5 oilskin cloak",            # blocks eel GRAB -> no drowning (3/13 deaths); + AC
    "blessed +7 helm of telepathy",        # see monsters through walls -> avoid ambush instadeaths
    "blessed +7 pair of speed boots",      # speed -> outrun demons
    "blessed +7 pair of leather gloves",   # AC + wield cockatrice safely
    "blessed amulet of life saving",       # last-resort revive
    "blessed ring of free action",         # anti paralysis/sleep lock
    "blessed ring of sustain ability",     # anti mind-flayer INT drain + stat drain
    # weapons: silver vs demons/vampires/undead; cold vs fire-resistant demons
    "blessed +7 silver saber","blessed +7 frost brand",
    # prayer/Elbereth substitutes
    "blessed unicorn horn",                # cure blind/confuse/stun/sick/stat-drain
    "blessed scroll of scare monster","blessed scroll of scare monster","blessed scroll of scare monster",
    "blessed scroll of scare monster","blessed scroll of scare monster",  # the Gehennom panic button
    # escape stack
    "blessed bag of holding",
    "blessed wand of teleportation (0:7)", # escape (works on no-teleport levels)
    "blessed wand of digging (0:8)","blessed wand of digging (0:8)","blessed wand of digging (0:8)",
    "blessed wand of death (0:8)","blessed wand of death (0:8)",  # instakill demons (reflection-safe)
    "blessed wand of fire (0:8)",          # green-slime cure (no prayer here)
    "blessed wand of cold (0:8)",          # strand giant eels
    "blessed scroll of teleportation","blessed scroll of teleportation","blessed scroll of teleportation",
    "cursed potion of gain level","cursed potion of gain level",  # instant escape UP a level
    # navigation: magic-map each maze level
    "blessed scroll of magic mapping","blessed scroll of magic mapping","blessed scroll of magic mapping",
    "blessed scroll of magic mapping","blessed scroll of magic mapping","blessed scroll of magic mapping",
    # healing (bagged; tracked by letter for the heal reflex)
    "8 blessed potions of full healing",
    # food: non-rotting
    "blessed lizard corpse","blessed lizard corpse","blessed lizard corpse","blessed lizard corpse",
    "5 blessed food rations","5 blessed food rations",
    # poison-resistance source (eaten fresh at full HP in setup)
    "blessed killer bee corpse","blessed killer bee corpse",
]
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
        # RELIABLE vibrating-square detection: check EVERY step's message (the
        # per-descend-iteration check misses it -- explore1 takes many steps and
        # the transient 'strange vibration' message gets overwritten).
        try:
            m=bytes(obs["message"]).decode("latin1").lower()
            if "vibrat" in m and lore_patches.COUNTERS.get("vibration_found") is None:
                lore_patches.COUNTERS["vibration_found"]=1
                lore_patches.COUNTERS["vibration_msg"]=m.strip()[:60]
        except Exception: pass
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
          "sick_after_eat":C.get("sick_after_eat"),"setup_prayed_sick":C.get("setup_prayed_sick"),"heal_letter":C.get("heal_letter"),
          "descent_prays":C.get("descent_prays"),"reflex_fights":C.get("reflex_fights"),"reflex_flees":C.get("reflex_flees"),
          "reveals":C.get("reveals"),"reveal_err":C.get("reveal_err"),"dig_to_stair":C.get("dig_to_stair"),
          "hit_iter_cap":C.get("hit_iter_cap"),"last_action":C.get("last_action"),"act_exc":C.get("act_exc"),
          "mons_seen":C.get("mons_seen"),"nearest_mon_min":C.get("nearest_mon_min"),"min_hp_frac":C.get("min_hp_frac"),"stair_steps":C.get("stair_steps"),
          "af_action":C.get("agentfinished_action"),"af_iter":C.get("agentfinished_iter"),
          "af_tb":C.get("agentfinished_tb"),
          "down_diag":C.get("down_diag"),"level_no_dig":C.get("level_no_dig"),
          "stair_descents":C.get("stair_descents"),
          "t":round(time.time()-t0),"xl_after":C.get("xl_after"),"wishes":C.get("wishes")},
          open(OUT,"w"), default=str)
print("DONE", flush=True)
