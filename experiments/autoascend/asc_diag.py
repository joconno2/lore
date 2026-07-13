"""Ascension diagnostic: engine config, per-seed depth/XL/milestone/killer.
Goal: find where runs stall. Are shallow deaths underleveled (low XL) -> XP lever,
or do deep runs also die leveled -> survival lever? Captures the depth-vs-XL cloud."""
import sys, json, time, re
import gym
import lore_patches

applied = lore_patches.apply()                       # safe_solve_sokoban
# --- engine config (the median-DL5 baseline: DL1-unstick + stability/safety) ---
for fn in ("apply_unstick_dl1", "apply_crash_recovery", "apply_petrifier_avoidance",
           "apply_anti_starvation", "apply_decode_hardening", "apply_obs_sanitize"):
    f = getattr(lore_patches, fn, None)
    if f:
        try:
            r = f(); applied.append(fn)
        except Exception as e:
            applied.append(f"{fn}:ERR:{repr(e)[:40]}")
print("ENGINE applied:", applied, flush=True)

from autoascend.env_wrapper import EnvWrapper

seeds = [int(s) for s in sys.argv[1].split(",")]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/asc_diag.json"

def killer(reason):
    m = re.search(r"[Kk]illed by (?:an? )?([a-z ]+?)(?:,|\.|$)", reason or "")
    if m: return m.group(1).strip()
    for k in ("starv", "poison", "petrif", "AssertionError", "trap", "drown", "food"):
        if k.lower() in (reason or "").lower(): return k
    return (reason or "?")[:30]

rows = []
for seed in seeds:
    t0 = time.time()
    env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                     agent_args=dict(panic_on_errors=False, verbose=False))
    env.env.seed(seed, seed)
    dungeon = None; xl = None; depth = None
    try:
        env.main()
    except BaseException as e:
        env.end_reason = f"harness_exc: {repr(e)[:120]}"
    try:
        bl = env.agent.blstats
        xl = int(bl.experience_level); depth = int(bl.depth)
        dungeon = int(getattr(env.agent.current_level(), "dungeon_number", -1))
    except Exception:
        pass
    s = env.get_summary()
    er = str(s.get("end_reason"))
    r = {"seed": seed, "score": s.get("score"), "depth": depth,
         "level_num": s.get("level_num"), "xl": xl if xl is not None else s.get("experience_level"),
         "dungeon": dungeon, "milestone": str(s.get("milestone")),
         "killer": killer(er), "end": er[:80], "turns": s.get("step_count") or s.get("turns"),
         "t": round(time.time()-t0, 1)}
    rows.append(r)
    print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"applied": applied, "rows": rows}, open(OUT, "w"), indent=2, default=str)
print("DONE", len(rows), flush=True)
