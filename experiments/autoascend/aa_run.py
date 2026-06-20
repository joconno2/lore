"""Run AutoAscend with the LORE patch layer applied. Args: seeds (comma-sep), out."""
import sys, json, time
import gym
import lore_patches
applied = lore_patches.apply()
print("LORE patches applied:", applied, flush=True)

from autoascend.env_wrapper import EnvWrapper

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [126, 135, 137, 138, 141]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/aa_patched.json"

rows = []
for seed in seeds:
    t0 = time.time()
    env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                     agent_args=dict(panic_on_errors=False, verbose=False))
    env.env.seed(seed, seed)
    try:
        env.main()
    except BaseException as e:
        env.end_reason = f"harness_exc: {repr(e)[:160]}"
    s = env.get_summary()
    stats = {k: v for k, v in s.items() if "sokoban" in str(k).lower()}
    r = {"seed": seed, "score": s.get("score"), "level_num": s.get("level_num"),
         "xl": s.get("experience_level"), "milestone": s.get("milestone"),
         "panic_total": len(getattr(env.agent, "all_panics", [])),
         "end_reason": str(s.get("end_reason"))[:90], "soko_stats": stats,
         "t": round(time.time()-t0, 1)}
    rows.append(r)
    print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"patched": True, "rows": rows}, open(OUT, "w"), indent=2, default=str)
crashes = sum(1 for r in rows if "AssertionError" in r["end_reason"])
print("SUMMARY patched crashes=", crashes, "/", len(rows), flush=True)
