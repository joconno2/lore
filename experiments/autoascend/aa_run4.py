"""AutoAscend + sokoban patch + petrifier avoidance (heatmap repulsion).
Args: seeds(csv) out."""
import sys, json, time
import gym
import lore_patches

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [105, 107, 145]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/aa_petav.json"

print("sokoban:", lore_patches.apply(), flush=True)
print("petrifier avoidance:", lore_patches.apply_petrifier_avoidance(), flush=True)

from autoascend.env_wrapper import EnvWrapper

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
    r = {"seed": seed, "score": s.get("score"), "level_num": s.get("level_num"),
         "xl": s.get("experience_level"), "turns": s.get("turns"),
         "end_reason": str(s.get("end_reason"))[:90], "t": round(time.time()-t0, 1)}
    rows.append(r)
    print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"rows": rows, "counters": dict(lore_patches.COUNTERS)}, open(OUT, "w"), indent=2, default=str)
petrified = sum(1 for r in rows if "Petrified" in r["end_reason"])
print("COUNTERS", dict(lore_patches.COUNTERS), flush=True)
print("SUMMARY petrified=", petrified, "/", len(rows), flush=True)
