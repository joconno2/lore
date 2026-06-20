"""Headless AutoAscend benchmark. No Ray, no X11. Runs N episodes, dumps summaries."""
import sys, json, time
import numpy as np
import gym
from autoascend.env_wrapper import EnvWrapper

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BASE_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
OUT = sys.argv[3] if len(sys.argv) > 3 else "/aa_bench.json"

rows = []
for i in range(N):
    seed = BASE_SEED + i
    t0 = time.time()
    env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                     agent_args=dict(panic_on_errors=False, verbose=False))
    env.env.seed(seed, seed)
    try:
        env.main()
    except BaseException as e:
        env.end_reason = f"exception: {repr(e)[:160]}"
    s = env.get_summary()
    s["duration"] = round(time.time() - t0, 1)
    s["req_seed"] = seed
    rows.append(s)
    print(i, {k: s.get(k) for k in ("score", "turns", "level_num", "experience_level", "milestone", "end_reason")}, flush=True)
    try: env.env.close()
    except: pass

scores = [r["score"] for r in rows]
out = {"rows": rows, "n": len(rows),
       "mean": float(np.mean(scores)), "median": float(np.median(scores)),
       "max": int(np.max(scores)), "min": int(np.min(scores)), "std": float(np.std(scores))}
depths = [r.get("level_num", 0) for r in rows]
out["depth_dist"] = {str(d): depths.count(d) for d in sorted(set(depths))}
json.dump(out, open(OUT, "w"), indent=2, default=str)
print("SUMMARY", {k: out[k] for k in ("n", "mean", "median", "max", "min", "depth_dist")}, flush=True)
