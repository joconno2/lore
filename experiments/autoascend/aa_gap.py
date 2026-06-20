"""AutoAscend gap-analysis harness. Runs N episodes, captures panic reasons,
end reasons, milestone reached, and depth. Finds where the frozen base is weak
so the LORE oracle can target those decision points. Frozen base: no edits to
autoascend, only introspection after each episode."""
import sys, json, time
from collections import Counter
import numpy as np
import gym
from autoascend.env_wrapper import EnvWrapper

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BASE_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
OUT = sys.argv[3] if len(sys.argv) > 3 else "/workspace/aa_gap.json"

rows = []
all_panics = Counter()
end_reasons = Counter()

for i in range(N):
    seed = BASE_SEED + i
    t0 = time.time()
    env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                     agent_args=dict(panic_on_errors=False, verbose=False))
    env.env.seed(seed, seed)
    try:
        env.main()
    except BaseException as e:
        env.end_reason = f"harness_exc: {repr(e)[:160]}"

    # panic reasons (the decision points AutoAscend stumbles on)
    panics = []
    for p in getattr(env.agent, "all_panics", []):
        try:
            panics.append(str(p.args[0]) if getattr(p, "args", None) else type(p).__name__)
        except Exception:
            panics.append(type(p).__name__)
    pr = Counter(panics)
    all_panics.update(pr)

    s = env.get_summary()
    # normalize end_reason to a coarse bucket (strip seed-specific detail)
    er = (s.get("end_reason") or "").strip()
    bucket = er.split(",")[0].split("(")[0][:60] if er else "unknown"
    end_reasons[bucket] += 1

    s["duration"] = round(time.time() - t0, 1)
    s["req_seed"] = seed
    s["panic_reasons"] = dict(pr)
    s["panic_total"] = len(panics)
    rows.append(s)
    print(i, {k: s.get(k) for k in ("score", "turns", "level_num", "experience_level",
              "milestone", "panic_total", "end_reason")}, flush=True)
    try: env.env.close()
    except Exception: pass

scores = [r["score"] for r in rows]
depths = [r.get("level_num", 0) for r in rows]
milestones = [r.get("milestone", -1) for r in rows]
out = {
    "n": len(rows),
    "mean": float(np.mean(scores)), "median": float(np.median(scores)),
    "max": int(np.max(scores)), "min": int(np.min(scores)), "std": float(np.std(scores)),
    "depth_dist": {str(d): depths.count(d) for d in sorted(set(depths))},
    "milestone_dist": {str(m): milestones.count(m) for m in sorted(set(milestones))},
    "end_reasons": dict(end_reasons.most_common()),
    "top_panics": dict(all_panics.most_common(25)),
    "panic_total_all": sum(all_panics.values()),
    "rows": rows,
}
json.dump(out, open(OUT, "w"), indent=2, default=str)
print("SUMMARY", {k: out[k] for k in ("n", "mean", "median", "max", "depth_dist",
      "milestone_dist", "panic_total_all")}, flush=True)
print("TOP_PANICS", out["top_panics"], flush=True)
print("END_REASONS", out["end_reasons"], flush=True)
