"""Test panic_on_errors=True on the seeds that crashed with AssertionError.
Does AutoAscend's built-in error-recovery mode save the deep runs?"""
import sys, json, time
import gym
from autoascend.env_wrapper import EnvWrapper

SEEDS = [126, 135, 137, 138, 141]
PANIC = sys.argv[1] == "1" if len(sys.argv) > 1 else True
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/aa_crashtest.json"

rows = []
for seed in SEEDS:
    t0 = time.time()
    env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                     agent_args=dict(panic_on_errors=PANIC, verbose=False))
    env.env.seed(seed, seed)
    try:
        env.main()
    except BaseException as e:
        env.end_reason = f"harness_exc: {repr(e)[:160]}"
    s = env.get_summary()
    r = {"seed": seed, "score": s.get("score"), "level_num": s.get("level_num"),
         "xl": s.get("experience_level"), "panic_total": len(getattr(env.agent, "all_panics", [])),
         "end_reason": str(s.get("end_reason"))[:90], "t": round(time.time()-t0, 1)}
    rows.append(r)
    print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"panic_on_errors": PANIC, "rows": rows}, open(OUT, "w"), indent=2, default=str)
crashes = sum(1 for r in rows if "AssertionError" in r["end_reason"])
print("SUMMARY panic_on_errors=", PANIC, "crashes=", crashes, "/", len(rows), flush=True)
