"""Smoke: AutoAscend + sokoban patch + lore_planner executor (StockpileFood goal).
Verifies the planning seam drives behavior without breaking the agent."""
import sys, json, time
import gym
import lore_patches, lore_planner

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [100, 107]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/aa_plan.json"

print("sokoban:", lore_patches.apply(), flush=True)
print("planner:", lore_planner.install(), flush=True)

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
         "end_reason": str(s.get("end_reason"))[:80], "t": round(time.time()-t0, 1)}
    rows.append(r)
    print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"rows": rows, "counters": dict(lore_patches.COUNTERS)}, open(OUT, "w"), indent=2, default=str)
print("COUNTERS", dict(lore_patches.COUNTERS), flush=True)
