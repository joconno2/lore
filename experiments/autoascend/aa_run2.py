"""Run AutoAscend with both LORE interventions (sokoban crash patch + oracle
melee veto). Args: seeds(csv) out mock(1/0)."""
import sys, json, time
import gym
import lore_patches

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [105, 107, 145]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/aa_veto.json"
MOCK = (sys.argv[3] != "0") if len(sys.argv) > 3 else True

print("sokoban patch:", lore_patches.apply(), flush=True)
print("oracle veto:", lore_patches.apply_oracle_veto(mock=MOCK), flush=True)

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
    vetoes = {k: v for k, v in s.items() if "lore_veto" in str(k)}
    r = {"seed": seed, "score": s.get("score"), "level_num": s.get("level_num"),
         "xl": s.get("experience_level"), "turns": s.get("turns"),
         "end_reason": str(s.get("end_reason"))[:90], "vetoes": vetoes,
         "t": round(time.time()-t0, 1)}
    rows.append(r)
    print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"mock": MOCK, "rows": rows}, open(OUT, "w"), indent=2, default=str)
petrified = sum(1 for r in rows if "Petrified" in r["end_reason"])
print("SUMMARY mock=", MOCK, "petrified=", petrified, "/", len(rows), flush=True)
