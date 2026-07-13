"""Config A/B for depth. arg3 = config: min | nostick | engine.
Tests whether apply_unstick_dl1 (leave DL1 underleveled to dodge starvation) helps or
hurts depth vs keeping AA's own XL8 DL1 gate. Same per-seed capture as asc_diag."""
import sys, json, time, re
import gym
import lore_patches

CONFIG = sys.argv[3] if len(sys.argv) > 3 else "engine"
applied = lore_patches.apply()                       # safe_solve_sokoban (all configs)
STAB = ("apply_crash_recovery", "apply_petrifier_avoidance", "apply_decode_hardening",
        "apply_obs_sanitize")
if CONFIG == "min":
    pass
elif CONFIG == "nostick":
    for fn in STAB + ("apply_anti_starvation",):
        f = getattr(lore_patches, fn, None)
        if f:
            try: f(); applied.append(fn)
            except Exception as e: applied.append(f"{fn}:ERR")
elif CONFIG == "engine":
    for fn in ("apply_unstick_dl1",) + STAB + ("apply_anti_starvation",):
        f = getattr(lore_patches, fn, None)
        if f:
            try: f(); applied.append(fn)
            except Exception as e: applied.append(f"{fn}:ERR")
print(f"CONFIG={CONFIG} applied:", applied, flush=True)

from autoascend.env_wrapper import EnvWrapper
seeds = [int(s) for s in sys.argv[1].split(",")]
OUT = sys.argv[2]

def killer(reason):
    m = re.search(r"[Kk]illed by (?:an? )?([a-z ]+?)(?:,|\.|$)", reason or "")
    if m: return m.group(1).strip()
    for k in ("starv", "poison", "petrif", "AssertionError", "trap", "drown"):
        if k.lower() in (reason or "").lower(): return k
    return (reason or "?")[:30]

rows = []
for seed in seeds:
    t0 = time.time()
    env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                     agent_args=dict(panic_on_errors=False, verbose=False))
    env.env.seed(seed, seed)
    xl = None
    try: env.main()
    except BaseException as e: env.end_reason = f"harness_exc: {repr(e)[:120]}"
    try: xl = int(env.agent.blstats.experience_level)
    except Exception: pass
    s = env.get_summary(); er = str(s.get("end_reason"))
    r = {"seed": seed, "cfg": CONFIG, "level_num": s.get("level_num"),
         "xl": xl if xl is not None else s.get("experience_level"),
         "milestone": str(s.get("milestone")).replace("Milestone.", ""),
         "killer": killer(er), "t": round(time.time()-t0, 1)}
    rows.append(r); print(r, flush=True)
    try: env.env.close()
    except Exception: pass

json.dump({"cfg": CONFIG, "applied": applied, "rows": rows}, open(OUT, "w"), indent=2, default=str)
print("DONE", CONFIG, len(rows), flush=True)
