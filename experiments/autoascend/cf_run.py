"""Counterfactual replay: measure the veto's CAUSAL effect at the single
petrification-risk decision, free of whole-game RNG-divergence confound.
mode=base -> crash_recovery only. mode=cf -> crash_recovery + veto-ONCE (fires
the first petrif-risk melee, then disabled). NLE determinism => base and cf are
byte-identical until that one decision, so (cf score - base score) on seeds where
the veto fired is the veto's causal value at the branch. base|cf, LLM via env."""
import sys, json, gym, nle, lore_patches, time, os
seed = int(sys.argv[1]); mode = sys.argv[2]; OUT = sys.argv[3]

lore_patches.apply_crash_recovery()
if mode == "cf":
    os.environ["LORE_VETO_ONCE"] = "1"
    lore_patches.apply_oracle_veto(mock=os.environ.get("LORE_CF_LLM") != "1")

env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
_MSGS = []; _md = [1]
_orig = env.step
def _hook(a):
    r = _orig(a)
    try:
        o = r[0]
        m = bytes(o["message"]).decode("latin1").strip("\x00").strip()
        if m and (not _MSGS or _MSGS[-1] != m):
            _MSGS.append(m)
            if len(_MSGS) > 12: _MSGS.pop(0)
        d = int(o["blstats"][12])
        if d > _md[0]: _md[0] = d
    except Exception: pass
    return r
env.step = _hook

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
t0 = time.time()
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:80]
s = w.get_summary(); C = lore_patches.COUNTERS
low = (str(s.get("end_reason") or w.__dict__.get("end_reason") or "") + " " + " ".join(_MSGS[-5:])).lower()
def cat():
    if "lack of food" in low or "starv" in low: return "starvation"
    if "turn to stone" in low or "petrif" in low: return "petrification"
    if "frozen" in low or "paralys" in low: return "paralysis"
    if "killed by" in low or "you die" in low: return "combat"
    if "assert" in low or "recursion" in low or "runtime" in low: return "crash"
    return "other"
json.dump({"seed": seed, "mode": mode, "score": s.get("score"), "turns": s.get("turns"),
           "max_depth": _md[0], "death_cat": cat(),
           "veto_fired": C.get("veto_fired", 0), "veto_fire_turn": C.get("veto_fire_turn"),
           "veto_fire_mon": C.get("veto_fire_mon"), "veto_fire_action": C.get("veto_fire_action"),
           "t": round(time.time() - t0)}, open(OUT, "w"), default=str)
print("DONE", flush=True)
