"""Head-to-head: base AutoAscend vs AA + LLM food-economy oracle, FULL real games
(DL1, no wizard help). The bar-clearing test -- beat the SOTA on score in fair
play, with the LLM making the food/prayer calls. mode = base|mock|llm."""
import sys, json, gym, nle, lore_patches, time, os
seed = int(sys.argv[1]); mode = sys.argv[2]; OUT = sys.argv[3]

if mode == "mock":
    lore_patches.apply_food_oracle(mock=True)
elif mode == "llm":
    lore_patches.apply_food_oracle(mock=False)

from autoascend.agent import Agent as _Ag
_orig_pray = _Ag.pray
_PRAY = {"n": 0}
def _cpray(self):
    _PRAY["n"] += 1
    return _orig_pray(self)
_Ag.pray = _cpray

env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
_MSGS = []
_orig = env.step
def _hook(a):
    r = _orig(a)
    try:
        m = bytes(r[0]["message"]).decode("latin1").strip("\x00").strip()
        if m and (not _MSGS or _MSGS[-1] != m):
            _MSGS.append(m)
            if len(_MSGS) > 30: _MSGS.pop(0)
    except Exception: pass
    return r
env.step = _hook

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
t0 = time.time()
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:80]
s = w.get_summary(); C = lore_patches.COUNTERS

tail = " || ".join(_MSGS[-8:]); low = tail.lower()
cat = "other"
if "faint" in low or "starv" in low or "lack of food" in low: cat = "starvation"
elif "turn to stone" in low or "petrif" in low: cat = "petrification"
elif "drown" in low: cat = "drowning"
elif "killed by" in low or "you die" in low: cat = "combat/other-death"

json.dump({"seed": seed, "mode": mode, "score": s.get("score"), "turns": s.get("turns"),
           "depth": s.get("level_num"), "xl": s.get("experience_level"),
           "death_cat": cat, "prayers": _PRAY["n"],
           "food_query": C.get("food_query"), "food_EAT": C.get("food_EAT"),
           "food_PRAY": C.get("food_PRAY"), "end_reason": str(s.get("end_reason"))[:70],
           "t": round(time.time() - t0)}, open(OUT, "w"), default=str)
print("DONE", flush=True)
