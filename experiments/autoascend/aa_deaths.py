"""Foundational death taxonomy (the 'pro who wouldn't die' program, phase 0).
Characterize HOW base AutoAscend dies and which deaths are AVOIDABLE -- the
empirical bedrock every later survival intervention is measured against. Per
game: full killer, HP trajectory (sudden vs slow bleed), whether it was fighting
at death, prayer/food state, depth/xl/turns/score. Avoidability is assessed
downstream from these signals."""
import sys, json, gym, nle, time
seed = int(sys.argv[1]); OUT = sys.argv[2]

from autoascend.agent import Agent as _Ag
_orig_pray = _Ag.pray
_PRAY = {"n": 0, "turns": []}
def _cpray(self):
    _PRAY["n"] += 1
    try: _PRAY["turns"].append(int(self.blstats.time))
    except Exception: pass
    return _orig_pray(self)
_Ag.pray = _cpray

env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass

_MSGS = []        # rolling recent messages
_HP = []          # (turn, hp, maxhp, hunger) sampled
_orig = env.step
def _hook(a):
    r = _orig(a)
    try:
        o = r[0]
        m = bytes(o["message"]).decode("latin1").strip("\x00").strip()
        if m and (not _MSGS or _MSGS[-1] != m):
            _MSGS.append(m)
            if len(_MSGS) > 25: _MSGS.pop(0)
        bl = o["blstats"]
        t, hp, mhp, hung = int(bl[20]), int(bl[10]), int(bl[11]), int(bl[21])
        if not _HP or t - _HP[-1][0] >= 200 or (hp / max(1, mhp)) < 0.3:
            _HP.append((t, hp, mhp, hung))
            if len(_HP) > 60: _HP.pop(0)
    except Exception: pass
    return r
env.step = _hook

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
t0 = time.time()
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:90]
s = w.get_summary()
end = str(s.get("end_reason") or w.__dict__.get("end_reason") or "")

# parse killer
killer = ""
low = end.lower()
if "killed by" in low:
    killer = end[low.find("killed by") + 10:].split(",")[0].split(".")[0].strip()

# fine category
def categorize():
    e = low + " " + " ".join(_MSGS[-6:]).lower()
    if "lack of food" in e or "starv" in e or ("faint" in e and "food" in e): return "starvation"
    if "turn to stone" in e or "petrif" in e or "solidif" in e: return "petrification"
    if "frozen" in e or "paralys" in e: return "paralysis"
    if "drown" in e: return "drowning"
    if "while praying" in e or "smote" in e or "wrath" in e: return "prayer"
    if "killed by" in low or "you die" in low: return "combat/other"
    if "assert" in low or "panic" in low or "runtime" in low: return "crash"
    return "other"
cat = categorize()

# bleed vs sudden: lowest hp_frac in the last ~2000 turns before death
tail_hp = [hp / max(1, mhp) for (t, hp, mhp, hung) in _HP[-12:]]
min_recent = min(tail_hp) if tail_hp else None
# was it fighting at death? combat verbs in the last messages
fighting = any(any(v in m.lower() for v in ("hits", "bites", "you hit", "you miss", "swings", "casts"))
               for m in _MSGS[-4:])

json.dump({"seed": seed, "score": s.get("score"), "turns": s.get("turns"),
           "depth": s.get("level_num"), "xl": s.get("experience_level"),
           "death_cat": cat, "killer": killer, "end_reason": end[:120],
           "fighting_at_death": fighting, "min_hp_frac_recent": round(min_recent, 2) if min_recent is not None else None,
           "prayers": _PRAY["n"], "last_pray": _PRAY["turns"][-1] if _PRAY["turns"] else None,
           "hp_traj": _HP[-12:], "msgs_tail": " || ".join(_MSGS[-6:])[:300],
           "t": round(time.time() - t0)}, open(OUT, "w"), default=str)
print("DONE", flush=True)
