"""Diagnose WHY frozen AutoAscend sticks at DL1 on ~half of NetHackChallenge-v0
seeds. Run one seed, capture: rolled character, whether the downstair glyph
ever enters the agent's map, whether AA ever issues '>' (descend), the distinct
action/message loop it falls into, and where it dies. No LORE patches -- pure
frozen AA via its own EnvWrapper.main()."""
import sys, json, gym, nle, time, os, collections
import numpy as np

seed = int(sys.argv[1]); OUT = sys.argv[2]
STEP_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 40000

env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass

STATE = {"role": None, "saw_downstair": 0, "downstair_first_turn": None,
         "descend_cmd": 0, "descend_turn": [], "max_depth": 1, "steps": 0,
         "last_depth_seen": 1}
_MSGS = collections.Counter()
_ACTS = collections.Counter()
# '>' in NLE action space is Command.DOWN. find its index.
from nle import nethack as NH
DOWN = None
try:
    for i, a in enumerate(env.unwrapped.actions):
        if int(a) == ord('>'): DOWN = i
except Exception: pass

_orig = env.step
def _hook(a):
    r = _orig(a)
    STATE["steps"] += 1
    try:
        # action a is an index into env.actions; map to char
        try:
            ch = int(env.unwrapped.actions[int(a)])
            _ACTS[chr(ch) if 32 <= ch < 127 else ch] += 1
            if ch == ord('>'):
                STATE["descend_cmd"] += 1
        except Exception: pass
        o = r[0]
        m = bytes(o["message"]).decode("latin1").strip("\x00").strip()
        if m: _MSGS[m[:50]] += 1
        bl = o["blstats"]; dep = int(bl[12]); t = int(bl[20])
        STATE["last_depth_seen"] = dep
        if dep > STATE["max_depth"]:
            STATE["max_depth"] = dep
            if STATE["descend_turn"] is not None:
                STATE["descend_turn"].append((t, dep))
        # downstair glyph in the visible map? NLE glyph for '>' downstair.
        # cmap S_dnstair; detect via tty char '>' anywhere on map rows.
        tc = o["tty_chars"]
        rows = [bytes(row).decode("latin1") for row in tc[1:22]]
        if any('>' in row for row in rows):
            STATE["saw_downstair"] += 1
            if STATE["downstair_first_turn"] is None:
                STATE["downstair_first_turn"] = t
    except Exception: pass
    return r
env.step = _hook

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
# capture rolled role from the agent's character once available
t0 = time.time()
try:
    w.main()
except BaseException as e:
    STATE["end_err"] = repr(e)[:90]
try:
    ag = w.agent
    STATE["role"] = str(getattr(getattr(ag, "character", None), "role", None))
    STATE["self_glyph"] = None
except Exception: pass
s = w.get_summary()
rec = {"seed": seed, "max_depth": STATE["max_depth"], "steps": STATE["steps"],
       "turns": s.get("turns"), "score": s.get("score"), "xl": s.get("experience_level"),
       "end_reason": str(s.get("end_reason"))[:120], "role": STATE["role"],
       "saw_downstair_steps": STATE["saw_downstair"],
       "downstair_first_turn": STATE["downstair_first_turn"],
       "descend_cmd_count": STATE["descend_cmd"],
       "descend_events": STATE["descend_turn"],
       "top_msgs": _MSGS.most_common(15),
       "top_acts": _ACTS.most_common(15),
       "t": round(time.time() - t0)}
json.dump(rec, open(OUT, "w"), default=str)
print("DONE seed", seed, "depth", STATE["max_depth"], "sawdown", STATE["saw_downstair"],
      "descend_cmd", STATE["descend_cmd"], flush=True)
