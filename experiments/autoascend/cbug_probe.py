"""Confirm the DL1-stick root cause: empty inventory container with
container_id=None -> check_container_content asserts -> AgentPanic (swallowed)
-> content never assigned -> re-checked every update forever. Instrument (no
fix): log each check_container_content call's item text, container_id, whether
content got assigned, and any panic. Run one stuck seed, short cap."""
import sys, json, gym, nle, os, collections
seed = int(sys.argv[1]); OUT = sys.argv[2]; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

from autoascend.item import inventory as INV
from autoascend.exceptions import AgentPanic

LOG = collections.Counter()
SAMPLES = []
_orig = INV.Inventory.check_container_content
def _wrapped(self, item):
    txt = None; cid = "?"
    try:
        txt = str(getattr(item, "text", None))[:40]
        cid = getattr(item, "container_id", "NOATTR")
    except Exception: pass
    before = getattr(item, "content", "ERR")
    key = (txt, str(cid), "content_was_None=%s" % (before is None))
    LOG[key] += 1
    err = None
    try:
        r = _orig(self, item)
    except AgentPanic as e:
        err = str(e)[:50]
        LOG[("PANIC", err)] += 1
        raise
    finally:
        after = getattr(item, "content", "ERR")
        if len(SAMPLES) < 25:
            SAMPLES.append({"item": txt, "container_id": str(cid),
                            "content_before_None": before is None,
                            "content_after_None": (after is None),
                            "panic": err})
    return r
INV.Inventory.check_container_content = _wrapped

env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
STEPS = [0]; DEPTH = [1]
_os = env.step
def _h(a):
    r = _os(a); STEPS[0] += 1
    try:
        d = int(r[0]["blstats"][12]); DEPTH[0] = max(DEPTH[0], d)
    except Exception: pass
    if STEPS[0] > CAP:
        raise KeyboardInterrupt("cap")
    return r
env.step = _h

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: pass
rec = {"seed": seed, "steps": STEPS[0], "max_depth": DEPTH[0],
       "check_calls": sum(v for k, v in LOG.items() if k[0] not in ("PANIC",)),
       "top_check_keys": [[list(k), v] for k, v in LOG.most_common(12)],
       "samples": SAMPLES}
json.dump(rec, open(OUT, "w"), default=str)
print("DONE seed", seed, "depth", DEPTH[0], "checkcalls", rec["check_calls"], flush=True)
