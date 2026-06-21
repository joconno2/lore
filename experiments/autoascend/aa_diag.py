"""Diagnose the petrification path: record the last actions/messages and
cockatrice proximity before the agent dies. One seed, verbose ring buffer."""
import sys, json
import gym
import nle.nethack as nh
from autoascend import agent as agent_lib
from autoascend.env_wrapper import EnvWrapper

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 107

ring = []  # (turn, action, message, hp, nearest_cockatrice_dist)

from autoascend.agent import Agent
orig_step = Agent.step

def traced_step(self, action, *a, **k):
    r = orig_step(self, action, *a, **k)
    try:
        bl = self.blstats
        cdist = None
        for m in self.get_visible_monsters():
            _, y, x, mon, _ = m
            if "cockatrice" in (mon.mname or "").lower() or "chickatrice" in (mon.mname or "").lower():
                d = max(abs(y - bl.y), abs(x - bl.x))
                cdist = d if cdist is None else min(cdist, d)
        msg = ""
        try:
            msg = bytes(self.last_observation["message"]).decode("latin-1").strip("\x00").strip()
        except Exception:
            pass
        if cdist is not None or "stone" in msg.lower() or "cockatrice" in msg.lower() or "Elbereth" in msg:
            ring.append({"turn": int(bl.time), "action": str(action), "hp": int(bl.hitpoints),
                         "cockatrice_dist": cdist, "msg": msg[:70]})
            if len(ring) > 60:
                ring.pop(0)
    except Exception:
        pass
    return r

Agent.step = traced_step

env = EnvWrapper(gym.make("NetHackChallenge-v0", no_progress_timeout=1000),
                 agent_args=dict(panic_on_errors=False, verbose=False))
env.env.seed(SEED, SEED)
try:
    env.main()
except BaseException as e:
    env.end_reason = f"exc: {repr(e)[:120]}"
s = env.get_summary()
print("END:", s.get("end_reason"), "score", s.get("score"), "DL", s.get("level_num"), flush=True)
print("=== last cockatrice-proximity events ===", flush=True)
for e in ring[-30:]:
    print(e, flush=True)
json.dump({"seed": SEED, "end": str(s.get("end_reason")), "ring": ring}, open("/workspace/aa_diag.json", "w"), default=str)
