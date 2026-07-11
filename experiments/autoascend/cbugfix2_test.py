"""Container-recheck fix v2 (the real one). Root cause: an inventory container's
content is cached under a NUMERIC container_id on first check but re-looked-up
under the parsed '#N' comment STRING on every rebuild (item_manager.py:196-206).
'#0' (str) != 0 (int) -> cache miss -> content stays None -> is_possible_container
stays True -> the empty sack is re-opened every update -> applying it perturbs
inv_strs -> the inventory update never completes -> AA never descends. ~50% of
seeds stick at DL1.

Fix: NetHack itself labels a known-empty container 'empty' in the item text. Guard
check_container_content: if the text says 'empty', set an empty ContainerContent
(so is_possible_container() stops firing) and skip the re-open interaction (which
is what perturbs inv_strs). Independent of the id-mismatch. use_container's own
refresh path still works: a non-empty container's text won't contain 'empty'."""
import sys, json, gym, nle
from autoascend.item import inventory as INV
from autoascend.item import ContainerContent

_orig = INV.Inventory.check_container_content
def _guarded(self, item):
    try:
        txt = getattr(item, "text", None) or ""
    except Exception:
        txt = ""
    if "empty" in txt:
        if getattr(item, "content", None) is None:
            item.content = ContainerContent()  # mark known-empty; stops is_possible_container
        return  # don't re-open: NetHack already told us it's empty
    return _orig(self, item)
INV.Inventory.check_container_content = _guarded

seed = int(sys.argv[1]); OUT = sys.argv[2]; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
STEPS = [0]; DEPTH = [1]
_os = env.step
def _h(a):
    r = _os(a); STEPS[0] += 1
    try: DEPTH[0] = max(DEPTH[0], int(r[0]["blstats"][12]))
    except Exception: pass
    if STEPS[0] > CAP:
        raise KeyboardInterrupt("cap")
    return r
env.step = _h
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException: pass
s = w.get_summary()
json.dump({"seed": seed, "steps": STEPS[0], "max_depth": DEPTH[0], "turns": s.get("turns"),
           "xl": s.get("experience_level"), "end_reason": str(s.get("end_reason"))[:120]},
          open(OUT, "w"), default=str)
print("DONE seed", seed, "depth", DEPTH[0], "steps", STEPS[0], flush=True)
