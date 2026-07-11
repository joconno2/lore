"""Container-recheck fix v3 (cache-key normalization + empty guard). Root cause:
an inventory container's content is stored under a NUMERIC container_id on first
check (item_manager._get_new_container_identifier returns int) but re-looked-up
under the parsed comment STRING '0' on every rebuild (get_item_from_text:
identifier = item.comment). int 0 != str '0' -> cache miss -> content stays None
-> is_possible_container() True -> the empty sack is re-opened every update
(wastes turns, adds starvation pressure).

Fix: (a) make _get_new_container_identifier return a STRING so store/lookup keys
match; (b) belt-and-suspenders: skip re-opening a container whose text already
says 'empty' (NetHack's own known-empty label). Measure: does the wasted-reopen
count ('The sack is empty' / check_container_content calls) drop to ~O(1)?"""
import sys, json, gym, nle, collections
from autoascend.item import item_manager as IM
from autoascend.item import inventory as INV
from autoascend.item import ContainerContent

# (a) normalize new container ids to strings so they match the parsed '#N' comment
_orig_id = IM.ItemManager._get_new_container_identifier
def _str_id(self):
    return "auto:" + str(_orig_id(self))
IM.ItemManager._get_new_container_identifier = _str_id

# (b) don't re-open a container NetHack already labels 'empty'
_orig_check = INV.Inventory.check_container_content
_CHK = collections.Counter()
def _guarded(self, item):
    txt = getattr(item, "text", None) or ""
    _CHK[txt[:30]] += 1
    if "empty" in txt:
        if getattr(item, "content", None) is None:
            item.content = ContainerContent()
        return
    return _orig_check(self, item)
INV.Inventory.check_container_content = _guarded

seed = int(sys.argv[1]); OUT = sys.argv[2]; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
STEPS = [0]; DEPTH = [1]; MSG = collections.Counter()
_os = env.step
def _h(a):
    r = _os(a); STEPS[0] += 1
    try: DEPTH[0] = max(DEPTH[0], int(r[0]["blstats"][12]))
    except Exception: pass
    try:
        m = bytes(r[0]["message"]).decode("latin1").strip("\x00").strip()
        if m: MSG[m[:40]] += 1
    except Exception: pass
    if STEPS[0] > CAP: raise KeyboardInterrupt("cap")
    return r
env.step = _h
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException: pass
json.dump({"seed": seed, "steps": STEPS[0], "max_depth": DEPTH[0],
           "container_checks": sum(_CHK.values()), "top_checks": _CHK.most_common(6),
           "sack_empty_msgs": sum(v for k, v in MSG.items() if "empty" in k.lower()),
           "top_msgs": MSG.most_common(8)}, open(OUT, "w"), default=str)
print("DONE seed", seed, "depth", DEPTH[0], "checks", sum(_CHK.values()), flush=True)
