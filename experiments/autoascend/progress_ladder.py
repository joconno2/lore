"""Progress-ladder evaluation -- the OFFICIAL LORE metric (Jul 10: success =
ascension-progress, not score). Scores a config's game profiles on how far toward
the endgame the agent gets, not how many points it farms. Run per config glob;
compare configs on the ladder. Ladder milestones (ordered), % of games reaching:
  DL5, DL10, DL15, DL20  (depth frontier)
  Mines, Minetown, Sokoban, Solved-Sokoban  (branches / kit route)
  any ascension-kit item  (reflection/bag/wishing/dragon scale/free action/...)
plus median/p90/deepest depth and the starvation rate (the DL1 waste we fixed)."""
import json, glob, sys, re
from statistics import median, mean

pat = sys.argv[1] if len(sys.argv) > 1 else "prof_*.json"
# exact-match by trailing _<digits>.json so prof_* doesn't swallow prof_cr_* etc.
prefix = pat.replace("*.json", "").rstrip("_")
rx = re.compile(re.escape(prefix) + r"_(\d+)\.json$")
R = []
for f in glob.glob(prefix + "_*.json"):
    if rx.search(f):
        try: R.append(json.load(open(f)))
        except Exception: pass
n = len(R)
if not n:
    print("no games for", prefix); sys.exit()

def depth(r): return int(r.get("max_depth") or 1)
def pct(cond): return 100.0 * sum(1 for r in R if cond(r)) / n
def milestone_num(r):
    m = str(r.get("milestone") or "")
    order = ["BE_ON_FIRST_LEVEL", "FIND_GNOMISH_MINES", "FIND_MINETOWN",
             "FIND_SOKOBAN", "SOLVE_SOKOBAN", "FIND_MINES_END", "GO_DOWN"]
    for i, k in enumerate(order):
        if k in m: return i
    return -1

deps = [depth(r) for r in R]
KIT = ("reflection", "bag of holding", "wand of wishing", "dragon scale",
       "ring of free action", "magic marker", "unicorn horn", "amulet of life")
def has_kit(r): return any(any(k in str(it).lower() for k in KIT) for it in (r.get("key_items") or []))

print("=== PROGRESS LADDER  [%s]  n=%d ===" % (prefix, n))
print("depth   median %d  mean %.1f  p90 %d  deepest %d" % (
    median(deps), mean(deps), sorted(deps)[int(.9*n)], max(deps)))
print("reached DL5  %5.0f%%   DL10 %5.0f%%   DL15 %5.0f%%   DL20 %5.0f%%" % (
    pct(lambda r: depth(r) >= 5), pct(lambda r: depth(r) >= 10),
    pct(lambda r: depth(r) >= 15), pct(lambda r: depth(r) >= 20)))
def truly_solved_soko(r):
    # milestone advanced past SOLVE_SOKOBAN AND did NOT abandon (dropped). AA's
    # abandon path also advances the milestone, so milestone alone overcounts.
    return milestone_num(r) >= 5 and (r.get("sokoban_dropped") in (0, None))
# Minetown: PHYSICAL reach if the field is present (new profiles), else the old
# milestone proxy -- which the macro director contaminates by setting FIND_SOKOBAN
# as an objective, so prefer the physical did_minetown wherever we have it.
def reached_minetown(r):
    if "did_minetown" in r: return bool(r.get("did_minetown"))
    return milestone_num(r) >= 3
print("Mines   %5.0f%%   Minetown %5.0f%%   Sokoban %5.0f%%   SOLVED-Soko %5.0f%%  (abandoned-past %5.0f%%)" % (
    pct(lambda r: r.get("did_mines")), pct(reached_minetown),
    pct(lambda r: r.get("did_sokoban")), pct(truly_solved_soko),
    pct(lambda r: milestone_num(r) >= 5 and r.get("sokoban_dropped"))))
print("ascension-kit item held  %5.0f%%   |   starvation deaths %5.0f%%" % (
    pct(has_kit), pct(lambda r: r.get("death_cat") == "starvation")))
