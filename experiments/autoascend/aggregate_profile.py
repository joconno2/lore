"""Aggregate the per-game AA profiles into a macro-strategy picture: outcome
distribution, dungeon/branch completion rates, depth reached, ascension-kit
acquisition, death taxonomy, dawdling, crash rate. This is the empirical half of
the macro gap analysis (vs expert play)."""
import json, glob, sys
from statistics import median, mean
from collections import Counter

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "prof_*.json"))
R = []
for f in files:
    try: R.append(json.load(open(f)))
    except Exception: pass
n = len(R)
def num(x, d=0):
    try: return float(x)
    except Exception: return d
sc = sorted(num(r.get("score")) for r in R)
dep = sorted(int(num(r.get("max_depth"))) for r in R)
xl = sorted(int(num(r.get("xl"))) for r in R)
print("=== AA PROFILE  n=%d ===" % n)
print("score   mean %.0f median %.0f  p90 %.0f  max %.0f" % (
    mean(sc), median(sc), sc[int(.9*n)] if n else 0, max(sc) if sc else 0))
print("max_depth mean %.1f median %d  p90 %d  max %d" % (
    mean(dep), median(dep), dep[int(.9*n)] if n else 0, max(dep) if dep else 0))
print("xl      mean %.1f median %d  max %d" % (mean(xl), median(xl), max(xl) if xl else 0))
print()
print("--- depth histogram (max dlvl reached) ---")
dh = Counter(dep)
for d in sorted(dh): print("  DL%2d: %s (%d)" % (d, "#" * dh[d], dh[d]))
print()
print("--- branch completion ---")
for k in ("did_mines", "did_sokoban", "did_quest"):
    c = sum(1 for r in R if r.get(k)); print("  %-12s %d/%d (%.0f%%)" % (k, c, n, 100*c/max(1,n)))
# max depth in each branch
for bk in ("Mines", "Sokoban", "Quest"):
    mx = [int(r["branches"][bk]["max"]) for r in R if r.get("branches", {}).get(bk)]
    if mx: print("  %-12s reached: median max-level %d (n=%d)" % (bk, median(mx), len(mx)))
print()
print("--- death taxonomy ---")
dc = Counter(r.get("death_cat", "?") for r in R)
for k, v in dc.most_common(): print("  %-14s %d (%.0f%%)" % (k, v, 100*v/max(1,n)))
fa = sum(1 for r in R if r.get("fighting_at_death")); print("  fighting_at_death: %d/%d (%.0f%%)" % (fa, n, 100*fa/max(1,n)))
print()
print("--- ascension-kit / key items acquired (ever, in inv at death) ---")
ki = Counter()
for r in R:
    for it in (r.get("key_items") or []): ki[it] += 1
if not ki: print("  NONE acquired in any game")
for k, v in ki.most_common(): print("  %-22s %d/%d (%.0f%%)" % (k, v, n, 100*v/max(1,n)))
print()
print("--- resource / behavior ---")
pr = sorted(int(num(r.get("prayers"))) for r in R)
mt = sorted(int(num(r.get("max_turns_on_position"))) for r in R)
print("  prayers/game: median %d  max %d" % (median(pr), max(pr) if pr else 0))
print("  max_turns_on_one_tile: median %d  p90 %d  max %d (dawdling proxy)" % (
    median(mt), mt[int(.9*n)] if n else 0, max(mt) if mt else 0))
