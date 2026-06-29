"""Paired base vs unstick-DL1, same seeds. Does descending earlier (food-aware)
cut starvation and lift score? The validation of AA's #1 macro-flaw lever."""
import json, glob
from statistics import median, mean
def load(pat, strip):
    r = {}
    for f in glob.glob(pat):
        try:
            d = json.load(open(f)); r[int(d["seed"])] = d
        except Exception: pass
    return r
base = load("prof_1[0-9][0-9].json", "prof_")
unst = load("prof_u_*.json", "prof_u_")
c = sorted(set(base) & set(unst))
def num(d, k):
    try: return float(d.get(k) or 0)
    except Exception: return 0.0
print("paired n=%d" % len(c))
for nm, R in (("base", base), ("unstick", unst)):
    sc = [num(R[s], "score") for s in c]
    dp = [int(num(R[s], "max_depth")) for s in c]
    starv = sum(1 for s in c if R[s].get("death_cat") == "starvation")
    mines = sum(1 for s in c if R[s].get("did_mines"))
    soko = sum(1 for s in c if R[s].get("did_sokoban"))
    print("  %-8s score mean %6.0f median %6.0f | depth med %2d p90 %2d | starv %d%% | mines %d%% soko %d%%" % (
        nm, mean(sc), median(sc), median(dp), sorted(dp)[int(.9*len(dp))], 100*starv//len(c), 100*mines//len(c), 100*soko//len(c)))
w = sum(num(unst[s], "score") > num(base[s], "score") for s in c)
l = sum(num(unst[s], "score") < num(base[s], "score") for s in c)
print("paired score: unstick wins %d / losses %d / ties %d" % (w, l, len(c)-w-l))
# where unstick helped most
deltas = sorted(((num(unst[s], "score") - num(base[s], "score"), s) for s in c), reverse=True)
print("top gains:", [(s, int(d)) for d, s in deltas[:5]])
print("top losses:", [(s, int(d)) for d, s in deltas[-5:]])
# starvation conversion: of base-starvation games, what happened with unstick?
bs = [s for s in c if base[s].get("death_cat") == "starvation"]
print("base-starvation games: %d -> unstick outcomes:" % len(bs))
from collections import Counter
print("  ", dict(Counter(unst[s].get("death_cat") for s in bs)))
print("  base-starv depth med %d -> unstick depth med %d" % (
    median([int(num(base[s],"max_depth")) for s in bs]) if bs else 0,
    median([int(num(unst[s],"max_depth")) for s in bs]) if bs else 0))
print("  base-starv score mean %d -> unstick score mean %d" % (
    mean([num(base[s],"score") for s in bs]) if bs else 0,
    mean([num(unst[s],"score") for s in bs]) if bs else 0))
