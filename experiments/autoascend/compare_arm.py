"""Compare base vs an intervention arm, paired by seed. Usage: compare_arm.py <variant_glob> <label>"""
import json, glob, sys
from statistics import median, mean
from collections import Counter
def load(pat):
    r = {}
    for f in glob.glob(pat):
        try: d = json.load(open(f)); r[int(d["seed"])] = d
        except Exception: pass
    return r
base = load("prof_1[0-9][0-9].json")
var = load(sys.argv[1]); label = sys.argv[2] if len(sys.argv) > 2 else "arm"
c = sorted(set(base) & set(var))
def num(d, k):
    try: return float(d.get(k) or 0)
    except Exception: return 0.0
print("=== base vs %s  (paired n=%d) ===" % (label, len(c)))
for nm, R in (("base", base), (label, var)):
    sc = [num(R[s], "score") for s in c]; dp = [int(num(R[s], "max_depth")) for s in c]
    st = 100 * sum(1 for s in c if R[s].get("death_cat") == "starvation") // max(1, len(c))
    mi = 100 * sum(1 for s in c if R[s].get("did_mines")) // max(1, len(c))
    so = 100 * sum(1 for s in c if R[s].get("did_sokoban")) // max(1, len(c))
    print("  %-9s score mean %6.0f med %6.0f | depth med %2d | starv %2d%% combat %2d%% | mines %2d%% soko %d%%" % (
        nm, mean(sc), median(sc), median(dp),
        100*sum(1 for s in c if R[s].get("death_cat")=="combat")//max(1,len(c)),
        st, mi, so) if False else
        "  %-9s score mean %6.0f med %6.0f | depth med %2d | starv %2d%% | mines %2d%% soko %d%%" % (
        nm, mean(sc), median(sc), median(dp), st, mi, so))
w = sum(num(var[s], "score") > num(base[s], "score") for s in c)
l = sum(num(var[s], "score") < num(base[s], "score") for s in c)
print("  paired score: %s wins %d / losses %d / ties %d" % (label, w, l, len(c)-w-l))
import statistics
deltas = [num(var[s],"score")-num(base[s],"score") for s in c]
print("  mean delta %+.0f  median delta %+.0f" % (mean(deltas), median(deltas)))
