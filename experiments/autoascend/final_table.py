"""Final tail-preservation result: base vs crash_recovery vs crveto(mock) vs
crveto(llm), matched seeds. Shows score + what happened to the tail's
crash/petrification deaths."""
import json, glob, sys
from statistics import mean, median
def load(pat):
    r = {}
    for f in glob.glob(pat):
        try: d = json.load(open(f)); r[int(d["seed"])] = d
        except Exception: pass
    return r
arms = {"base": load("prof_[123]*.json"), "crashrec": load("prof_cr_*.json"),
        "crveto_mock": load("prof_crv_*.json"), "crveto_llm": load("prof_crvl_*.json")}
def num(d, k):
    try: return float(d.get(k) or 0)
    except Exception: return 0.0
# common seeds across base + each arm
base = arms["base"]
print("arm           n    mean   median  | vs base (paired): W/L  meanDelta")
for nm in ("base", "crashrec", "crveto_mock", "crveto_llm"):
    R = arms[nm]
    if not R: continue
    c = sorted(set(base) & set(R))
    sc = [num(R[s], "score") for s in c]
    line = "%-12s %3d  %6.0f  %6.0f" % (nm, len(R), mean(sc) if sc else 0, median(sc) if sc else 0)
    if nm != "base" and c:
        w = sum(num(R[s],"score") > num(base[s],"score") for s in c)
        l = sum(num(R[s],"score") < num(base[s],"score") for s in c)
        dl = mean([num(R[s],"score")-num(base[s],"score") for s in c])
        line += "  | %d/%d  %+.0f" % (w, l, dl)
    print(line)
# tail: what happened to base petrif/paralysis + crash games under each arm
for cat in ("petrification", "paralysis", "crash"):
    seeds = [s for s in base if base[s].get("death_cat") == cat]
    print("\nbase %s seeds (n=%d):" % (cat, len(seeds)))
    for nm in ("base", "crashrec", "crveto_mock", "crveto_llm"):
        R = arms[nm]
        ss = [s for s in seeds if s in R]
        if not ss: continue
        print("  %-12s mean %6.0f  (%s)" % (nm, mean([num(R[s],"score") for s in ss]),
              ",".join("%d:%d" % (s, int(num(R[s],"score"))) for s in ss[:6])))
