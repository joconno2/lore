import json, glob, statistics as st
def num(x):
    try: return int(x)
    except: return None
def dmap(pat, seeds):
    m = {}
    for f in glob.glob(pat):
        r = json.load(open(f)); s = num(r.get("seed"))
        if s in seeds and num(r.get("max_depth")) is not None:
            m[s] = r
    return m
sd = set(range(1200, 1300))
A = dmap("abl_full_*.json", sd); B = dmap("elb_*.json", sd)
common = sorted(set(A) & set(B))
da = [num(A[s]["max_depth"]) for s in common]; db = [num(B[s]["max_depth"]) for s in common]
diffs = [db[i] - da[i] for i in range(len(common))]
pos = sum(1 for x in diffs if x > 0); neg = sum(1 for x in diffs if x < 0); tie = sum(1 for x in diffs if x == 0)
print("paired n=%d  full med %.1f mean %.2f | elbfix med %.1f mean %.2f" % (
    len(common), st.median(da), st.mean(da), st.median(db), st.mean(db)))
print("  elbfix-full: mean diff %.2f  W+/-/0 = %d/%d/%d" % (st.mean(diffs), pos, neg, tie))
try:
    from scipy.stats import wilcoxon
    if any(diffs): print("  Wilcoxon p=%.4f" % wilcoxon(db, da).pvalue)
except Exception as e: print("  scipy:", e)
# elbfix games that died to @ / wand -- what did AA do?
wanddie = [B[s] for s in common if "wand" in str(B[s].get("killer","")).lower() or B[s].get("killer","") in ("a soldier","a lieutenant","a watchman")]
print("\nelbfix games dying to wand/@ (n=%d):" % len(wanddie))
for r in wanddie[:6]:
    print("  seed", r.get("seed"), "dl", r.get("max_depth"), "| msgs:", str(r.get("msgs_tail"))[:120])
