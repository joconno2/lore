import json, glob, sys
from scipy import stats
import numpy as np
a, b = sys.argv[1], sys.argv[2]
def load(pfx, s):
    try: return json.load(open(f"/workspace/{pfx}_{s}.json"))
    except Exception: return None
seeds = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"/workspace/{a}_*.json"))
pairs = [(load(a, s), load(b, s)) for s in seeds]
pairs = [(x, y) for x, y in pairs if x and y]
n = len(pairs)
da = np.array([int(x.get("max_depth") or 1) for x, y in pairs])
db = np.array([int(y.get("max_depth") or 1) for x, y in pairs])
diff = db - da  # b - a (positive = b deeper)
wins_b = int((diff > 0).sum()); wins_a = int((diff < 0).sum()); ties = int((diff == 0).sum())
# Wilcoxon signed-rank (paired), two-sided
try:
    W, p = stats.wilcoxon(db, da, zero_method="wilcox")
    wtxt = f"W={W:.0f} p={p:.2e}"
except Exception as e:
    wtxt = f"(wilcoxon err {e})"
# sign test via binomial on non-ties

nz = wins_a + wins_b
sp = stats.binom_test(wins_b, nz, 0.5) if nz else 1.0
# bootstrap 95% CI on mean(diff) -- seeded for reproducibility
rng = np.random.default_rng(12345)
boot = [rng.choice(diff, size=n, replace=True).mean() for _ in range(5000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"[{a}] vs [{b}]  n={n}")
print(f"  mean depth: {a} {da.mean():.2f}  {b} {db.mean():.2f}   median {int(np.median(da))} / {int(np.median(db))}")
print(f"  paired mean diff ({b}-{a}): {diff.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
print(f"  depth wins: {b} {wins_b} | {a} {wins_a} | tie {ties}   sign-test p={sp:.2e}")
print(f"  Wilcoxon signed-rank: {wtxt}")
