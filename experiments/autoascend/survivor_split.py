import json, glob, sys
from statistics import median
pfx = sys.argv[1]
R = [json.load(open(f)) for f in glob.glob(f"{pfx}_*.json")]
def d(r): return int(r.get("max_depth") or 1)
deep = [r for r in R if d(r) >= 10]
shallow = [r for r in R if d(r) <= 4]
def summ(grp, name):
    if not grp: print(f"{name}: n=0"); return
    n = len(grp)
    xl = [int(r.get("xl") or 1) for r in grp]
    kit = sum(1 for r in grp if r.get("key_items"))
    mines = sum(1 for r in grp if r.get("did_mines"))
    soko = sum(1 for r in grp if r.get("did_sokoban"))
    town = sum(1 for r in grp if r.get("did_minetown"))
    from collections import Counter
    dc = Counter(r.get("death_cat") for r in grp)
    # XL at the moment they crossed DL5, from traj (turn,depth,xl,hpfrac,hung)
    xl_at5 = []
    for r in grp:
        for t in (r.get("traj") or []):
            if len(t) >= 3 and t[1] >= 5:
                xl_at5.append(t[2]); break
    print(f"{name}: n={n} medXL={median(xl):.0f} kit={100*kit//n}% mines={100*mines//n}% town={100*town//n}% soko={100*soko//n}% deaths={dict(dc.most_common(4))}")
    if xl_at5: print(f"     medXL_when_first_at_DL5 = {median(xl_at5):.1f}  (n={len(xl_at5)})")
print(f"=== {pfx}: deep (DL>=10) vs shallow (DL<=4) ===")
summ(deep, "DEEP ")
summ(shallow, "SHALLOW")
