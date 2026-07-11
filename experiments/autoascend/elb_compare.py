import json, glob, statistics as st, collections
def num(x):
    try: return int(x)
    except: return None
def loop(r):
    m = str(r.get("msgs_tail") or "")
    return m.count("write in the dust") >= 2 or m.count("You read:") >= 2
def agg(name, pat, seeds):
    R = [json.load(open(f)) for f in glob.glob(pat)]
    R = [r for r in R if num(r.get("seed")) in seeds]
    d = [num(r.get("max_depth")) for r in R if num(r.get("max_depth")) is not None]
    if not d: 
        print(name, "no data"); return
    comb = sum(1 for r in R if r.get("death_cat") == "combat")
    lp = sum(1 for r in R if loop(r))
    print("%-10s n=%3d med %s mean %.2f max %2d | combat %d starv %d | elb-loop-deaths %d (%d%%)" % (
        name, len(R), st.median(d), st.mean(d), max(d), comb,
        sum(1 for r in R if r.get("death_cat")=="starvation"), lp, 100*lp//len(R)))
sd = set(range(1200, 1300))
agg("full", "abl_full_*.json", sd)
agg("elbfix", "elb_*.json", sd)
