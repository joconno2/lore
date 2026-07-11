import json, sys, glob
a1, a2 = sys.argv[1], sys.argv[2]   # arm prefixes, e.g. h2h_mock h2h_llm
def load(pfx, s):
    try: return json.load(open(f"{pfx}_{s}.json"))
    except Exception: return None
# infer seed range from files present in arm1
seeds = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{a1}_*.json"))
both = [(s, load(a1, s), load(a2, s)) for s in seeds]
both = [(s, m, l) for s, m, l in both if m and l]
n = len(both)
def depth(r): return int(r.get("max_depth") or 1)
def mnum(r):
    m = str(r.get("milestone") or "")
    order = ["BE_ON_FIRST","GNOMISH_MINES","MINETOWN","FIND_SOKOBAN","SOLVE_SOKOBAN","MINES_END","GO_DOWN"]
    for i,k in enumerate(order):
        if k in m: return i
    return -1
def mines_no_town(r): return bool(r.get("did_mines")) and mnum(r) < 2
def combat(r): return r.get("death_cat") == "combat"
a_deeper = sum(1 for s,m,l in both if depth(m) > depth(l))
b_deeper = sum(1 for s,m,l in both if depth(l) > depth(m))
print(f"paired n={n}   [{a1}] vs [{a2}]")
print(f"  depth wins: {a1} {a_deeper}  |  {a2} {b_deeper}  |  tie {n-a_deeper-b_deeper}")
print(f"  mean depth: {a1} {sum(depth(m) for s,m,l in both)/n:.2f}   {a2} {sum(depth(l) for s,m,l in both)/n:.2f}")
print(f"  reached Mines but stalled pre-Minetown: {a1} {sum(mines_no_town(m) for s,m,l in both)}   {a2} {sum(mines_no_town(l) for s,m,l in both)}")
print(f"  combat deaths: {a1} {sum(combat(m) for s,m,l in both)}   {a2} {sum(combat(l) for s,m,l in both)}")
print(f"  reached Minetown: {a1} {sum(mnum(m)>=2 for s,m,l in both)}   {a2} {sum(mnum(l)>=2 for s,m,l in both)}")
