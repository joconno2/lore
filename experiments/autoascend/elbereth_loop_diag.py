import json, glob, collections
R = [json.load(open(f)) for f in glob.glob("be_engine_*.json")]
def has_engrave_loop(r):
    m = str(r.get("msgs_tail") or "")
    return m.count("write in the dust") >= 2 or m.count("You read:") >= 2
loop = [r for r in R if has_engrave_loop(r)]
print("games ending in Elbereth-engrave loop signature:", len(loop), "/", len(R))
print("  their death_cat:", collections.Counter(r.get("death_cat") for r in loop).most_common())
print("  their depth median:", sorted([r.get("max_depth") for r in loop if isinstance(r.get("max_depth"), int)])[len(loop)//2] if loop else "-")
# broader: any game whose msgs_tail is dominated by dust/Elbereth
def elb(r):
    m = str(r.get("msgs_tail") or "").lower()
    return "elbereth" in m or "write in the dust" in m
elbg = [r for r in R if elb(r)]
print("games with Elbereth/dust in final msgs:", len(elbg), "/", len(R), "->", collections.Counter(r.get("death_cat") for r in elbg).most_common())
