"""Quantify the debugger's RETRIEVAL accuracy: for each known bug (symptom +
ground-truth function), run grounded+rarity retrieval and report the RANK of the
ground-truth function. No LLM calls -- pure retrieval metric (retrieval@K)."""
import re, glob, os
AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")
SRC = {}
for path in glob.glob(AA_SRC + "/**/*.py", recursive=True):
    try: SRC[path] = open(path).read().split("\n")
    except: pass
ALL = "\n".join("\n".join(v) for v in SRC.values()).lower()

def retrieve(S):
    words = re.findall(r"[a-z]+", S.lower())
    cands = set(words) | {words[i]+"_"+words[i+1] for i in range(len(words)-1)}
    STOP = {"turns","games","level","never","the","its","then","dies","gets","other","kill",
            "while","bot","and","reaches","single","set","with","work","well","monster","monsters"}
    g = sorted({c for c in cands if len(c)>=6 and c in ALL and c not in STOP}, key=lambda c: ALL.count(c))[:5]
    funcs = {}
    for path, lines in SRC.items():
        defs = [(i, mm.group(1)) for i, mm in ((i, re.match(r"\s*def (\w+)", l)) for i, l in enumerate(lines)) if mm]
        for di, n in defs:
            end = next((d for d,_ in defs if d>di), len(lines))
            body = "\n".join(lines[di:end]).lower()
            sc = sum(body.count(x)/max(1,ALL.count(x)) for x in g)
            if sc > 0: funcs[n] = max(funcs.get(n,0), sc)
    return [n for n,_ in sorted(funcs.items(), key=lambda x:-x[1])]

CASES = [
    ("stays on Dungeon Level 1 for 14000 turns, experience level rises to 7, then starves, never descends though downstairs visible", "current_strategy"),
    ("dies frozen by a floating eye gaze: it melees the floating eye, gets paralyzed, adjacent monsters kill it", "melee_monster_priority"),
    ("dies while praying: a hostile monster is adjacent and kills it during the multi-turn prayer", "is_safe_to_pray"),
    ("loops writing Elbereth in the dust against a human soldier and its own pet, taking damage until it dies, Elbereth does not scare them", "elbereth_action"),
    ("re-opens the same empty sack in inventory hundreds of times, wasting turns, never remembering it checked it", "check_container_content"),
]
at1=at3=at6=miss=0
for S, gt in CASES:
    ranked = retrieve(S)
    rank = ranked.index(gt)+1 if gt in ranked else 0
    print("%-26s -> rank %s (%s)" % (gt, rank if rank else "MISS", "@1" if rank==1 else "@3" if rank<=3 and rank else "@6" if rank<=6 and rank else "miss"))
    if rank==1: at1+=1
    if rank and rank<=3: at3+=1
    if rank and rank<=6: at6+=1
    if not rank: miss+=1
n=len(CASES)
print("\nRETRIEVAL@1: %d/%d  @3: %d/%d  @6: %d/%d  MISS: %d/%d"%(at1,n,at3,n,at6,n,miss,n))
