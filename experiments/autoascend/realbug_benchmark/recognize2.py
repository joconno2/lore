"""Recognition, forced-choice (removes the NO-bias that broke the yes/no version).
4 lettered candidate fixes = the REAL fix + 3 real edits from other cases; ask which
letter fixes THIS bug. Chance=25%. If pick-real >> 25%, the model RECOGNIZES the fix it
can't generate (13%) -> generation-limited (scale/candidate-presentation may help). If
~25%, no recognition -> knowledge/reasoning limit. Deterministic shuffle by index (no rng)."""
import json, urllib.request, re, sys
CASES=json.load(open("/tmp/aa_bench_cases.json"))
def added(diff):
    return "\n".join(l[1:] for l in diff.split("\n")
                     if l.startswith("+") and not l.startswith("+++") and l[1:].strip())[:280]
def llm(p,mt=40):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
n=len(CASES); res=[]; correct=0; total=0
for i,c in enumerate(CASES):
    real=added(c["fix_diff"])
    d1=added(CASES[(i+5)%n]["fix_diff"]); d2=added(CASES[(i+9)%n]["fix_diff"]); d3=added(CASES[(i+12)%n]["fix_diff"])
    cands=[real,d1,d2,d3]
    if any(not x.strip() for x in cands) or len(set(cands))<4: continue
    pos=i%4                       # deterministic real position
    cands[0],cands[pos]=cands[pos],cands[0]
    letters="ABCD"; ans=letters[pos]
    block="\n".join(f"{letters[k]}:\n{cands[k]}" for k in range(4))
    p=("A NetHack bot bug:\n"+c["symptom"]+"\n\nBuggy function:\n```python\n"+c["buggy_func"][:2600]+
       "\n```\n\nExactly ONE of these is the real fix. Which?\n\n"+block+
       "\n\nReply with ONLY the letter A, B, C, or D.")
    out=llm(p).strip().upper()
    m=re.search(r"[ABCD]",out); pick=m.group(0) if m else "?"
    ok=pick==ans; correct+=ok; total+=1
    res.append({"sha":c["sha"][:7],"answer":ans,"pick":pick,"ok":ok})
    print(f"{c['sha'][:7]} ans={ans} pick={pick} {'OK' if ok else ''}",file=sys.stderr)
json.dump(res,open("/tmp/recognize2.json","w"),indent=1)
print(f"\nRECOGNITION (4-way forced choice): {correct}/{total} = {correct/total:.0%}  (chance=25%)",file=sys.stderr)
print(f"vs GENERATION (co-pilot, name the fix): 2/15 = 13%",file=sys.stderr)
