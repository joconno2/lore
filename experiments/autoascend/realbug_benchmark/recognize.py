"""Recognition vs generation. The exact-token thesis predicts the model KNOWS the fix but
can't GENERATE it. Test: show the buggy function + symptom + a PROPOSED fix, ask YES/NO
does it resolve the bug. Do this for (a) the REAL fix and (b) a WRONG fix (a real edit from
a different case). If it accepts real >> accepts wrong, it RECOGNIZES correct fixes despite
13% generation -> ceiling is generation (bigger model / candidate-presentation could help).
If it accepts both or neither, it's a knowledge/reasoning limit -> scale unlikely to help.
Non-blocked (14B); informs the GPU-blocked bigger-model question."""
import json, urllib.request, re, sys
CASES=json.load(open("/tmp/aa_bench_cases.json"))
def added_lines(diff):
    return "\n".join(l[1:] for l in diff.split("\n")
                     if l.startswith("+") and not l.startswith("+++") and l[1:].strip())[:500]
def llm(p,mt=120):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
def ask(c, fix):
    p=("A NetHack bot has this reported bug:\n"+c["symptom"]+"\n\nThe buggy function:\n```python\n"+
       c["buggy_func"][:3000]+"\n```\n\nProposed fix (added/changed lines):\n```python\n"+fix+
       "\n```\n\nDoes applying THIS proposed fix actually resolve the reported bug? Answer with ONLY "
       "'YES' or 'NO' then one clause.")
    d=llm(p).strip()
    return "YES" if d.upper().lstrip().startswith("YES") else ("NO" if d.upper().lstrip().startswith("NO") else "?")
res=[]
n=len(CASES)
for i,c in enumerate(CASES):
    real=added_lines(c["fix_diff"])
    wrong=added_lines(CASES[(i+7)%n]["fix_diff"])  # a real edit from a different case
    if not real.strip() or not wrong.strip(): continue
    vr=ask(c,real); vw=ask(c,wrong)
    res.append({"sha":c["sha"][:7],"real":vr,"wrong":vw})
    print(f"{c['sha'][:7]} real={vr:3} wrong={vw:3} {c['symptom'][:38]}",file=sys.stderr)
json.dump(res,open("/tmp/recognize.json","w"),indent=1)
ar=sum(1 for r in res if r["real"]=="YES"); aw=sum(1 for r in res if r["wrong"]=="YES")
m=len(res)
print(f"\naccept REAL fix:  {ar}/{m} = {ar/m:.0%}",file=sys.stderr)
print(f"accept WRONG fix: {aw}/{m} = {aw/m:.0%}",file=sys.stderr)
print(f"recognition margin (real-wrong): {(ar-aw)/m:+.0%}",file=sys.stderr)
print("high real & low wrong -> RECOGNIZES fixes (generation-limited, scale may help).",file=sys.stderr)
print("both high or both low -> no recognition (knowledge/reasoning limit).",file=sys.stderr)
