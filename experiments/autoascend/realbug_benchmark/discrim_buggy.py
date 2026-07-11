"""Complement: same neutral 'BUG or NO BUG' prompt, but on the BUGGY function. Combined
with discrim.py (fixed) this gives the real discrimination: P(BUG|buggy) vs P(BUG|fixed).
If it says BUG more on buggy than fixed, it discriminates; if similar, it's just cautious."""
import json, urllib.request, sys
CASES=json.load(open("/tmp/aa_bench_cases.json"))
def llm(p,mt=250):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
res=[]
for c in CASES:
    prompt=("This function is from a NetHack bot. A developer once reported this concern about "
            "it:\n\""+c["symptom"]+"\"\n\nHere is the CURRENT version of the function:\n```python\n"+
            c["buggy_func"][:3500]+"\n```\n\nDoes this current version actually contain that bug? "
            "If it genuinely has the bug, reply starting with 'BUG:' and name the exact line. If the "
            "code already handles this correctly, reply starting with 'NO BUG:' and one clause why.")
    d=llm(prompt).strip()
    verdict="BUG" if d.upper().lstrip().startswith("BUG") else ("NOBUG" if "NO BUG" in d.upper()[:15] else "UNCLEAR")
    res.append({"sha":c["sha"],"symptom":c["symptom"],"verdict":verdict,"reply":d[:200]})
    print(f"{c['sha'][:7]} {verdict:7} {c['symptom'][:42]}",file=sys.stderr)
json.dump(res,open("/tmp/discrim_buggy.json","w"),indent=1)
bug=sum(1 for r in res if r["verdict"]=="BUG")
print(f"\nsays BUG on BUGGY code: {bug}/{len(res)} (= true-positive detection under neutral prompt)",file=sys.stderr)
