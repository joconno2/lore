"""Discrimination / false-positive test. Feed the ALREADY-FIXED function + the original
symptom, with a neutral prompt that explicitly allows 'NO BUG'. If the debugger fabricates
a bug in correct code, that's a false positive -- the tool cries wolf. Measures precision,
the complement to the ~13% recall."""
import json, urllib.request, sys
CASES=json.load(open("/tmp/aa_fixed_cases.json"))
def llm(p,mt=250):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
res=[]
for c in CASES:
    prompt=("This function is from a NetHack bot. A developer once reported this concern about "
            "it:\n\""+c["symptom"]+"\"\n\nHere is the CURRENT version of the function:\n```python\n"+
            c["fixed_func"][:3500]+"\n```\n\nDoes this current version actually contain that bug? "
            "If it genuinely has the bug, reply starting with 'BUG:' and name the exact line. If the "
            "code already handles this correctly, reply starting with 'NO BUG:' and one clause why.")
    d=llm(prompt).strip()
    verdict="BUG" if d.upper().lstrip().startswith("BUG") else ("NOBUG" if "NO BUG" in d.upper()[:15] else "UNCLEAR")
    res.append({"sha":c["sha"],"symptom":c["symptom"],"func":c["func"],"verdict":verdict,"reply":d[:300]})
    print(f"{c['sha'][:7]} {verdict:7} {c['symptom'][:42]}",file=sys.stderr)
json.dump(res,open("/tmp/discrim.json","w"),indent=1)
fp=sum(1 for r in res if r["verdict"]=="BUG")
tn=sum(1 for r in res if r["verdict"]=="NOBUG")
print(f"\nFALSE POSITIVE (claims bug in fixed code): {fp}/{len(res)}",file=sys.stderr)
print(f"TRUE NEGATIVE (says NO BUG): {tn}/{len(res)}",file=sys.stderr)
print(f"UNCLEAR: {len(res)-fp-tn}/{len(res)}",file=sys.stderr)
