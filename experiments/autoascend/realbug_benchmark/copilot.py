"""Co-pilot test: measure diagnosis given PERFECT localization (the human/tool points at
the exact buggy function). Feed ONLY the single correct buggy function + the author
symptom -> diagnose. Compare to the fully-autonomous rate (2/9). Separates the
localization bottleneck from the diagnosis bottleneck, and measures the realistic
deployment (dev points at suspect code, LLM diagnoses)."""
import json, urllib.request, sys
CASES=json.load(open("/tmp/aa_bench_cases.json"))  # v2: has single buggy_func per case
def llm(p,mt=300):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
res=[]
for c in CASES:
    prompt=("A NetHack bot (a strong competition winner) has this reported problem:\n"+c["symptom"]+
            "\n\nThe bug is in this function:\n```python\n"+c["buggy_func"][:3500]+
            "\n```\n\nIdentify the specific bug and the exact fix. Be concrete about which line(s).")
    d=llm(prompt).strip()[:600]
    res.append({"sha":c["sha"],"symptom":c["symptom"],"func":c["func"],"diagnosis":d})
    print(f"{c['sha']} {c['symptom'][:45]}",file=sys.stderr)
json.dump(res,open("/tmp/copilot.json","w"),indent=1)
print("DONE",file=sys.stderr)
