"""Prompt-diversity recall@5. pass@k tested TEMPERATURE diversity (same prompt -> consistent
failures). This tests PROMPT diversity: 5 different framings prime different reasoning. If the
union of 5 diverse-prompt candidates contains the real fix >> pass@1 (13%), a SELF-CONTAINED
ranker (generate-diverse -> recognize at 67%) beats open generation without needing external
candidates. Dumps candidates for recall@5 judging."""
import json, urllib.request, sys
CASES=json.load(open("/tmp/aa_bench_cases.json"))
FRAMINGS=[
 "Trace the variable values step by step; where does the logic produce the wrong value?",
 "Look specifically for an off-by-one, boundary, or empty-collection (empty list/mask) error.",
 "Check every conditional carefully: is any condition inverted, negated wrong, or using the wrong test?",
 "Check for a type / identifier / attribute / wrong-constant mistake (e.g. wrong field or wrong enum).",
 "Check the control flow: an early return, a missing case, a loop that doesn't advance, or a missing guard.",
]
def llm(p,mt=160):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.3}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
res=[]
for c in CASES:
    cands=[]
    for fr in FRAMINGS:
        p=("A NetHack bot bug:\n"+c["symptom"]+"\n\nFunction:\n```python\n"+c["buggy_func"][:3000]+
           "\n```\n\n"+fr+"\n\nState the ONE-LINE fix concisely.")
        cands.append(llm(p).strip().replace("\n"," ")[:200])
    res.append({"sha":c["sha"][:7],"symptom":c["symptom"],"candidates":cands})
    print(f"{c['sha'][:7]} 5 diverse candidates",file=sys.stderr)
json.dump(res,open("/tmp/diverse.json","w"),indent=1)
print("DONE",file=sys.stderr)
