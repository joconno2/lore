import json, urllib.request
CASES=json.load(open("/tmp/aa_bench_cases.json"))
def llm(p,mt=200):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
      data=json.dumps({"model":"Qwen/Qwen2.5-32B-Instruct-AWQ","messages":[{"role":"user","content":p}],
      "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
res=[]
for c in CASES:
    p=("A NetHack bot bug: "+c["symptom"]+"\n\nBuggy function:\n```python\n"+c["buggy_func"][:3000]+
       "\n```\n\nState ONLY the exact one-line fix (the specific code change). No analysis, no explanation.")
    res.append({"sha":c["sha"][:7],"symptom":c["symptom"],"fix":llm(p).strip().replace(chr(10)," ")[:250]})
json.dump(res,open("/tmp/gen32.json","w"))
print("done", len(res))
