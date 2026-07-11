"""Calibration / self-filter test. Re-run the AA co-pilot diagnoses asking for a 0-100
confidence. If correct diagnoses carry higher confidence than wrong ones, the tool can
self-filter (abstain when unsure) and the 13% is usable. If confidence is flat across
correct/wrong, the 13% is buried in confident wrong answers -> untrustworthy. Decisive
usability question, predicted by the zero-discrimination + confident-fabrication findings."""
import json, urllib.request, re, sys
CASES=json.load(open("/tmp/aa_bench_cases.json"))
# my hand verdicts from the co-pilot test (copilot_score.py)
VERD={"073d770":"CORRECT","0114ad4":"CORRECT",
 "bad5143":"PARTIAL","6bd6eee":"PARTIAL","195eaad":"PARTIAL","bde22a3":"PARTIAL","97a231d":"PARTIAL",
 "bf8617a":"WRONG","ec3cc35":"WRONG","88a1b9b":"WRONG","3b25d57":"WRONG","4b536a3":"WRONG",
 "e3328d1":"WRONG","179a676":"WRONG","326741c":"WRONG"}
def llm(p,mt=300):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
res=[]
for c in CASES:
    prompt=("A NetHack bot has this reported problem:\n"+c["symptom"]+"\n\nThe bug is in this function:\n```python\n"+
        c["buggy_func"][:3500]+"\n```\n\nIdentify the specific bug and the exact fix. Then on the LAST line write "
        "exactly 'CONFIDENCE: N' where N is 0-100 = your probability this is the ACTUAL bug the developer fixed.")
    d=llm(prompt)
    m=re.search(r"CONFIDENCE:\s*(\d+)",d)
    conf=int(m.group(1)) if m else None
    res.append({"sha":c["sha"][:7],"verdict":VERD.get(c["sha"][:7],"?"),"conf":conf})
    print(f"{c['sha'][:7]} {VERD.get(c['sha'][:7],'?'):8} conf={conf}",file=sys.stderr)
json.dump(res,open("/tmp/copilot_conf.json","w"),indent=1)
def avg(vs):
    xs=[r["conf"] for r in res if r["verdict"] in vs and r["conf"] is not None]
    return (sum(xs)/len(xs),len(xs)) if xs else (None,0)
cA,nA=avg(["CORRECT"]); cP,nP=avg(["PARTIAL"]); cW,nW=avg(["WRONG"])
print(f"\nmean confidence: CORRECT={cA} (n={nA})  PARTIAL={cP} (n={nP})  WRONG={cW} (n={nW})",file=sys.stderr)
print("If CORRECT ~ WRONG confidence, the tool cannot self-filter -> 13% is buried.",file=sys.stderr)
