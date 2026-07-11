"""Gate stability: retrieval is deterministic (term-rarity+grep), so only the LLM
gate/diagnosis vary. Call the BUG/FUNDAMENTAL gate 3x per case; report flip rate.
Tests the reliability of the 'correctly declines fundamental' calibration claim.
All 19 cases are REAL bugs -> correct gate = BUG every time; FUNDAMENTAL = a miss."""
import json, urllib.request, re, glob, os, subprocess, sys
UP=os.path.expanduser("~/aa_upstream")
CASES=json.load(open("/tmp/aa_bench3_cases.json"))
def llm(p,mt=120):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
def checkout(sha):
    d="/tmp/aacase";subprocess.run(f"rm -rf {d}&&mkdir -p {d}",shell=True)
    subprocess.run(f"git -C {UP} archive {sha}~1|tar -x -C {d}",shell=True);return d
def blob_for(symptom,src):
    SRC={}
    for p in glob.glob(src+"/**/*.py",recursive=True):
        try:SRC[p]=open(p).read().split("\n")
        except:pass
    ALL="\n".join("\n".join(v) for v in SRC.values()).lower()
    words=re.findall(r"[a-z]+",symptom.lower())
    cands=set(words)|{words[i]+"_"+words[i+1] for i in range(len(words)-1)}
    STOP={"turns","games","level","never","the","its","then","dies","gets","other","kill","while","bot","and",
        "reaches","single","set","with","work","well","some","case","being","blocked","measure","longer","fix",
        "fixes","fixed","bug","add","small","make","improve","better"}
    g=sorted({c for c in cands if len(c)>=6 and c in ALL and c not in STOP},key=lambda c:ALL.count(c))[:5]
    funcs={}
    for path,lines in SRC.items():
        defs=[(i,mm.group(1)) for i,mm in ((i,re.match(r"\s*def (\w+)",l)) for i,l in enumerate(lines)) if mm]
        for di,n in defs:
            end=next((d for d,_ in defs if d>di),len(lines));body="\n".join(lines[di:end]).lower()
            sc=sum(body.count(x)/max(1,ALL.count(x)) for x in g)
            if sc>0:funcs[os.path.basename(path)+":"+n]=(sc,"\n".join(lines[di:end])[:1200])
    top=sorted(funcs.items(),key=lambda x:-x[1][0])[:6]
    return "\n\n".join("# %s\n%s"%(k,v[1]) for k,v in top)[:5000]
def gate(symptom,blob):
    o=llm("A NetHack bot (a strong competition winner) failure:\n"+symptom+
      "\n\nRelevant retrieved functions:\n"+blob+"\n\nIs this caused by a SPECIFIC FIXABLE BUG in one of "
      "these functions, or is it FUNDAMENTAL game difficulty (no specific bug)? Answer with ONLY the first "
      "word: BUG or FUNDAMENTAL, then one sentence why.",120)
    return "BUG" if o.strip().upper().startswith("BUG") else "FUND"
res=[];flips=0;allbug=0
for i,c in enumerate(CASES):
    b=blob_for(c["symptom"],checkout(c["sha"]))
    verdicts=[gate(c["symptom"],b) for _ in range(3)]
    flip=len(set(verdicts))>1;flips+=flip
    if all(v=="BUG" for v in verdicts):allbug+=1
    res.append({"sha":c["sha"][:7],"symptom":c["symptom"][:40],"verdicts":verdicts,"flip":flip})
    print(f"[{i+1}/{len(CASES)}] {c['sha'][:7]} {verdicts} {'FLIP' if flip else ''}",file=sys.stderr)
n=len(CASES)
print(f"\nGATE STABILITY over 3 runs, N={n} (all are real bugs -> correct=BUG x3):",file=sys.stderr)
print(f"  flipped across runs: {flips}/{n} = {flips/n:.0%}",file=sys.stderr)
print(f"  stable BUG x3 (correct+stable): {allbug}/{n} = {allbug/n:.0%}",file=sys.stderr)
json.dump(res,open("/tmp/gate_stability.json","w"),indent=1)
