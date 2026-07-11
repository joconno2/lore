"""pass@k on the generative diagnosis stage. The gate is stable and retrieval is
deterministic, so the ~5-22% + the run variance live in diagnosis. Question: given the
RIGHT code (localized cases) + symptom, is the low rate a CEILING (model can't diagnose
complex real bugs) or a SAMPLING problem (gets it sometimes)? Sample diagnosis 5x on the
6 localized v3 cases; judge pass@1 (per-sample) vs pass@5 (any correct)."""
import json, urllib.request, re, glob, os, subprocess, sys
UP=os.path.expanduser("~/aa_upstream")
LOCALIZED=["073d770","e328738","bad5143","de0f93a","bde22a3","0114ad4"]
CASES={c["sha"][:7]:c for c in json.load(open("/tmp/aa_bench3_cases.json"))}
def llm(p,mt=300,temp=0.2):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":temp}).encode(),headers={"Content-Type":"application/json"})
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
res=[]
for sha in LOCALIZED:
    c=CASES[sha];b=blob_for(c["symptom"],checkout(c["sha"]))
    samples=[]
    for k in range(5):
        d=llm("Bug:\n"+c["symptom"]+"\n\nFunctions:\n"+b+"\n\nWhich function, what bug, what fix? Brief.",
              300,temp=0.2 if k<3 else 0.7)
        samples.append(d.strip()[:500])
    res.append({"sha":sha,"symptom":c["symptom"],"changed_funcs":c["changed_funcs"],"samples":samples})
    print(f"{sha} sampled 5x",file=sys.stderr)
json.dump(res,open("/tmp/pass_k.json","w"),indent=1)
print("DONE",file=sys.stderr)
