"""Truncation-fix re-test: the pass@k run found a false 'syntax error' diagnosis caused
by truncating each retrieved function to 1200 chars. Re-run diagnosis on the 6 localized
cases showing the top-3 functions at FULL length (bug-sized funcs, ~fits 4k context).
Separates the method artifact from the capability ceiling: if the rate stays low with
whole functions, the 5-22% is real, not a truncation artifact."""
import json, urllib.request, re, glob, os, subprocess, sys
UP=os.path.expanduser("~/aa_upstream")
LOCALIZED=["073d770","e328738","bad5143","de0f93a","bde22a3","0114ad4"]
CASES={c["sha"][:7]:c for c in json.load(open("/tmp/aa_bench3_cases.json"))}
def llm(p,mt=300):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
        "max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))["choices"][0]["message"]["content"]
def checkout(sha):
    d="/tmp/aacase";subprocess.run(f"rm -rf {d}&&mkdir -p {d}",shell=True)
    subprocess.run(f"git -C {UP} archive {sha}~1|tar -x -C {d}",shell=True);return d
def diagnose(symptom,src):
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
            if sc>0:funcs[os.path.basename(path)+":"+n]=(sc,"\n".join(lines[di:end]))  # FULL body, no [:1200]
    top=sorted(funcs.items(),key=lambda x:-x[1][0])[:3]   # top-3 (fewer) shown WHOLE
    blob="\n\n".join("# %s\n%s"%(k,v[1][:2600]) for k,v in top)[:7200]  # generous per-func, fits ~4k ctx
    return llm("Bug:\n"+symptom+"\n\nFunctions (complete):\n"+blob+
               "\n\nWhich function, what bug, what fix? Brief.",300).strip()[:500]
res=[]
for sha in LOCALIZED:
    c=CASES[sha];d=diagnose(c["symptom"],checkout(c["sha"]))
    res.append({"sha":sha,"symptom":c["symptom"],"diagnosis":d})
    print(f"{sha} done",file=sys.stderr)
json.dump(res,open("/tmp/notrunc.json","w"),indent=1)
print("DONE",file=sys.stderr)
