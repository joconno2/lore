"""v5 -- CALIBRATED autonomous debugger: grounded+rarity retrieval + a NEUTRAL
calibration gate (BUG vs FUNDAMENTAL) before diagnosis, fixing v4's false-positive
on fundamental issues. Finds real bugs AND declines fundamental ones."""
import json, urllib.request, re, glob, os, sys
AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")
def llm(p, mt=350):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=100))["choices"][0]["message"]["content"]
SRC={}
for path in glob.glob(AA_SRC+"/**/*.py",recursive=True):
    try: SRC[path]=open(path).read().split("\n")
    except: pass
ALL="\n".join("\n".join(v) for v in SRC.values()).lower()
S=sys.argv[1]
words=re.findall(r"[a-z]+",S.lower()); cands=set(words)|{words[i]+"_"+words[i+1] for i in range(len(words)-1)}
STOP={"turns","games","level","never","the","its","then","dies","gets","other","kill","while","bot","and","reaches","single","set","with","its","work","well"}
g=sorted({c for c in cands if len(c)>=6 and c in ALL and c not in STOP},key=lambda c:ALL.count(c))[:5]
funcs={}
for path,lines in SRC.items():
    defs=[(i,mm.group(1)) for i,mm in ((i,re.match(r"\s*def (\w+)",l)) for i,l in enumerate(lines)) if mm]
    for di,n in defs:
        end=next((d for d,_ in defs if d>di),len(lines)); body="\n".join(lines[di:end]).lower()
        sc=sum(body.count(x)/max(1,ALL.count(x)) for x in g)
        if sc>0: funcs[os.path.basename(path)+":"+n]=(sc,"\n".join(lines[di:end])[:1300])
top=sorted(funcs.items(),key=lambda x:-x[1][0])[:6]
blob="\n\n".join("# %s\n%s"%(k,v[1]) for k,v in top)
# CALIBRATION GATE (neutral)
gate=llm("A NetHack bot (a strong competition winner) failure:\n"+S+"\n\nRelevant retrieved "
    "functions:\n"+blob[:5500]+"\n\nIs this caused by a SPECIFIC FIXABLE BUG in one of these "
    "functions, or is it FUNDAMENTAL game difficulty (no specific bug)? Answer with ONLY the "
    "first word: BUG or FUNDAMENTAL, then one sentence why.",120)
print("GATE:",gate.strip()[:200])
if gate.strip().upper().startswith("BUG"):
    print("\nDIAGNOSIS:",llm("Bug:\n"+S+"\n\nFunctions:\n"+blob[:5500]+"\n\nWhich function, what "
        "bug, what fix? Brief.",300)[:400])
else:
    print("\n(declined -- fundamental, no autonomous fix attempted)")
