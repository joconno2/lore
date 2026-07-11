"""v4 -- grounded retrieval + RARITY ranking (fixes v3's crowding). Rank grounded
terms by rarity (rare = specific = useful), rank functions by rare-term hits."""
import json, urllib.request, re, glob, os
AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")
def llm(p, mt=380):
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],
            "max_tokens":mt,"temperature":0.2}).encode(), headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=100))["choices"][0]["message"]["content"]
SRC={}
for path in glob.glob(AA_SRC+"/**/*.py",recursive=True):
    try: SRC[path]=open(path).read().split("\n")
    except: pass
ALL="\n".join("\n".join(v) for v in SRC.values()).lower()
import sys; SYMPTOMS=sys.argv[1] if len(sys.argv)>1 else ("The bot stays on Dungeon Level 1 for 14000+ turns, experience level rises to ~7, "
 "then it starves. It never issues the descend command though the downstairs is visible.")
words=re.findall(r"[a-z]+",SYMPTOMS.lower())
cands=set(words)|{words[i]+"_"+words[i+1] for i in range(len(words)-1)}
STOP={"turns","games","level","never","the","its","then","it","for","on","to","though","stays","rises","visible"}
# grounded + ranked by RARITY (rarer term = more specific)
grounded=sorted({c for c in cands if len(c)>=6 and c in ALL and c not in STOP}, key=lambda c: ALL.count(c))
grounded=grounded[:5]
print("grounded (rare-first):",[(g,ALL.count(g)) for g in grounded])
# rank functions by weighted rare-term hits (weight = 1/frequency)
funcs={}
for path,lines in SRC.items():
    defs=[(i,m.group(1)) for i,m in ((i,re.match(r"\s*def (\w+)",l)) for i,l in enumerate(lines)) if m]
    for di,n in [(defs[k][0],defs[k][1]) for k in range(len(defs))]:
        end=next((d for d,_ in defs if d>di),len(lines))
        body="\n".join(lines[di:end]).lower()
        score=sum(body.count(g)/max(1,ALL.count(g)) for g in grounded)
        if score>0:
            funcs[os.path.basename(path)+":"+n]=(score,"\n".join(lines[di:end])[:1400])
top=sorted(funcs.items(),key=lambda x:-x[1][0])[:6]
print("top functions:",[k for k,_ in top])
blob="\n\n".join("# --- %s ---\n%s"%(k,v[1]) for k,v in top)
print("\nDIAGNOSIS:\n"+llm("A NetHack bot bug:\n"+SYMPTOMS+"\n\nMost relevant functions:\n\n"
    +blob[:6500]+"\n\nWhich function has the bug, what is it, the fix? Brief.",380))
