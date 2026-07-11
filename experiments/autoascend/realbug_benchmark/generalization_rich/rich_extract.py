"""Generalization extractor: mine real logic-bug fixes from rich (different domain,
pure Python) the same way as AA. Single primary function, bug-sized, descriptive symptom.
Excludes docs/tests/typing/typo/mypy/lint commits (not diagnosable logic bugs)."""
import subprocess, re, json, sys
from collections import defaultdict
REPO="/tmp/rich_gen"
EXCLUDE=re.compile(r"\b(doc|docs|docstring|test|tests|typo|typos|mypy|lint|typing|type|"
    r"changelog|readme|ci|font|fonts|date|dates|comment|format|black|import|annotation)\b",re.I)
def git(*a): return subprocess.run(["git","-C",REPO,*a],capture_output=True,text=True).stdout
def enclosing_func(pl,ln):
    idx=ln-1
    if idx<0 or idx>=len(pl): return None
    for i in range(idx,-1,-1):
        m=re.match(r"^(\s*)def (\w+)",pl[i])
        if m:
            ind=len(m.group(1)); start=i; name=m.group(2); end=len(pl)
            for j in range(start+1,len(pl)):
                l=pl[j]
                if not l.strip(): continue
                ci=len(l)-len(l.lstrip())
                if ci<=ind and l.lstrip().startswith(("def ","class ","@")): end=j; break
            return name,start,end
    return None
shas=[]
for line in git("log","--format=%H %s").splitlines():
    sha,_,subj=line.partition(" ")
    if not re.search(r"\bfix",subj,re.I): continue
    if subj.lower().startswith("merge"): continue
    if EXCLUDE.search(subj): continue
    shas.append((sha,subj))
cases=[]
for sha,subj in shas:
    stat=git("show","--stat","--format=",sha)
    files=[l.split("|")[0].strip() for l in stat.splitlines() if "|" in l]
    lf=[f for f in files if f.startswith("rich/") and f.endswith(".py") and "test" not in f]
    if len(lf)!=1: continue
    f=lf[0]
    psrc=git("show",f"{sha}~1:{f}")
    if not psrc.strip(): continue
    pl=psrc.splitlines()
    diff=git("show","--format=",sha,"--",f)
    changed=[]
    for m in re.finditer(r"@@ -(\d+)(?:,(\d+))? \+",diff):
        s=int(m.group(1)); n=int(m.group(2) or "1"); changed.extend(range(s,s+n))
    if not changed: continue
    funcs=defaultdict(int); bodies={}
    for ln in changed:
        ef=enclosing_func(pl,ln)
        if ef:
            funcs[ef]+=1; bodies[ef]="\n".join(pl[ef[1]:ef[2]])
    if not funcs: continue
    key=max(funcs,key=lambda k:funcs[k]); frac=funcs[key]/len(changed)
    name,start,end=key; body=bodies[key]; nl=body.count("\n")+1
    if frac<0.6 or nl>90 or nl<4: continue
    if len(changed)>45 or len(changed)<2: continue
    cases.append({"sha":sha,"symptom":subj,"file":f,"func":name,"func_lines":nl,
        "n_changed":len(changed),"buggy_func":body,"fix_diff":diff.strip()[:6000]})
print(f"extracted {len(cases)} rich cases",file=sys.stderr)
json.dump(cases,open("/tmp/rich_cases.json","w"),indent=1)
for c in cases[:40]:
    print(f"  {c['sha'][:7]} {c['func_lines']:>2}L n={c['n_changed']:>2} [{c['func']}] {c['symptom'][:52]}",file=sys.stderr)
