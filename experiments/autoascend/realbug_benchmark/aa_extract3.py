"""Extraction v3: ALL logic-touching fix commits (not just single-function), capped
to bug-sized diffs. Record EVERY changed function so scoring can be 'diagnosis matches
ANY changed function' -- this dissolves the bundled-commit confound that forced the
clean/bundled split in v2, and lifts N from 15 to ~30 for a real confidence interval.
"""
import subprocess, re, json, sys
from collections import defaultdict

REPO = "/tmp/aa_upstream"
LOGIC_RE = re.compile(r"heur/(agent|item|fight_heur|global_logic|exploration_logic|"
                      r"character|combat|level|soko\w*|strategy|inventory)\w*\.py$")

def git(*a): return subprocess.run(["git","-C",REPO,*a],capture_output=True,text=True).stdout

def enclosing_func(plines, ln):
    idx = ln-1
    if idx<0 or idx>=len(plines): return None
    for i in range(idx,-1,-1):
        m=re.match(r"^(\s*)def (\w+)",plines[i])
        if m:
            indent=len(m.group(1)); start=i; name=m.group(2); end=len(plines)
            for j in range(start+1,len(plines)):
                l=plines[j]
                if not l.strip(): continue
                ci=len(l)-len(l.lstrip())
                if ci<=indent and l.lstrip().startswith(("def ","class ","@")): end=j; break
            return name,start,end
    return None

shas=[]
for line in git("log","--format=%H %s").splitlines():
    sha,_,subj=line.partition(" ")
    if re.search(r"\bfix(es|ed)?\b|\bbug\b",subj,re.I): shas.append((sha,subj))

STOPWORDS={"fix","fixes","fixed","bug","bugs","add","small","some","case","the","and",
    "measure","longer","make","improve","better"}
cases=[]
for sha,subj in shas:
    stat=git("show","--stat","--format=",sha)
    files=[l.split("|")[0].strip() for l in stat.splitlines() if "|" in l]
    lf=[f for f in files if LOGIC_RE.search(f)]
    if not lf: continue
    changed_funcs=set(); total=0; difftxt=""
    for f in lf:
        psrc=git("show",f"{sha}~1:{f}")
        if not psrc.strip(): continue
        plines=psrc.splitlines()
        d=git("show","--format=",sha,"--",f)
        difftxt+=d
        for m in re.finditer(r"@@ -(\d+)(?:,(\d+))? \+",d):
            s=int(m.group(1)); n=int(m.group(2) or "1"); total+=n
            for ln in range(s,s+n):
                ef=enclosing_func(plines,ln)
                if ef: changed_funcs.add(f.split("/")[-1]+":"+ef[0])
    if not changed_funcs: continue
    if total<2 or total>45: continue   # bug-sized, not a refactor
    # groundability: does the symptom carry any 6+ char token present in the codebase?
    toks=[w for w in re.findall(r"[a-z_]+",subj.lower()) if len(w)>=6 and w not in STOPWORDS]
    cases.append({"sha":sha,"symptom":subj,"changed_funcs":sorted(changed_funcs),
                  "n_changed":total,"symptom_tokens":toks,"groundable":bool(toks),
                  "fix_diff":difftxt.strip()[:7000]})

print(f"extracted {len(cases)} cases (groundable-symptom: {sum(c['groundable'] for c in cases)})",file=sys.stderr)
json.dump(cases,open("/tmp/aa_bench3_cases.json","w"),indent=1)
for c in cases:
    print(f"  {c['sha']} n={c['n_changed']:>2} g={int(c['groundable'])} funcs={len(c['changed_funcs'])} {c['symptom'][:52]}",file=sys.stderr)
