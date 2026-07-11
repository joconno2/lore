"""For each AA v2 case, extract the POST-FIX (correct) version of the buggy function
(at the commit sha, by name). Used to test discrimination: given FIXED code + the
original symptom, does the debugger fabricate a bug (false positive) or recognize the
code is already correct?"""
import subprocess, re, json, sys
REPO="/tmp/aa_upstream"
def git(*a): return subprocess.run(["git","-C",REPO,*a],capture_output=True,text=True).stdout
def func_body(src_lines, name):
    for i,l in enumerate(src_lines):
        m=re.match(r"^(\s*)def "+re.escape(name)+r"\b",l)
        if m:
            ind=len(m.group(1)); end=len(src_lines)
            for j in range(i+1,len(src_lines)):
                x=src_lines[j]
                if not x.strip(): continue
                ci=len(x)-len(x.lstrip())
                if ci<=ind and x.lstrip().startswith(("def ","class ","@")): end=j; break
            return "\n".join(src_lines[i:end])
    return None
CASES=json.load(open("/tmp/aa_bench_cases.json"))
out=[]
for c in CASES:
    fixed_src=git("show",f"{c['sha']}:{c['file']}").splitlines()
    fb=func_body(fixed_src,c["func"])
    if fb and fb.strip()!=c["buggy_func"].strip():
        out.append({"sha":c["sha"],"symptom":c["symptom"],"func":c["func"],"fixed_func":fb})
json.dump(out,open("/tmp/aa_fixed_cases.json","w"),indent=1)
print(f"extracted {len(out)} fixed functions (differ from buggy)",file=sys.stderr)
