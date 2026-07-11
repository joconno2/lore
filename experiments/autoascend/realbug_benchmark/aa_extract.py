"""Extract real bug-fix benchmark cases from upstream AutoAscend history.

For each fix commit, find the PRIMARY changed function (most changed parent-side
lines) among agent-logic files. Keep the case if that one function holds >=60% of
the commit's changed logic lines (so the buggy function we hand the debugger really
contains the bug). Emit (symptom, buggy_func, fix_diff, changed_lines).

Unbiased within the structural filter; not hand-picked.
"""
import subprocess, re, json, sys
from collections import defaultdict

REPO = "/tmp/aa_upstream"
LOGIC = ("agent.py", "item.py", "fight_heur.py", "global_logic.py",
         "exploration_logic.py", "character.py", "combat", "level.py",
         "soko", "strategy", "inventory")
SKIP = ("visualize", "filter_for_vis", "stats", "summary.py", "muzero",
        "rl", "sim", "__init__", "logger", "server")

def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout

def is_logic(f):
    if any(s in f for s in SKIP): return False
    return f.endswith(".py") and any(k in f for k in LOGIC)

def enclosing_func(plines, lineno):
    idx = lineno - 1
    if idx >= len(plines) or idx < 0: return None
    for i in range(idx, -1, -1):
        m = re.match(r"^(\s*)def (\w+)", plines[i])
        if m:
            indent = len(m.group(1)); start = i; name = m.group(2)
            end = len(plines)
            for j in range(start + 1, len(plines)):
                l = plines[j]
                if l.strip() == "": continue
                ci = len(l) - len(l.lstrip())
                if ci <= indent and l.lstrip().startswith(("def ", "class ", "@")):
                    end = j; break
            return name, start, end
    return None

subjects = git("log", "--oneline").splitlines()
fix_shas = []
for line in subjects:
    sha, _, subj = line.partition(" ")
    if re.search(r"\bfix(es|ed)?\b|\bbug\b", subj, re.I):
        fix_shas.append((sha, subj))

cases = []
for sha, subj in fix_shas:
    stat = git("show", "--stat", "--format=", sha)
    files = [l.split("|")[0].strip() for l in stat.splitlines() if "|" in l]
    lf = [f for f in files if is_logic(f)]
    if not lf:
        continue
    # collect changed parent-side lines per file, map to enclosing function
    func_hits = defaultdict(list)   # (file,func,start,end) -> [linenos]
    total_changed = 0
    func_bodies = {}
    for f in lf:
        parent_src = git("show", f"{sha}~1:{f}")
        if not parent_src.strip(): continue
        plines = parent_src.splitlines()
        diff = git("show", "--format=", sha, "--", f)
        changed = []
        for m in re.finditer(r"@@ -(\d+)(?:,(\d+))? \+", diff):
            s = int(m.group(1)); n = int(m.group(2) or "1")
            changed.extend(range(s, s + n))
        total_changed += len(changed)
        for ln in changed:
            ef = enclosing_func(plines, ln)
            if ef:
                name, start, end = ef
                key = (f, name, start, end)
                func_hits[key].append(ln)
                func_bodies[key] = "\n".join(plines[start:end])
    if not func_hits or total_changed == 0:
        continue
    # primary function = most changed lines
    key = max(func_hits, key=lambda k: len(func_hits[k]))
    f, name, start, end = key
    frac = len(func_hits[key]) / total_changed
    body = func_bodies[key]
    nlines = body.count("\n") + 1
    if frac < 0.55 or nlines > 95 or nlines < 4:
        continue
    cases.append({
        "sha": sha, "symptom": subj, "file": f, "func": name,
        "func_start_line": start + 1, "func_lines": nlines,
        "primary_frac": round(frac, 2),
        "changed_lines": sorted(l - start for l in func_hits[key]),  # relative to func
        "buggy_func": body,
        "fix_diff": git("show", "--format=", sha, "--", f).strip()[:6000],
    })

print(f"extracted {len(cases)} cases", file=sys.stderr)
json.dump(cases, open("/tmp/aa_bench_cases.json", "w"), indent=1)
for c in cases:
    print(f"  {c['sha']} {c['file'].split('/')[-1]:>18}:{c['func']:<28} {c['func_lines']:>3}L frac={c['primary_frac']}  {c['symptom'][:52]}", file=sys.stderr)
