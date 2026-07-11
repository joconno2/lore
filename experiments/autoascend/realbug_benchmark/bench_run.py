"""Real-bug benchmark runner (runs ON trx, uses local vLLM Qwen-14B).

For each extracted upstream fix commit: checkout the PARENT (buggy) tree, run the
EXACT autonomous_debug5 pipeline (grounded+rarity retrieval -> BUG/FUNDAMENTAL gate
-> diagnosis) from the author's commit-subject symptom alone. Record what it
retrieved, its gate call, and its diagnosis. Scoring (localization vs target func,
diagnosis vs real fix) done afterward by hand from this dump.
"""
import json, urllib.request, re, glob, os, subprocess, sys

UP = os.path.expanduser("~/aa_upstream")
CASES = json.load(open("/tmp/aa_bench_cases.json"))

def llm(p, mt=350):
    r = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ",
            "messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),
        headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))["choices"][0]["message"]["content"]

def checkout_parent(sha):
    d = "/tmp/aacase"
    subprocess.run(f"rm -rf {d} && mkdir -p {d}", shell=True)
    subprocess.run(f"git -C {UP} archive {sha}~1 | tar -x -C {d}", shell=True)
    return d

def run_debugger(symptom, aa_src):
    """Exact debug5 logic, but returns dict of intermediate outputs for scoring."""
    SRC = {}
    for path in glob.glob(aa_src + "/**/*.py", recursive=True):
        try: SRC[path] = open(path).read().split("\n")
        except: pass
    ALL = "\n".join("\n".join(v) for v in SRC.values()).lower()
    S = symptom
    words = re.findall(r"[a-z]+", S.lower())
    cands = set(words) | {words[i]+"_"+words[i+1] for i in range(len(words)-1)}
    STOP = {"turns","games","level","never","the","its","then","dies","gets","other","kill",
            "while","bot","and","reaches","single","set","with","work","well","some","case",
            "being","blocked","measure","longer","fix","fixes","fixed","bug","add","small"}
    g = sorted({c for c in cands if len(c) >= 6 and c in ALL and c not in STOP},
               key=lambda c: ALL.count(c))[:5]
    funcs = {}
    for path, lines in SRC.items():
        defs = [(i, mm.group(1)) for i, mm in
                ((i, re.match(r"\s*def (\w+)", l)) for i, l in enumerate(lines)) if mm]
        for di, n in defs:
            end = next((d for d, _ in defs if d > di), len(lines))
            body = "\n".join(lines[di:end]).lower()
            sc = sum(body.count(x)/max(1, ALL.count(x)) for x in g)
            if sc > 0:
                funcs[os.path.basename(path)+":"+n] = (sc, "\n".join(lines[di:end])[:1200])
    top = sorted(funcs.items(), key=lambda x: -x[1][0])[:6]
    blob = "\n\n".join("# %s\n%s" % (k, v[1]) for k, v in top)[:5000]
    gate = llm("A NetHack bot (a strong competition winner) failure:\n"+S+
        "\n\nRelevant retrieved functions:\n"+blob+
        "\n\nIs this caused by a SPECIFIC FIXABLE BUG in one of these functions, or is it "
        "FUNDAMENTAL game difficulty (no specific bug)? Answer with ONLY the first word: "
        "BUG or FUNDAMENTAL, then one sentence why.", 120)
    diag = ""
    if gate.strip().upper().startswith("BUG"):
        diag = llm("Bug:\n"+S+"\n\nFunctions:\n"+blob+
            "\n\nWhich function, what bug, what fix? Brief.", 300)
    return {"search_terms": g, "retrieved": [k for k, _ in top],
            "gate": gate.strip()[:220], "diagnosis": diag.strip()[:600]}

results = []
for i, c in enumerate(CASES):
    print(f"[{i+1}/{len(CASES)}] {c['sha']} {c['symptom'][:50]}", file=sys.stderr)
    try:
        aa_src = checkout_parent(c["sha"])
        out = run_debugger(c["symptom"], aa_src)
    except Exception as e:
        out = {"error": str(e)}
    target = os.path.basename(c["file"]) + ":" + c["func"]
    out.update({"sha": c["sha"], "symptom": c["symptom"], "target": target,
                "loc_hit": target in out.get("retrieved", [])})
    results.append(out)
    print("   gate:", out.get("gate","")[:80], "| loc_hit:", out["loc_hit"], file=sys.stderr)

json.dump(results, open("/tmp/aa_bench_results.json", "w"), indent=1)
print("DONE ->", "/tmp/aa_bench_results.json", file=sys.stderr)
