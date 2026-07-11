"""Autonomous debug pipeline v2 -- GROUNDED retrieval (fixes v1's keyword
hallucination). Stage-1: give the LLM the ACTUAL list of AA function names and
let it SELECT relevant ones (grounded -> no hallucination, mirrors the wish
retrieval fix). Then retrieve those function bodies + diagnose."""
import json, urllib.request, re, glob, os, sys
AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")

def llm(prompt, max_tokens=300):
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"]

# index: {file:func -> body}
INDEX = {}
NAMES = []
for path in glob.glob(AA_SRC + "/**/*.py", recursive=True):
    try: lines = open(path).read().split("\n")
    except Exception: continue
    defs = [(i, re.match(r"(\s*)def (\w+)", l)) for i, l in enumerate(lines)]
    defs = [(i, m.group(2)) for i, m in defs if m]
    for idx, (i, name) in enumerate(defs):
        end = defs[idx+1][0] if idx+1 < len(defs) else len(lines)
        key = os.path.basename(path) + ":" + name
        INDEX[key] = "\n".join(lines[i:end])[:1400]
        NAMES.append(key)
print("indexed", len(NAMES), "functions")

SYMPTOMS = ("The bot stays on Dungeon Level 1 for 14000+ turns on half its games, its "
    "experience level rises to ~7, then it starves. It never issues the descend command "
    "though the downstairs is visible. It fights early monsters the whole time.")

# Stage-1 GROUNDED: LLM selects from real function names
namelist = ", ".join(sorted(set(n.split(":")[1] for n in NAMES)))
sel = llm("A NetHack bot bug:\n" + SYMPTOMS + "\n\nHere is the FULL list of function "
    "names in its source (pick ONLY from these -- do not invent names):\n" + namelist[:8000] +
    "\n\nWhich 3-6 functions are most likely relevant to this bug? Reply ONLY their exact "
    "names, comma-separated.", 100)
picked = [s.strip().strip("`'\"()") for s in re.split(r"[,\n]", sel) if s.strip()]
print("STAGE1 picked:", picked)
# retrieve
blob_keys = [k for k in NAMES if k.split(":")[1] in picked]
print("STAGE2 retrieved:", blob_keys[:8])
blob = "\n\n".join("# --- %s ---\n%s" % (k, INDEX[k]) for k in blob_keys[:8])

diag = llm("A NetHack bot bug:\n" + SYMPTOMS + "\n\nRelevant functions from its source:\n\n"
    + blob[:6000] + "\n\nWhich function has the bug, what is it, and the fix? Brief.", 350)
print("\nSTAGE3 diagnosis:\n" + diag)
