"""Minimal AUTONOMOUS symbolic-agent debug pipeline (RAG-lite, no embeddings):
symptoms -> (LLM: search keywords) -> (grep AA source: enclosing functions) ->
(LLM: diagnose the retrieved code). Tests the full pipeline WITHOUT pre-selecting
the buggy function -- the caveat the earlier tests couldn't close."""
import json, urllib.request, re, glob, os, sys

AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")

def llm(prompt, max_tokens=300):
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"]

def enclosing_functions(keywords):
    """Return function bodies in AA source containing any keyword."""
    funcs = {}
    for path in glob.glob(AA_SRC + "/**/*.py", recursive=True):
        try: lines = open(path).read().split("\n")
        except Exception: continue
        # find def lines and their bodies (by indentation)
        defs = [(i, re.match(r"(\s*)def (\w+)", l)) for i, l in enumerate(lines)]
        defs = [(i, m.group(1), m.group(2)) for i, m in defs if m]
        for idx, (i, indent, name) in enumerate(defs):
            end = defs[idx+1][0] if idx+1 < len(defs) else len(lines)
            body = "\n".join(lines[i:end])
            if any(kw.lower() in body.lower() for kw in keywords):
                score = sum(body.lower().count(kw.lower()) for kw in keywords)
                key = os.path.basename(path) + ":" + name
                if key not in funcs or funcs[key][0] < score:
                    funcs[key] = (score, body[:1400])
    top = sorted(funcs.items(), key=lambda x: -x[1][0])[:6]
    return top

SYMPTOMS = sys.argv[1] if len(sys.argv) > 1 else (
    "The bot stays on Dungeon Level 1 for 14000+ turns on half its games, its "
    "experience level rises to ~7, then it starves. It never issues the descend "
    "command though the downstairs is visible. It fights early monsters the whole time.")

# Stage 1: LLM -> keywords
kw_raw = llm("A NetHack bot has this bug:\n" + SYMPTOMS + "\n\nList 4-8 short code "
    "SEARCH KEYWORDS (identifiers/terms likely in the relevant source functions) to grep "
    "for. Reply ONLY a comma-separated list, no prose.", 80)
keywords = [k.strip().strip("`'\"") for k in re.split(r"[,\n]", kw_raw) if k.strip()][:8]
print("STAGE1 keywords:", keywords)

# Stage 2: grep retrieval
top = enclosing_functions(keywords)
print("STAGE2 retrieved functions:", [k for k, _ in top])
blob = "\n\n".join("# --- %s ---\n%s" % (k, v[1]) for k, v in top)

# Stage 3: LLM diagnose
diag = llm("A NetHack bot bug:\n" + SYMPTOMS + "\n\nHere are the most relevant functions "
    "retrieved from its source:\n\n" + blob[:6000] + "\n\nWhich function has the bug, what "
    "is it, and the fix? Be specific and brief.", 400)
print("\nSTAGE3 diagnosis:\n" + diag)
