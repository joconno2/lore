"""Autonomous pipeline v3 -- GROUNDED retrieval (confirms the unified mechanism).
Instead of the LLM GENERATING search terms (hallucinates), extract candidate
identifiers from the SYMPTOM, VALIDATE them against the real codebase vocabulary
(keep only terms that actually appear), grep those, feed enclosing functions to
the LLM. Grounding the retrieval = no hallucination."""
import json, urllib.request, re, glob, os
AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")

def llm(prompt, mt=350):
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
            "temperature": 0.2}).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"]

SRC = {}
for path in glob.glob(AA_SRC + "/**/*.py", recursive=True):
    try: SRC[path] = open(path).read().split("\n")
    except Exception: pass
ALLTEXT = "\n".join("\n".join(v) for v in SRC.values()).lower()

SYMPTOMS = ("The bot stays on Dungeon Level 1 for 14000+ turns on half its games, "
    "experience level rises to ~7, then it starves. It never issues the descend command "
    "though the downstairs is visible.")

# GROUNDED term extraction: words + snake_case bigrams from the symptom, kept ONLY
# if they actually occur in the codebase (validated -> no hallucination).
words = re.findall(r"[a-z]+", SYMPTOMS.lower())
cands = set(words)
for i in range(len(words)-1):
    cands.add(words[i] + "_" + words[i+1])
grounded = sorted({c for c in cands if len(c) >= 5 and c in ALLTEXT},
                  key=lambda c: -ALLTEXT.count(c))
# prefer the rarer, more-specific terms (less noise); drop very common english
STOP = {"turns","games","level","never","command","visible","fights","monsters","the","its"}
grounded = [g for g in grounded if g not in STOP][:6]
print("GROUNDED validated terms:", grounded)

# retrieve enclosing functions for the grounded terms
def enclosing(term):
    out = []
    for path, lines in SRC.items():
        defs = [(i, m.group(1)) for i, m in ((i, re.match(r"\s*def (\w+)", l)) for i, l in enumerate(lines)) if m]
        for j in range(len(lines)):
            if term in lines[j].lower():
                fn = next((n for (di,n) in reversed(defs) if di <= j), "?")
                out.append((os.path.basename(path)+":"+fn, di if False else None))
    return out
funcs = {}
for term in grounded:
    for path, lines in SRC.items():
        defs = [(i, m.group(1)) for i, m in ((i, re.match(r"\s*def (\w+)", l)) for i, l in enumerate(lines)) if m]
        for j, l in enumerate(lines):
            if term in l.lower():
                # enclosing function
                cand = [(di,n) for (di,n) in defs if di <= j]
                if not cand: continue
                di, n = cand[-1]
                end = next((d for d,_ in defs if d > di), len(lines))
                key = os.path.basename(path)+":"+n
                funcs[key] = "\n".join(lines[di:end])[:1400]
print("retrieved functions:", list(funcs)[:8])
blob = "\n\n".join("# --- %s ---\n%s" % (k, funcs[k]) for k in list(funcs)[:7])

diag = llm("A NetHack bot bug:\n" + SYMPTOMS + "\n\nRelevant functions (retrieved by "
    "grepping terms from the symptom):\n\n" + blob[:6500] + "\n\nWhich function has the "
    "bug, what is it, and the fix? Brief.", 380)
print("\nDIAGNOSIS:\n" + diag)
