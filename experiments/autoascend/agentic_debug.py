"""Agentic autonomous debug loop (ReAct-style): the LLM iteratively GREPs and
READs the real AA source to HUNT the bug -- following the logic like a human,
which one-shot keyword/name retrieval couldn't. Tools: grep(pattern), read(func).
No embeddings needed. Tests whether agentic search finds the INDIRECT bug
(stuck-at-DL1 -> the milestone experience_level>=8 condition)."""
import json, urllib.request, re, glob, os
AA_SRC = os.environ.get("AA_SRC", "/workspace/autoascend")

def llm(prompt, max_tokens=350):
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"]

# index functions
INDEX = {}
FILES = {}
for path in glob.glob(AA_SRC + "/**/*.py", recursive=True):
    try: lines = open(path).read().split("\n")
    except Exception: continue
    FILES[os.path.basename(path)] = lines
    defs = [(i, m.group(2)) for i, m in ((i, re.match(r"\s*def (\w+)", l)) for i, l in enumerate(lines)) if m] if False else []
    defs = [(i, re.match(r"\s*def (\w+)", l)) for i, l in enumerate(lines)]
    defs = [(i, m.group(1)) for i, m in defs if m]
    for idx, (i, name) in enumerate(defs):
        end = defs[idx+1][0] if idx+1 < len(defs) else len(lines)
        INDEX.setdefault(name, "\n".join(lines[i:end])[:1600])

def tool_grep(pattern):
    hits = []
    for fname, lines in FILES.items():
        for i, l in enumerate(lines):
            if pattern.lower() in l.lower():
                hits.append("%s:%d: %s" % (fname, i, l.strip()[:90]))
    return "\n".join(hits[:25]) or "(no matches)"

def tool_read(func):
    return INDEX.get(func, "(no such function: %s)" % func)

SYMPTOMS = ("The bot stays on Dungeon Level 1 for 14000+ turns on half its games, "
    "experience level rises to ~7, then it starves. It never issues the descend command "
    "though the downstairs is visible.")

transcript = ""
for step in range(8):
    p = ("You are debugging a NetHack bot (AutoAscend) by searching its source. Bug:\n"
        + SYMPTOMS + "\n\nSo far:\n" + (transcript or "(nothing yet)") + "\n\n"
        "Choose ONE next action. Reply EXACTLY one line:\n"
        "GREP <text>   (search source for a substring)\n"
        "READ <func>   (show a function body)\n"
        "DIAGNOSE <the bug and fix>   (only when you've found it)")
    action = llm(p, 200).strip().split("\n")[0]
    print(">>> STEP %d: %s" % (step, action[:100]))
    if action.upper().startswith("DIAGNOSE"):
        print("\nFINAL:", action[8:].strip()[:600]); break
    elif action.upper().startswith("GREP"):
        arg = action[4:].strip().strip("'\"`")
        res = tool_grep(arg)
        transcript += "\nGREP %s ->\n%s\n" % (arg, res[:800])
    elif action.upper().startswith("READ"):
        arg = action[4:].strip().strip("'\"`()")
        res = tool_read(arg)
        transcript += "\nREAD %s ->\n%s\n" % (arg, res[:1000])
    else:
        transcript += "\n(unparsed action: %s)\n" % action[:60]
