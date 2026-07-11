import json, urllib.request
PROMPT = ("AutoAscend is a strong symbolic NetHack bot (a NeurIPS competition winner). "
"On the games where it reaches Dungeon Level 8-10, it dies to combat against a DIVERSE "
"set of mid-game monsters -- leocrotta, lynx, mumak, soldier ant, master mind flayer, "
"gargoyle, werewolf -- at experience level ~9, with NO single dominant killer or repeated "
"pattern. Its early game and descent logic work well. Is there a specific fixable BUG "
"causing these deaths, or is this fundamental difficulty? Be honest and brief -- say so "
"if there's no specific bug.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
