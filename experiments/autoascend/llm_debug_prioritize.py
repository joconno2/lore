import json, urllib.request
PROMPT = ("Here is the aggregate outcome data for base AutoAscend (symbolic NetHack bot) "
"over 300 games. Depth reached: median Dungeon Level 2, but ~50% of games only reach "
"DL1. Death causes: 41% starvation, 48% combat, rest misc. Among the starvation deaths, "
"the bot survived a median of ~14000 game turns and its experience level rose to ~7, but "
"it stayed on DL1 the whole time. You want to maximize how DEEP the bot gets (ascension "
"progress). What is the SINGLE highest-impact fixable problem to investigate first, and "
"why? Be concrete and brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
