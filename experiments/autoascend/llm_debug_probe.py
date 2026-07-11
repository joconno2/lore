import json, urllib.request
SYMPTOMS = ("You are debugging AutoAscend, a symbolic NetHack bot. Observed failure "
"(52% of games): the bot stays on Dungeon Level 1 for 14000+ turns, its experience "
"level rises to ~7, HP stays full, hunger fine at first, then it starves to death. "
"The downstairs is visible on the map from turn 1, but the bot issues the descend "
"command ZERO times. It writes Elbereth and fights early monsters the whole time. "
"What is the most likely ROOT CAUSE in the bot's logic, and what specific code would "
"you inspect to confirm it? Be concrete and brief.")
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                     "messages": [{"role": "user", "content": SYMPTOMS}],
                     "max_tokens": 400, "temperature": 0.3}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=100))
print(r["choices"][0]["message"]["content"])
