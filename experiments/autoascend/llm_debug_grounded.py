import json, urllib.request
CODE = '''# AutoAscend global_logic.py -- milestone advance conditions (the bot advances
# to the next milestone / leaves the current objective when `condition` is True):
if self.milestone == Milestone.BE_ON_FIRST_LEVEL:
    condition = lambda: self.agent.blstats.experience_level >= 8
elif self.milestone == Milestone.FIND_GNOMISH_MINES:
    condition = lambda: ...
# (BE_ON_FIRST_LEVEL is the starting milestone; the bot stays on Dungeon Level 1
#  pursuing it until `condition` becomes True, then advances and descends.)'''
PROMPT = ("AutoAscend (NetHack bot) stays on Dungeon Level 1 for 14000+ turns on 52% "
"of games, XP level rises to ~7, then it starves. It never descends though the "
"downstairs is visible. Here is the relevant code:\n\n" + CODE + "\n\nWhat is the "
"specific bug causing it to never leave DL1 on these games, and the fix? Be brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 350, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
