import json, urllib.request
CODE = '''# AutoAscend fight_heur.py -- decides when to engrave Elbereth (which scares
# most monsters) as a defensive action:
def elbereth_action(agent, monsters):
    if agent.inventory.engraving_below_me.lower() == 'elbereth':
        return []
    adj_monsters_count = 0
    for monster in monsters:
        _, my, mx, mon, _ = monster   # mon has .mname and .mlet (monster class)
        if not adjacent((my, mx), (agent.pos)): continue
        adj_monsters_count += 1
    if agent.blstats.hitpoints < 30 and adj_monsters_count > 0:
        return [(priority, ('elbereth',))]   # engrave Elbereth and wait
    return []'''
PROMPT = ("AutoAscend (NetHack bot) dies on ~24% of games stuck in a loop: it repeatedly "
"engraves Elbereth in the dust and waits, but keeps taking damage and dies. The monsters "
"killing it are human (@) soldiers and its own hungry pet. (Fact: Elbereth scares most "
"monsters but does NOT scare @ humans, minotaurs, or your own pet.) Here is the code:\n\n"
+ CODE + "\n\nWhat is the bug and the fix? Be brief and specific.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
