import json, urllib.request
CODE = '''# AutoAscend melee_monster_priority (higher priority => more likely to melee this
# monster). Floating eye is in ONLY_RANGED_SLOW_MONSTERS:
if mon.mname in ONLY_RANGED_SLOW_MONSTERS:
    if not consider_melee_only_ranged_if_hp_full(agent, monster):  # False for floating eye
        ret -= 100                       # discourage meleeing it
        if mon.mname == 'floating eye':
            ret -= 10                    # extra discouragement (total -110)
# ret is a PRIORITY score; the bot picks the highest-priority available action.
# NetHack fact: meleeing a floating eye triggers its passive gaze -> the bot is
# PARALYZED (frozen) for many turns, unable to act; any other monster then kills it.'''
PROMPT = ("AutoAscend (NetHack bot) still dies ~3% of games 'frozen by a floating eye's "
"gaze' -- the logs show 'You hit the floating eye! You are frozen!' and then OTHER "
"adjacent monsters (goblin, killer bee, gecko) kill it while frozen. The code below "
"already applies a -110 priority penalty to meleeing a floating eye, yet it still melees "
"them in these multi-monster situations. Here's the code:\n\n" + CODE + "\n\nWhy does it "
"STILL melee the floating eye despite the penalty, and what's the robust fix? Be brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 350, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
