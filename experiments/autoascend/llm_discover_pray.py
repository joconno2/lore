import json, urllib.request
CODE = '''# AutoAscend emergency_strategy (runs when the bot is in trouble):
if (
    (self.is_safe_to_pray(500) and
     (self.blstats.hitpoints < 1/5 * self.blstats.max_hitpoints or self.blstats.hitpoints < 6))
    or (self.is_safe_to_pray(400) and self.blstats.hunger_state >= Hunger.FAINTING)
):
    yield True
    self.pray()   # praying restores HP / cures starvation
    return
# is_safe_to_pray(timeout) only checks the PRAYER TIMEOUT (turns since the bot last
# prayed, i.e. whether the god is in a good mood). Praying takes ~4-10 game turns to
# complete, during which the bot takes no other action.'''
PROMPT = ("AutoAscend (NetHack bot) dies on 3% of games with the message 'Killed by "
"<a monster>, while praying'. In every case a hostile monster is ADJACENT and hitting "
"the bot when it starts praying (low HP), e.g. 'You begin praying... The master mind "
"flayer hits!' then it dies. Here is the code:\n\n" + CODE + "\n\nWhat is the bug and "
"the fix? Be specific and brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 320, "temperature": 0.2}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=100))["choices"][0]["message"]["content"])
