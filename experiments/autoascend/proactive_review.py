"""Test PROACTIVE code review (no symptom) vs the symptom-driven debugger. Give the
LLM buggy AA functions with NO failure description, ask 'any bugs?' -- does it find
the known bug from code alone?"""
import json, urllib.request
def llm(p, mt=280):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=100))["choices"][0]["message"]["content"]
CASES = [
 ("is_safe_to_pray (missing adjacent-monster check)",
  "def is_safe_to_pray(self, limit=500):\n    return ((self.last_prayer_turn is None and self.blstats.time > 300) or\n            (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > limit))",
  ["adjacent","monster","nearby","praying takes","multi-turn","vulnerable","interrupt"]),
 ("elbereth_action (no immunity check)",
  "def elbereth_action(agent, monsters):\n    adj = sum(1 for m in monsters if adjacent(m, agent))\n    if agent.blstats.hitpoints < 30 and adj > 0:\n        return [('elbereth',)]  # engrave Elbereth and wait\n    return []",
  ["immune","@","human","scare","doesn't work","minotaur","pet","some monsters"]),
]
for name, code, keys in CASES:
    d = llm("Review this NetHack-bot function for any BUGS or issues (no other context "
        "given):\n\n"+code+"\n\nAre there bugs? Be specific and brief.").lower()
    hit = any(k.lower() in d for k in keys)
    print("%-45s -> %s"%(name[:45], "FOUND the real bug" if hit else "MISS (didn't flag it)"))
    print("   ", d[:160].replace("\n"," "))
