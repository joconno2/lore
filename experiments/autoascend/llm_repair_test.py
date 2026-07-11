import json, urllib.request, re
CODE = '''class Agent:
    def is_safe_to_pray(self, limit=500):
        return (
            (self.last_prayer_turn is None and self.blstats.time > 300) or
            (self.last_prayer_turn is not None and self.blstats.time - self.last_prayer_turn > limit)
        )
    def get_visible_monsters(self):
        """returns list of (distance, y, x, permonst, glyph) for HOSTILE monsters"""
        ...'''
PROMPT = ("Bug: the bot prays (a multi-turn action) while a hostile monster is ADJACENT, "
"and dies mid-prayer. Fix: is_safe_to_pray should also return False if any hostile monster "
"is adjacent (chebyshev distance <=1) using get_visible_monsters() and self.blstats.y/x. "
"Here is the code:\n\n" + CODE + "\n\nOutput ONLY the complete corrected is_safe_to_pray "
"method as valid Python (no prose, no markdown fences).")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":PROMPT}],
        "max_tokens":300,"temperature":0.2}).encode(), headers={"Content-Type":"application/json"})
out = json.load(urllib.request.urlopen(req,timeout=100))["choices"][0]["message"]["content"]
out = re.sub(r"^```\w*|```$", "", out.strip(), flags=re.MULTILINE).strip()
print("=== GENERATED PATCH ===\n"+out)
# verify: valid Python? contains the key elements?
import ast
try:
    ast.parse(out.replace("self.", "self_").replace("get_visible_monsters()", "gvm()") if False else "def _wrap(self):\n"+"\n".join("    "+l for l in out.split("\n")) if not out.startswith("def") else out)
    syntax="VALID"
except SyntaxError as e:
    try: ast.parse(out); syntax="VALID"
    except SyntaxError as e2: syntax="SYNTAX ERROR: "+str(e2)[:60]
has_monsters = "get_visible_monsters" in out
has_adjacent = ("<= 1" in out or "<=1" in out or "abs(" in out)
has_timeout = "last_prayer_turn" in out
print("\n=== VERIFY ===")
print("syntax:", syntax, "| checks adjacent monsters:", has_monsters and has_adjacent, "| keeps timeout:", has_timeout)
