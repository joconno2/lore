"""Closed autonomous debug->repair->TEST loop: generate a patch, unit-test it
(exec + call with a mock), feed any error back to the LLM, regenerate until it
passes. Demonstrates that a test-in-the-loop makes autonomous repair CONVERGE
(catching the subtle bugs one-shot codegen produces)."""
import json, urllib.request, re, types
def llm(p, mt=300):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=100))["choices"][0]["message"]["content"]

class Blstats: pass
class MockAgent:
    def __init__(self, monsters):
        self.blstats=Blstats(); self.blstats.time=1000; self.blstats.y=10; self.blstats.x=10
        self.last_prayer_turn=None; self._monsters=monsters
    def get_visible_monsters(self): return self._monsters  # (dist,y,x,permonst,glyph)

def unit_test(fn):
    # adjacent hostile at (10,11) -> must be UNSAFE (False); none -> SAFE (timeout ok)
    a_adj=MockAgent([(1,10,11,None,0)]); a_none=MockAgent([])
    r_adj=fn(a_adj,500); r_none=fn(a_none,500)
    if r_adj is not False: return "FAIL: returned %r with an adjacent monster (should be False/unsafe)"%r_adj
    if r_none is not True: return "FAIL: returned %r with no monsters at turn 1000 (should be True/safe)"%r_none
    return None

BASE = ("Bug: is_safe_to_pray lets the bot pray while a hostile monster is adjacent (it dies "
"mid multi-turn prayer). Fix it to ALSO return False if any monster from get_visible_monsters() "
"-- tuples (distance, y, x, permonst, glyph) -- is adjacent (chebyshev distance <=1 from "
"self.blstats.y/x). Keep the timeout logic (safe if last_prayer_turn is None and time>300, or "
"time-last_prayer_turn>limit). Output ONLY the complete is_safe_to_pray(self, limit=500) method, "
"valid Python, no markdown.")
feedback=""; 
for it in range(5):
    out=llm(BASE+feedback, 300)
    out=re.sub(r"```\w*","",out).replace("```","").strip()
    # extract the def
    m=re.search(r"(def is_safe_to_pray.*)", out, re.S)
    code=m.group(1) if m else out
    ns={}
    try:
        exec("import math\n"+code, ns)
        fn=ns["is_safe_to_pray"]
        err=unit_test(fn)
    except Exception as e:
        err="RUNTIME ERROR: %s: %s"%(type(e).__name__, str(e)[:80])
    print("--- iter %d: %s"%(it, "PASS" if err is None else err))
    if err is None:
        print("\nCONVERGED PATCH:\n"+code); break
    feedback="\n\nYour previous attempt FAILED this test: "+err+"\nFix it. Output ONLY the method."
