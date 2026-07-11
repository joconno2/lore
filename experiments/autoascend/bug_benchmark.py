"""Rigorous debugger benchmark: N known Python bug types (ground truth), each a
small function + symptom. Run the LLM debugger (given the function + symptom),
score whether it identifies the bug, classify by RECALL (known gotcha) vs
VALUE-TRACING (multi-step reasoning) to test the sharp-limit hypothesis."""
import json, urllib.request
def llm(p, mt=280):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=100))["choices"][0]["message"]["content"]

# (name, kind, code, symptom, key phrases that indicate a correct diagnosis)
CASES = [
 ("mutable_default","recall",
  "def add(x, acc=[]):\n    acc.append(x)\n    return acc",
  "add(1) returns [1], but a later add(2) returns [1, 2] -- the list persists across calls.",
  ["mutable default","default argument","shared","acc=[]","None"]),
 ("late_binding","recall",
  "fns = [lambda: i for i in range(3)]\nresult = [f() for f in fns]",
  "result is [2, 2, 2] but was expected to be [0, 1, 2].",
  ["late binding","closure","captured","default argument","i="]),
 ("is_vs_eq","tracing",
  "def check(x):\n    return x is 1000   # x may equal 1000",
  "check(1000) returns False even when x equals 1000 (works for small ints like 5).",
  ["is","identity","==","interning","small int"]),
 ("off_by_one","tracing",
  "def total(arr):\n    s = 0\n    for i in range(1, len(arr)):\n        s += arr[i]\n    return s",
  "total([10, 20, 30]) returns 50, but the expected sum is 60.",
  ["range(1","skips","index 0","first element","range(len"]),
 ("unbound_after_discard","tracing",
  "def any_close(pts):\n    return any(dist <= 1 for _, _, dist in pts)",
  "raises NameError: name 'dist' is not defined.",
  ["dist","unbound","discarded","_","unpack"]),
 ("type_key_mismatch","tracing",
  "cache = {}\ndef store(k):\n    cache[int(k)] = 1        # store under int\ndef check(k):\n    return str(k) in cache   # look up under str",
  "after store(5), check(5) returns False -- the value is never found.",
  ["int","str","key","mismatch","type","5 != '5'"]),
 ("and_or_precedence","tracing",
  "def ok(a, b):\n    return a or b and False",
  "ok(False, True) returns True, but was expected to be False.",
  ["precedence","and binds","or","parentheses"]),
 ("shadowed_builtin","recall",
  "def f(items):\n    list = items[0]\n    return list(items)   # call list()",
  "raises TypeError: 'int' object is not callable.",
  ["shadow","list","builtin","reassign","variable named list"]),
]
rec_hit=rec_n=tr_hit=tr_n=0
for name, kind, code, sym, keys in CASES:
    d = llm("Python bug. Symptom: "+sym+"\n\nCode:\n"+code+"\n\nWhat is the bug? Be specific and brief.").lower()
    hit = any(k.lower() in d for k in keys)
    print("%-22s [%s] -> %s" % (name, kind, "FOUND" if hit else "MISS"))
    if kind=="recall": rec_n+=1; rec_hit+=hit
    else: tr_n+=1; tr_hit+=hit
print("\nRECALL bugs:  %d/%d found | VALUE-TRACING bugs: %d/%d found | TOTAL: %d/%d"%(rec_hit,rec_n,tr_hit,tr_n,rec_hit+tr_hit,rec_n+tr_n))
