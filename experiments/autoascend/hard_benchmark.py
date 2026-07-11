"""Harder benchmark: IMPLICIT/INDIRECT, multi-function bugs (like the AA container)
to find where the LLM debugger actually breaks -- characterizing the real limit."""
import json, urllib.request
def llm(p, mt=300):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=100))["choices"][0]["message"]["content"]
CASES = [
 ("implicit_type_key",
  "def new_id():\n    global _c; _c += 1; return _c        # returns an int\n\ndef store(item):\n    item.id = new_id() if not item.name else item.name  # name is a str like '#0'\n    cache[item.id] = compute(item)\n\ndef lookup(item):\n    item.id = item.name if item.name else new_id()\n    if item.id in cache: item.content = cache[item.id]   # miss when types differ",
  "an item is re-processed every cycle -- cache never hits for named items, even after store().",
  ["type","int","str","name","key","mismatch","different type"]),
 ("stale_alias",
  "def snapshot(state):\n    saved = state.items        # same list object, not a copy\n    state.items.append('x')\n    return saved               # caller expects the pre-append list",
  "the returned 'saved' snapshot unexpectedly contains 'x' -- it reflects the later append.",
  ["alias","copy","same list","reference","mutat","[:]","list("]),
 ("cross_fn_offbyone",
  "def bounds(n):\n    return 0, n          # intended inclusive upper\n\ndef fill(n, arr):\n    lo, hi = bounds(n)\n    for i in range(lo, hi):   # range excludes hi -> misses index n-... \n        arr[i] = compute(i)\n    return arr",
  "fill(len(arr), arr) leaves the LAST element unset -- off by one, but bounds() looks correct.",
  ["range","excludes","hi","inclusive","off by one","bounds returns n"]),
 ("default_mutated_elsewhere",
  "CONFIG = {'retries': 3}\ndef run(opts=CONFIG):     # shares the module dict\n    opts['retries'] -= 1\n    return opts['retries']",
  "run() returns 2, then 1, then 0 across calls -- retries decreases globally.",
  ["shared","module","CONFIG","mutat","default","global","copy"]),
]
hit=0
for name, code, sym, keys in CASES:
    d = llm("Python bug across functions. Symptom: "+sym+"\n\nCode:\n"+code+"\n\nWhat is the root-cause bug? Be specific and brief.").lower()
    h = any(k.lower() in d for k in keys)
    print("%-24s -> %s"%(name, "FOUND" if h else "MISS"))
    hit += h
print("\nHARD (implicit/multi-fn): %d/%d found"%(hit,len(CASES)))
