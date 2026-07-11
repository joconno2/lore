import json, urllib.request
CODE = '''# a test harness that exec's an LLM-generated function then calls it:
ns = {}
exec("import math\\n" + code, ns)
fn = ns["is_safe_to_pray"]          # def is_safe_to_pray(self, limit=500): ...
fn = types.FunctionType(fn.__code__, ns)   # rebind globals to ns
err = unit_test(fn)                 # unit_test calls: fn(mock_agent)'''
PROMPT = ("A Python test harness fails with: TypeError: is_safe_to_pray() missing 1 required "
"positional argument: 'limit' -- even though the original function is defined as "
"`def is_safe_to_pray(self, limit=500):` WITH a default for limit. The harness exec's the "
"function then rebinds it before calling. Here is the harness:\n\n" + CODE + "\n\nWhy does "
"the call lose the default for `limit`, and what's the fix? Be specific and brief.")
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":PROMPT}],
        "max_tokens":300,"temperature":0.2}).encode(), headers={"Content-Type":"application/json"})
print(json.load(urllib.request.urlopen(req,timeout=100))["choices"][0]["message"]["content"])
