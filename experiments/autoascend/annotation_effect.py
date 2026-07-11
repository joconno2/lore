"""Controlled demo of the synthetic-overestimation effect: the SAME int-vs-str
type-mismatch bug, ANNOTATED vs UN-ANNOTATED. Isolates the annotation as the only
variable -> shows synthetic (annotated) benchmarks inflate debugger accuracy."""
import json, urllib.request
def llm(p, mt=260):
    r=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps({"model":"Qwen/Qwen2.5-14B-Instruct-AWQ","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":0.2}).encode(),headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=100))["choices"][0]["message"]["content"]

SYMPTOM = "After store(item), lookup(item) never finds the cached value for named items -- it re-computes every time."
KEYS = ["type","int","str","key","mismatch","different type","new_container_identifier returns","comment is a str"]

ANNOTATED = '''def new_container_identifier():
    global _c; _c += 1; return _c              # returns an int (0, 1, 2, ...)

def store(item):
    item.cid = new_container_identifier()      # cid is an int here
    cache[item.cid] = check(item)              # stored under int key

def lookup(item):
    item.cid = item.comment                    # comment is a str like "0" (from "#0")
    if item.cid in cache:                       # str key -- misses the int key
        item.content = cache[item.cid]'''

UNANNOTATED = '''def new_container_identifier():
    global _c; _c += 1; return _c

def store(item):
    item.cid = new_container_identifier()
    cache[item.cid] = check(item)

def lookup(item):
    item.cid = item.comment
    if item.cid in cache:
        item.content = cache[item.cid]'''

for label, code in [("ANNOTATED", ANNOTATED), ("UN-ANNOTATED", UNANNOTATED)]:
    d = llm("Python bug across functions. Symptom: "+SYMPTOM+"\n\nCode:\n"+code+"\n\nWhat is the root-cause bug? Be specific and brief.").lower()
    hit = any(k.lower() in d for k in KEYS)
    print("%-14s -> %s" % (label, "FOUND (type mismatch)" if hit else "MISS"))
    print("   ", d[:180].replace("\n"," "))
