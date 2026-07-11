import json, glob, collections
def kf(r): return str(r.get("killer")).strip()
R = [json.load(open(f)) for f in glob.glob("be_engine_*.json")]
wand = [r for r in R if r.get("death_cat") == "combat" and kf(r) == "a wand"]
print("killed by 'a wand' n=", len(wand))
for r in wand:
    print("seed", r.get("seed"), "xl", r.get("xl"), "dl", r.get("max_depth"), "| end:", str(r.get("end_reason"))[:80])
    print("   msgs:", str(r.get("msgs_tail"))[:160])
# also kitten
kit = [r for r in R if r.get("death_cat") == "combat" and kf(r) in ("a kitten", "a large kitten", "a housecat")]
print("\nkilled by own pet n=", len(kit))
for r in kit[:6]:
    print("seed", r.get("seed"), "xl", r.get("xl"), "| end:", str(r.get("end_reason"))[:70], "| msgs:", str(r.get("msgs_tail"))[:110])
