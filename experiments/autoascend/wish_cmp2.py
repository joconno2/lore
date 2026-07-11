"""Harden the wish capstone: more inventory states + a VALIDITY check (is the LLM's
wish even a real NetHack item?). Quantifies agreement-with-rule and hallucination
rate -- the publishable negative result."""
import json, oracle, itertools
# a (non-exhaustive) set of real wishable survival items for a validity check
VALID = ["gray dragon scale mail","silver dragon scale mail","amulet of reflection",
    "amulet of life saving","ring of free action","ring of conflict","ring of slow digestion",
    "ring of fire resistance","ring of cold resistance","ring of shock resistance","ring of teleport control",
    "speed boots","boots of speed","cloak of magic resistance","cloak of protection","gauntlets of power",
    "helm of brilliance","wand of death","wand of digging","wand of teleportation","wand of cancellation",
    "bag of holding","unicorn horn","luckstone","scroll of genocide","blindfold","towel","magic marker","tinning kit"]
def valid(w):
    wl = w.lower()
    return any(v in wl for v in VALID)
HAVE_SETS = [[], ["gray dragon scale mail"], ["amulet of reflection"], ["ring of free action"],
    ["gray dragon scale mail","amulet of reflection"], ["gray dragon scale mail","ring of free action"],
    ["amulet of reflection","ring of free action"], ["gray dragon scale mail","amulet of reflection","ring of free action"],
    ["ring of free action","fire resistance"], ["speed boots"], ["gray dragon scale mail","speed boots"],
    ["cloak of magic resistance"], ["amulet of reflection","cloak of magic resistance"],
    ["gray dragon scale mail","amulet of reflection","ring of free action","speed boots"]]
agree=halluc=0; N=len(HAVE_SETS)
for have in HAVE_SETS:
    st={"have":have,"depth":25}
    m=oracle.query_wish(st,mock=True)["wish"]
    lw=oracle.query_wish(st,mock=False).get("wish","?")
    same=(m.lower() in lw.lower()) or (lw.lower() in m.lower()) or (m.split()[-2:]==lw.split()[-2:])
    v=valid(lw)
    agree+= 1 if same else 0; halluc+= 0 if v else 1
    print("have=%-52s RULE=%-30s LLM=%-30s %s%s"%(str(have)[:50],m[:30],lw[:30],"=" if same else "X"," HALLUC" if not v else ""))
print("\nAGREEMENT with rule: %d/%d   LLM HALLUCINATION (invalid item): %d/%d"%(agree,N,halluc,N))
