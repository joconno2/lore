"""Fair test of the FULL premise: LLM + RETRIEVAL grounding. Give the LLM the valid
wishable-items list + their properties in the prompt (what the EC-tuned retrieval
interface is meant to supply). Does grounding fix the 71% hallucination and let the
LLM match the expert rule? (parity=null) or does it still fail?"""
import json, oracle
GROUND = (
    "\n\nVALID survival wishes and the property each grants (wish ONLY from these):\n"
    "- gray dragon scale mail: magic resistance + AC\n"
    "- amulet of reflection: reflection\n"
    "- ring of free action: free action (anti-paralysis)\n"
    "- speed boots: intrinsic speed\n"
    "- ring of conflict: crowd control\n"
    "- cloak of magic resistance: magic resistance (redundant if gray dragon scale mail)\n"
    "- ring of fire resistance / ring of cold resistance: that resistance\n"
    "- amulet of life saving: revive once on death\n"
    "There is NO 'wand of magic resistance', NO 'wand of speed boost'. Wish EXACT item names above.")
oracle.WISH_SYSTEM = oracle.WISH_SYSTEM + GROUND
HAVE_SETS=[[],["gray dragon scale mail"],["amulet of reflection"],["ring of free action"],
    ["gray dragon scale mail","amulet of reflection"],["gray dragon scale mail","ring of free action"],
    ["amulet of reflection","ring of free action"],["gray dragon scale mail","amulet of reflection","ring of free action"],
    ["ring of free action","fire resistance"],["speed boots"],["gray dragon scale mail","speed boots"],
    ["cloak of magic resistance"],["amulet of reflection","cloak of magic resistance"],
    ["gray dragon scale mail","amulet of reflection","ring of free action","speed boots"]]
VALID=["gray dragon scale","silver dragon scale","amulet of reflection","amulet of life saving","ring of free action",
    "ring of conflict","ring of fire resistance","ring of cold resistance","speed boots","cloak of magic resistance",
    "cloak of protection","gauntlets of power","wand of death","wand of digging","bag of holding","unicorn horn"]
agree=halluc=0;N=len(HAVE_SETS)
for have in HAVE_SETS:
    st={"have":have,"depth":25}
    m=oracle.query_wish(st,mock=True)["wish"]; lw=oracle.query_wish(st,mock=False).get("wish","?")
    same=(m.lower() in lw.lower()) or (lw.lower() in m.lower()) or (m.split()[-2:]==lw.split()[-2:])
    v=any(x in lw.lower() for x in VALID)
    agree+=1 if same else 0; halluc+=0 if v else 1
    print("have=%-50s RULE=%-28s LLM=%-28s %s%s"%(str(have)[:48],m[:28],lw[:28],"=" if same else "X","" if v else " HALLUC"))
print("\nGROUNDED LLM -- AGREEMENT: %d/%d   HALLUCINATION: %d/%d"%(agree,N,halluc,N))
