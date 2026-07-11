"""Direct LLM-value test at the wish decision: for several inventory states, does
the adaptive LLM pick the same wish as the adaptive expert RULE? Agreement => null
(the LLM adds nothing a rule doesn't). Disagreement => investigate which is better."""
import json, oracle
STATES = [
    {"have": [], "depth": 20},
    {"have": ["gray dragon scale mail"], "depth": 20},
    {"have": ["gray dragon scale mail", "amulet of reflection"], "depth": 20},
    {"have": ["gray dragon scale mail", "amulet of reflection", "ring of free action"], "depth": 20},
    {"have": ["amulet of reflection"], "depth": 20},
    {"have": ["ring of free action", "fire resistance"], "depth": 20},
    {"have": ["gray dragon scale mail", "ring of free action"], "depth": 20},
]
agree = 0
for st in STATES:
    m = oracle.query_wish(st, mock=True)["wish"]
    l = oracle.query_wish(st, mock=False)
    lw = l.get("wish", "?")
    same = (m.split()[-2:] == lw.split()[-2:]) or (m.lower() in lw.lower()) or (lw.lower() in m.lower())
    agree += 1 if same else 0
    print("have=%-60s | RULE: %-32s | LLM: %-32s | %s" % (
        str(st["have"])[:58], m, lw[:32], "SAME" if same else "DIFF"))
print("\nAGREEMENT: %d/%d" % (agree, len(STATES)))
