"""Hand-scored verdicts on the 15 real-bug diagnoses (judge: Opus, ground-truth
diffs in hand). Each verdict: diagnosis vs the actual fix.

CLEAN = the author symptom genuinely describes the extracted function's bug (fair
diagnosis test). BUNDLED = commit bundled several changes; the subject describes a
DIFFERENT change than the primary-function my extractor picked (symptom/function
mismatch -> unfair, excluded from the clean rate but reported).

verdict: CORRECT = named the real buggy locus AND the correct fix nature.
         PARTIAL = right function/region, fix under-specified or echoed symptom.
         WRONG   = wrong function, fabricated bug, or false-declined a real bug.
"""
import json
V = {
 "073d770": ("CLEAN","CORRECT","named the exact cyclic-panic assert + correct fix direction (replace brittle assert w/ controlled quit)"),
 "bf8617a": ("CLEAN","WRONG","wrong function (melee_monster_priority not get_priorities); fabricated a 'gas-spore-trapped condition'; real fix was the actions-emptiness guard"),
 "ec3cc35": ("BUNDLED","WRONG","symptom 'Fix RL' ungroundable (0 search terms) -> generic RL boilerplate"),
 "88a1b9b": ("BUNDLED","WRONG","subject 'monster tracker' but extractor picked gather_items (.before chaining) -> mismatch; fabricated invisible-monster filter bug"),
 "bad5143": ("CLEAN","PARTIAL","echoed the symptom's own hint ('atom operation'); right area, named inner action_generator not the outer loop; added no diagnostic value beyond the symptom"),
 "6bd6eee": ("BUNDLED","WRONG","symptom 'Fixes' ungroundable -> hallucinated a calculate_area example, unrelated"),
 "3b25d57": ("BUNDLED","WRONG","subject 'pyinstrument total time' but extractor picked move() (monster_tracker mask) -> mismatch; diagnosed profiling, real fix was a glyph->mask change"),
 "195eaad": ("CLEAN","WRONG","0 groundable search terms -> vague 'adjust the formulas'; missed the MONK-role skip that is the actual fix"),
 "bde22a3": ("CLEAN","CORRECT","named go_to_item_to_pickup + the empty-items/empty-mask edge case = the exact fix"),
 "4b536a3": ("BUNDLED","PARTIAL","subject names get_available_actions, extractor picked fight2 (big refactor); return-in-loop guess plausible but not the real restructure"),
 "e3328d1": ("CLEAN","WRONG","localized to solve_sokoban_strategy but diagnosis is vague 'the logic is incomplete/incorrect' -- no specific bug (map-alignment rewrite)"),
 "179a676": ("BUNDLED","WRONG","'forking, agent reloading' infra; 0 search terms -> generic boilerplate"),
 "0114ad4": ("CLEAN","WRONG","localized to solve_sokoban_strategy but FABRICATED a syntax error (mask=boulder_map[offse]) that isn't the bug; real fix was trap-detection via G.TRAPS not char '^'"),
 "326741c": ("CLEAN","WRONG","gate FALSE-DECLINED a real bug as FUNDAMENTAL; real fix filters acid/poison corpses by race"),
 "97a231d": ("CLEAN","PARTIAL","symptom only named the function; still localized the --More-- parse loop + right problem area, but fix under-specified (large rewrite)"),
}
R = {r["sha"]: r for r in json.load(open("/tmp/aa_bench_results.json"))}
clean = {k:v for k,v in V.items() if v[0]=="CLEAN"}
def rate(subset, pred):
    return sum(1 for k in subset if pred(V[k][1]))
loc = sum(1 for k in V if R[k].get("loc_hit"))
loc_clean = sum(1 for k in clean if R[k].get("loc_hit"))
print(f"N total = {len(V)}   N clean (fair) = {len(clean)}   N bundled = {len(V)-len(clean)}")
print()
print("LOCALIZATION (target func in top-6 retrieved):")
print(f"  all-15: {loc}/15 = {loc/15:.0%}   clean-9: {loc_clean}/{len(clean)} = {loc_clean/len(clean):.0%}")
print("  loc-hit cases:", [k for k in V if R[k].get('loc_hit')])
print()
print("DIAGNOSIS:")
for label, pred in [("CORRECT (strict)", lambda v: v=="CORRECT"),
                    ("CORRECT+PARTIAL (lenient)", lambda v: v in ("CORRECT","PARTIAL"))]:
    a = rate(V, pred); c = rate(clean, pred)
    print(f"  {label:26} all-15: {a}/15={a/15:.0%}   clean-9: {c}/{len(clean)}={c/len(clean):.0%}")
print()
print("Every CORRECT diagnosis was also a localization hit:",
      all(R[k].get("loc_hit") for k in V if V[k][1]=="CORRECT"))
print()
print("PER-CASE:")
for k,(cln,verd,why) in V.items():
    print(f"  {k} {cln:7} {verd:8} loc={str(R[k].get('loc_hit')):5} {R[k]['symptom'][:40]:40} | {why[:70]}")
