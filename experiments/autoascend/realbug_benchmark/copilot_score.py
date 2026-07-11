import json
# co-pilot verdicts: diagnosis given the EXACT buggy function + symptom (perfect localization)
V={
 "073d770":("CORRECT","named the assert inactivity_counter<5 + quit-at-5000 = the fix"),
 "bf8617a":("WRONG","looked for gas-spore-specific code, found none; missed the generic actions-emptiness guard"),
 "ec3cc35":("WRONG","'Fix RL' too vague even with the function; generic yield True/False guess"),
 "88a1b9b":("WRONG","described the .before chain but never identified the parenthesization fix"),
 "bad5143":("PARTIAL","echoed 'atom operation'; right area, no precise with-atom_operation wrap"),
 "6bd6eee":("PARTIAL","flagged 'redundant type_text(y)' -- near the changed lines but vague fix"),
 "3b25d57":("WRONG","misled by 'pyinstrument' symptom into profiling; real fix = glyphs->monster_mask"),
 "195eaad":("PARTIAL","identified monk-armor special handling as the area; fix not precisely stated"),
 "bde22a3":("PARTIAL","headed to mask processing (the right place) but visible text vague; autonomously got it correct"),
 "4b536a3":("WRONG","'not directly visible... infer' -- couldn't find it even with the function (big refactor)"),
 "e3328d1":("WRONG","said 'trap handling'; e3328d1 fix is wall_map/soko_solver map-alignment, not traps"),
 "179a676":("WRONG","generic forking/reloading guesses; real fix = env.reset()->step(ESC)"),
 "0114ad4":("CORRECT","found the real bug: the '^' pit-char check is incorrect (=the exact fix). Truncation hallucination GONE with full function"),
 "326741c":("WRONG","mentioned mask init but missed the race-based acid/poison corpse filter"),
 "97a231d":("PARTIAL","identified the --More-- extended-message handling as the area; fix is a big rewrite"),
}
n=len(V)
strict=sum(1 for k in V if V[k][0]=="CORRECT")
lenient=sum(1 for k in V if V[k][0] in("CORRECT","PARTIAL"))
print(f"CO-PILOT (perfect localization, single buggy function given), N={n}")
print(f"  strict CORRECT:      {strict}/{n} = {strict/n:.0%}")
print(f"  lenient (+PARTIAL):  {lenient}/{n} = {lenient/n:.0%}")
print()
print("vs autonomous: v2 clean-9 2/9=22% strict / 44% lenient; v3 all-19 1/19=5% / 32%")
print("=> perfect localization does NOT rescue strict diagnosis (~13% vs ~5-22%).")
print("   The bottleneck is DIAGNOSIS reasoning, not retrieval. Lenient rose 32%->47%")
print("   (points at the right AREA more often, but names the specific fix only ~13%).")
