import json
# rich co-pilot verdicts (exact buggy function + symptom given). judge: stronger model, diffs in hand.
V={
 "7ef2d05":("WRONG","'issue not clear... infer' -- vague, no fix"),
 "60b064a":("CORRECT","identified index-not-advancing as the infinite-loop cause = the fix (add index+1)"),
 "95fe8ff":("PARTIAL","right area (raw-markup display) but didn't state the markup=True fix"),
 "a34914b":("PARTIAL","identified None tb_offset as the cause; didn't state the exact restructure"),
 "16b3830":("PARTIAL","right area (highlight on add_row columns); missed the highlight= param"),
 "a8c3b87":("WRONG","vague 'cell length vs char count'; didn't pinpoint"),
 "f591471":("PARTIAL","right locus (URL regex + @) but no exact char-class edit"),
 "ee43879":("WRONG","vague 'selective highlighting'; no fix"),
 "b41cb48":("WRONG","vague 'Windows size detection'; missed the fd-loop fix"),
 "e08d717":("PARTIAL","identified Padding as the locus; fix removes/changes it"),
 "720800e":("WRONG","focused on space-math; missed the assert->default(8) fix (EXACT-TOKEN)"),
 "956bfa5":("WRONG","punted 'no function linked'; fix = add ~ to URL regex (EXACT-TOKEN)"),
 "dbf66e8":("WRONG","'doesn't directly indicate a bug'"),
 "58f54ca":("PARTIAL","right locus (trailing backslash) but not the double-backslash check (EXACT-TOKEN)"),
 "d9eff2a":("WRONG","missed the ==->is identity fix (EXACT-TOKEN)"),
 "680d9d8":("PARTIAL","right area (jupyter empty line); missed the is_jupyter condition"),
 "6796675":("WRONG","'doesn't directly indicate a bug'"),
 "daf6e38":("WRONG","vague '8-bit palette logic'"),
 "aa09292":("PARTIAL","right area (pos calculation); missed the off-by-one len(text)-1 (EXACT-TOKEN)"),
 "c3a2450":("WRONG","framed as flush issue; missed the missing fileno() method"),
}
n=len(V)
strict=sum(1 for k in V if V[k][0]=="CORRECT")
lenient=sum(1 for k in V if V[k][0] in("CORRECT","PARTIAL"))
exact_token_misses=sum(1 for k in V if "EXACT-TOKEN" in V[k][1])
print(f"RICH co-pilot (different domain, perfect localization), N={n}")
print(f"  strict CORRECT:     {strict}/{n} = {strict/n:.0%}")
print(f"  lenient (+PARTIAL): {lenient}/{n} = {lenient/n:.0%}")
print(f"  misses that are EXACT-TOKEN edits (regex char, ==/is, off-by-one, assert->default): {exact_token_misses}")
print()
print("vs AA co-pilot: 2/15=13% strict / 7/15=47% lenient")
print("=> the diagnosis ceiling GENERALIZES: ~5-13% strict, ~45-47% lenient on BOTH")
print("   a NetHack bot and a terminal-rendering library. Points at the AREA (~45%),")
print("   misses the exact TOKEN. The exact-token floor holds cross-domain.")
