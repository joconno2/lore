import json
# verdicts on v3 all-19 (match-any-changed-function). judge: stronger model, diffs in hand.
V={
 "073d770":("PARTIAL","named the cyclic-panic assert but this run fixated on 'message formatting' -- MISSED the control-flow fix (assert->graceful quit). NB: v2 run got this CORRECT: same input, temp0.2, flipped"),
 "bf8617a":("WRONG","melee_monster_priority + fabricated gas-spore logging; real fix = actions-emptiness guard in get_priorities"),
 "ec3cc35":("WRONG","'Fix RL' 0 groundable terms -> generic RL boilerplate"),
 "5cc0b5b":("WRONG","right theme (ranged-slow avoidance) but wrong function; missed the acid-blob list change"),
 "88a1b9b":("WRONG","get_visible_monsters invisible-filter fabricated; real fix = gather_items .before chaining"),
 "e328738":("PARTIAL","right theme (ranged-only-monster priority, one of the bundled changes) but named a non-changed function, vague"),
 "68dfb02":("WRONG","fabricated a blind/skip condition; real fix = inverted 'read' in msg() + msg()->smsg()"),
 "bad5143":("PARTIAL","echoed the symptom ('atom operation','step calling order'); right area/changed-func but no diagnostic value beyond the symptom"),
 "5565c97":("WRONG","single_simulation ThreadPool; real fix = offer_corpses message matching"),
 "6bd6eee":("WRONG","'Fixes' 0 terms -> calculate_sum boilerplate"),
 "de0f93a":("PARTIAL","right theme (ranged priority) but wrong function, missed the double-y eat fix"),
 "3b25d57":("WRONG","bundled: symptom 'pyinstrument' != move() monster_mask fix; diagnosed profiling"),
 "195eaad":("WRONG","0 terms -> vague 'adjust monk armor'; missed the MONK role skip"),
 "bde22a3":("CORRECT","named go_to_item_to_pickup + the empty-mask edge = the exact fix"),
 "65cb197":("PARTIAL","right concept (position assertion failing) but named explore_stairs not move()"),
 "179a676":("WRONG","0 terms -> generic forking boilerplate"),
 "0114ad4":("WRONG","FABRICATED a typo 'offse'; real fix = trap detection char '^' -> G.TRAPS"),
 "4873b55":("WRONG","retrieved self_play.py (wrong area); real fix = drop get_dps, use calc_dps"),
 "326741c":("WRONG","gate FALSE-DECLINED a real bug; real fix = filter corpses by race acid/poison"),
}
R={r["sha"][:7]:r for r in json.load(open("/tmp/aa_bench3_results.json"))}
loc=sum(1 for k in V if R[k].get("loc_hit_any"))
strict=sum(1 for k in V if V[k][0]=="CORRECT")
lenient=sum(1 for k in V if V[k][0] in("CORRECT","PARTIAL"))
n=len(V)
print(f"N={n}")
print(f"localization (match-any, top-6): {loc}/{n} = {loc/n:.0%}")
print(f"diagnosis strict CORRECT:        {strict}/{n} = {strict/n:.0%}")
print(f"diagnosis lenient (+PARTIAL):     {lenient}/{n} = {lenient/n:.0%}")
def wilson(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);m=z*((p*(1-p)+z*z/(4*n))/n)**.5
    return((c-m)/d,(c+m)/d)
lo,hi=wilson(strict,n);print(f"  strict 95% CI: [{lo:.0%}, {hi:.0%}]")
lo,hi=wilson(lenient,n);print(f"  lenient 95% CI: [{lo:.0%}, {hi:.0%}]")
print(f"every CORRECT also localized: {all(R[k].get('loc_hit_any') for k in V if V[k][0]=='CORRECT')}")
