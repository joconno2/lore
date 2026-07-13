"""Large-scale behavioral profile of base AutoAscend. One full real game ->
rich record of WHAT it did and WHERE it fell short, for a macro-strategy gap
analysis vs expert play. Captures: outcome, dungeon/branch coverage, depth+XL
trajectory (dawdle vs dive), time-per-level, resource economy (food/prayer/HP),
key-item acquisition, death cause + whether fighting at death, and AA's own
internal event counters. Designed to aggregate over hundreds of seeds."""
import sys, json, gym, nle, time, os
seed = int(sys.argv[1]); OUT = sys.argv[2]

if os.environ.get("LORE_MACRO") in ("mock", "llm"):
    import lore_patches
    lore_patches.apply_macro_director(mock=os.environ.get("LORE_MACRO") == "mock")
elif os.environ.get("LORE_UNSTICK") == "1":
    import lore_patches
    lore_patches.apply_unstick_dl1()
elif os.environ.get("LORE_UNSTICK_LLM") in ("1", "mock"):
    import lore_patches
    lore_patches.apply_unstick_llm(mock=os.environ.get("LORE_UNSTICK_LLM") == "mock")
if os.environ.get("LORE_DGATE") in ("1", "mock", "llm"):
    # dive-readiness gate: hold '>' descent (consolidate XP/gear) until strong
    # enough for the depth. Composes ON TOP of the macro director -- macro sets
    # the deep objective, the gate paces the actual descent to cut the combat
    # deaths from rushing under-leveled. mock=fixed rule; llm=oracle judges.
    import lore_patches
    lore_patches.apply_descent_gate(mock=os.environ.get("LORE_DGATE") != "llm")
if os.environ.get("LORE_CRASHREC") == "1":
    import lore_patches
    lore_patches.apply_crash_recovery()
if os.environ.get("LORE_SOKOPATCH") == "1":
    import lore_patches
    lore_patches.apply()   # route Sokoban solver desync -> graceful abandon (survive past)
if os.environ.get("LORE_SOKOFIX") == "1":
    import lore_patches
    lore_patches.apply_sokoban_fix()   # structural: strip 4-col map indent (unblocks the solve)
if os.environ.get("LORE_ELBFIX2") == "1":
    import elbereth_fix2
    elbereth_fix2.apply()
if os.environ.get("LORE_EMERG") == "1":
    import emergency_boost
    emergency_boost.apply()
if os.environ.get("LORE_ELBFIX") == "1":
    import elbereth_fix
    elbereth_fix.apply()
if os.environ.get("LORE_SURV") in ("1", "mock", "llm"):
    import lore_patches
    lore_patches.apply_survival_oracle(mock=os.environ.get("LORE_SURV") != "llm")
if os.environ.get("LORE_CRVETO") in ("1", "mock", "llm"):
    # crash_recovery + knowledge-gated instadeath veto (petrification etc.) --
    # the two things that kill the high-score TAIL. veto isolated on top of CR.
    import lore_patches
    lore_patches.apply_crash_recovery()
    lore_patches.apply_oracle_veto(mock=os.environ.get("LORE_CRVETO") != "llm")
if os.environ.get("LORE_FOOD") in ("1", "mock", "llm"):
    # opportunistic food economy -- cut the ~27% starvation deaths (proper-descent
    # metric = depth, not score, so a score-costing food fix is fine if it survives).
    import lore_patches
    lore_patches.apply_food_oracle(mock=os.environ.get("LORE_FOOD") != "llm")
if os.environ.get("LORE_ANTISTARV") == "1":
    # emergency at WEAK (before FAINTING): quaff fruit juice / pray-for-food when
    # safe-to-pray -> refills nutrition. Targets the ~25-30% starvation deaths that
    # persist across roles (a food-SUPPLY problem, per FIRST_ASCENSION_PLAYBOOK).
    import lore_patches
    lore_patches.apply_anti_starvation()
if os.environ.get("LORE_FULL") == "1":
    # full stack, perfect-knowledge where applicable: structural robustness +
    # food-aware unstick + pro-survival disengage. The combined ceiling test.
    import lore_patches
    lore_patches.apply_crash_recovery()
    lore_patches.apply_unstick_dl1()
    lore_patches.apply_survival_oracle(mock=True)

from autoascend.agent import Agent as _Ag
_orig_pray = _Ag.pray
_PRAY = {"n": 0, "turns": []}
def _cpray(self):
    _PRAY["n"] += 1
    try: _PRAY["turns"].append(int(self.blstats.time))
    except Exception: pass
    return _orig_pray(self)
_Ag.pray = _cpray

env = gym.make("NetHackChallenge-v0", no_progress_timeout=1000) if os.environ.get("LORE_NPT")=="1" else gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
if os.environ.get("LORE_ROLE"):
    _want=os.environ["LORE_ROLE"].lower()
    import nle.nethack as _nh
    from autoascend import agent as _al
    _s2=seed
    for _ in range(300):
        env.seed(_s2,_s2); _o=env.reset()
        _bl=_al.BLStats(*_o["blstats"]); _g=_o["glyphs"][_bl.y,_bl.x]
        try: _mn=_nh.permonst(_nh.glyph_to_mon(_g)).mname.lower()
        except Exception: _mn=""
        if _want in _mn: break
        _s2+=10**9
    env.seed(_s2,_s2)

_MSGS, _TRAJ = [], []
_max_depth = [1]
_orig = env.step
def _hook(a):
    r = _orig(a)
    try:
        o = r[0]
        m = bytes(o["message"]).decode("latin1").strip("\x00").strip()
        if m and (not _MSGS or _MSGS[-1] != m):
            _MSGS.append(m)
            if len(_MSGS) > 25: _MSGS.pop(0)
        bl = o["blstats"]
        t, dep, hp, mhp, xl, hung = int(bl[20]), int(bl[12]), int(bl[10]), int(bl[11]), int(bl[18]), int(bl[21])
        if dep > _max_depth[0]: _max_depth[0] = dep
        if not _TRAJ or t - _TRAJ[-1][0] >= 250:
            _TRAJ.append((t, dep, xl, round(hp / max(1, mhp), 2), hung))
    except Exception: pass
    return r
env.step = _hook

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
t0 = time.time()
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:90]
s = w.get_summary()
ag = w.agent
end = str(s.get("end_reason") or w.__dict__.get("end_reason") or "")
low = end.lower()

killer = ""
if "killed by" in low:
    killer = end[low.find("killed by") + 10:].split(",")[0].split(".")[0].strip()
def cat():
    e = low + " " + " ".join(_MSGS[-6:]).lower()
    if "lack of food" in e or "starv" in e or ("faint" in e and "food" in e): return "starvation"
    if "turn to stone" in e or "petrif" in e: return "petrification"
    if "frozen" in e or "paralys" in e: return "paralysis"
    if "drown" in e: return "drowning"
    if "while praying" in e or "smote" in e or "wrath" in e: return "prayer"
    if "killed by" in low or "you die" in low: return "combat"
    if "assert" in low or "panic" in low or "recursion" in low or "runtime" in low: return "crash"
    return "other"

# dungeon/branch coverage from visited levels
DNAMES = {0: "DoD", 2: "Mines", 3: "Quest", 4: "Sokoban"}
branches = {}
try:
    for (dnum, lnum) in ag.levels.keys():
        branches.setdefault(DNAMES.get(dnum, "d%d" % dnum), []).append(lnum)
    branches = {k: {"levels": sorted(v), "max": max(v)} for k, v in branches.items()}
except Exception:
    branches = {}

# key items at death
keyitems = []
try:
    from autoascend.item import flatten_items as _fi
    for it in _fi(ag.inventory.items):
        nm = str(it).lower()
        for tag in ("reflection", "wand of wishing", "excalibur", "magic marker",
                    "bag of holding", "magic lamp", "unicorn horn", "speed boots",
                    "dragon scale", "ring of free action", "amulet of life"):
            if tag in nm: keyitems.append(tag)
except Exception:
    pass

fighting = any(any(v in m.lower() for v in ("hits", "bites", "you hit", "swings", "casts", "you miss"))
               for m in _MSGS[-4:])
try: stats = s.get  # already merged stats_dict in summary
except Exception: stats = lambda k, d=None: d

rec = {"seed": seed, "score": s.get("score"), "turns": s.get("turns"),
       "max_depth": _max_depth[0], "xl": s.get("experience_level"),
       "milestone": str(s.get("milestone")), "death_cat": cat(), "killer": killer,
       "fighting_at_death": fighting, "branches": branches,
       "did_mines": "Mines" in branches, "did_sokoban": "Sokoban" in branches,
       "did_quest": "Quest" in branches,
       # PHYSICAL Minetown reach (global_logic sets minetown_level only on arrival)
       # -- the ladder's milestone-based proxy is contaminated by the macro
       # director setting FIND_SOKOBAN as an objective, so capture the real thing.
       "did_minetown": bool(getattr(getattr(ag, "global_logic", None), "minetown_level", None)),
       "sokoban_dropped": s.get("sokoban_dropped"), "elbereth": s.get("elbereth_write"),
       "agent_panic": s.get("agent_panic"), "prayers": _PRAY["n"],
       "key_items": sorted(set(keyitems)), "gold_last": s.get("gold_last"),
       "max_turns_on_position": s.get("max_turns_on_position"),
       "traj": _TRAJ, "end_reason": end[:120], "msgs_tail": " || ".join(_MSGS[-5:])[:250],
       "t": round(time.time() - t0)}
json.dump(rec, open(OUT, "w"), default=str)
print("DONE", flush=True)
