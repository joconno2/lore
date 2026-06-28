"""Knowledge-gated THREAT ablation: spawn instant-death monsters (cockatrice /
floating eye) next to a vulnerable char (bare hands, no ranged) and measure
survival under: base AutoAscend combat | mock-veto | llm-veto. Tests where the
descent null said value should be -- knowledge-dependent decisions AA gets wrong.
LORE_VETO = none|mock|llm. The veto isolates the oracle's effect (no other
interventions)."""
import sys, json, gym, nle, lore_patches, lore_scenario, time, os
seed = int(sys.argv[1]); target = int(sys.argv[2]); OUT = sys.argv[3]
veto = os.environ.get("LORE_VETO", "none")

lore_scenario.patch_enhance_noop()
if veto == "mock":
    lore_patches.apply_oracle_veto(mock=True)
elif veto == "llm":
    lore_patches.apply_oracle_veto(mock=False)

# vulnerable char: XL via gain-level, body armor for AC, but NO gloves and NO
# ranged -> meleeing a cockatrice = petrification, a floating eye = paralysis.
KIT = ["12 blessed potions of gain level", "blessed +2 ring mail"]
SPAWN = os.environ.get("LORE_SPAWN", "cockatrice").split(",")
lore_scenario.install_threat(target, wishes=KIT, spawn=SPAWN)

env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
_MSGS = []
_orig = env.step
def _hook(a):
    r = _orig(a)
    try:
        obs = r[0]
        m = bytes(obs["message"]).decode("latin1").strip("\x00").strip()
        if m and (not _MSGS or _MSGS[-1] != m):
            _MSGS.append(m)
            if len(_MSGS) > 40: _MSGS.pop(0)
    except Exception: pass
    return r
env.step = _hook

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
t0 = time.time()
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:80]
s = w.get_summary(); C = lore_patches.COUNTERS

death = ""
try:
    death = " || ".join(_MSGS[-6:])[:400]   # last messages = the death cause
    KW = ("stone", "petrif", "paralyz", "frozen", "solidif", "stiffen", "slowing")
    cause = "petrify/paralyze" if any(k in " ".join(_MSGS).lower() for k in KW) else "damage/other"
    lore_patches.COUNTERS["cause"] = cause
except Exception as e:
    death = "cap_fail %r" % e

json.dump({"seed": seed, "target": target, "veto": veto,
           "score": s.get("score"), "turns": s.get("turns"),
           "xl": s.get("experience_level"), "tp_depth": C.get("tp_depth"),
           "xl_after": C.get("xl_after"), "spawned": C.get("spawned"),
           "veto_query": C.get("veto_query"), "veto_fired": C.get("veto_fired"),
           "end": str(w.__dict__.get("end_reason"))[:60], "death": death,
           "t": round(time.time() - t0)}, open(OUT, "w"), default=str)
print("DONE", flush=True)
