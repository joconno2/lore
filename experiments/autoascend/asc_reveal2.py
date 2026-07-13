"""Make wizard ^F level-reveal SAFE mid-descent. In a one-shot probe ^F works, but
fired inside the descend loop it corrupts AA's observation state: inventory.update
fails, _wand_letter returns None, the agent freezes (explored_cells=1) and dies.
This probe reproduces the mid-play context (teleport, take real steps) then tries
post-^F resync variants, checking after each: is a downstair known? is the wand
letter still accessible? can the agent still move? Pick the variant that reveals
the stair AND leaves inventory + movement intact.

Usage: python asc_reveal2.py <seed> <variant> <out>
  variant 0 = current (3x Enter + ESC + inventory.update)   [known-broken]
  variant 1 = ESC only, no inventory.update
  variant 2 = ESC + a real agent search(1) to resync
  variant 3 = read downstair from RAW glyphs, no AA resync at all
"""
import sys, json, gym, nle, lore_patches, lore_scenario
import numpy as np
seed = int(sys.argv[1]); variant = int(sys.argv[2]); OUT = sys.argv[3]
lore_scenario.patch_enhance_noop(); lore_scenario.patch_ring_parse()
from autoascend import global_logic as _gl
from autoascend.glyph import G
from autoascend import utils as _u
from autoascend.exceptions import AgentFinished
import autoascend.agent as _ag
import nle.nethack as _nh

R = {"seed": seed, "variant": variant}


def _wand_letter(agent):
    for oc, l in zip(agent.last_observation['inv_oclasses'],
                     agent.last_observation['inv_letters']):
        if int(oc) == _nh.WAND_CLASS and int(l) != 0:
            return chr(int(l))
    return None


def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    low = agent.env.env.unwrapped.env
    # place deep + a few real steps to establish a mid-play state
    try:
        lore_scenario._do_teleport(agent, 28)
    except Exception as e:
        R["tp_err"] = repr(e)[:60]
    for _ in range(4):
        try:
            lvl = agent.current_level(); bf = agent.bfs()
            mask = (bf != -1) & lvl.walkable & (~lvl.was_on)
            cand = list(zip(*mask.nonzero()))
            if cand:
                cand.sort(key=lambda p: bf[p[0], p[1]])
                agent.go_to(int(cand[0][0]), int(cand[0][1]), max_steps=1)
        except Exception:
            break
    R["wand_before"] = _wand_letter(agent)
    R["pos_before"] = [int(agent.blstats.y), int(agent.blstats.x)]
    # ---- fire ^F, then resync per variant ----
    try:
        low.step(6)   # ^F wiz_map
        if variant == 0:
            for _ in range(3): low.step(13)
            agent.step(_ag.A.Command.ESC); agent.inventory.update()
        elif variant == 1:
            low.step(27)  # single ESC, no inventory.update
        elif variant == 2:
            low.step(27)
            agent.step(_ag.A.Command.ESC)
            try: agent.search(1)   # a real turn to resync AA's observation
            except Exception: pass
        elif variant == 3:
            low.step(27)  # ESC dismiss; then read raw glyphs only (no AA resync)
    except Exception as e:
        R["reveal_err"] = repr(e)[:80]
    # ---- checks ----
    try:
        lvl = agent.current_level(); bf = agent.bfs()
        dmask = _u.isin(lvl.objects, G.STAIR_DOWN)
        R["aa_downstairs"] = int(dmask.sum())
        R["aa_down_reachable"] = int(sum(1 for y, x in zip(*dmask.nonzero()) if bf[y, x] != -1))
    except Exception as e:
        R["aa_scan_err"] = repr(e)[:60]
    # raw-glyph downstair (parser independent)
    try:
        g = np.asarray(agent.last_observation['glyphs'])
        R["raw_down_positions"] = [[int(y), int(x)] for y, x in zip(*(g >= 0).nonzero())][:0]  # placeholder
        # AA exposes stair glyph sets; count via objects fallback done above
    except Exception:
        pass
    R["wand_after"] = _wand_letter(agent)
    # can it still move?
    try:
        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
        lvl = agent.current_level(); bf = agent.bfs()
        mask = (bf != -1) & lvl.walkable & (~lvl.was_on)
        cand = list(zip(*mask.nonzero()))
        if cand:
            cand.sort(key=lambda p: bf[p[0], p[1]])
            agent.go_to(int(cand[0][0]), int(cand[0][1]), max_steps=1)
        R["moved"] = (int(agent.blstats.y), int(agent.blstats.x)) != (y0, x0)
    except Exception as e:
        R["move_err"] = repr(e)[:60]
    R["verdict"] = ("OK" if R.get("wand_after") and R.get("aa_downstairs", 0) > 0
                    else "BROKEN wand=%s down=%s" % (R.get("wand_after"), R.get("aa_downstairs")))
    raise AgentFinished()


_gl.GlobalLogic.global_strategy = probe
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: R["end"] = repr(e)[:60]
json.dump(R, open(OUT, "w"), default=str, indent=1)
print("DONE v%d" % variant, R.get("verdict"), flush=True)
