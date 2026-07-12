"""DECISIVE topology probe for rung 4 (reach the vibrating square).

The descent runs die 0/16 with n_downstairs_known=0 -- but that conflates two very
different failures: (A) the ^V-teleported Gehennom level has NO reachable downstair
(degenerate placement -> the whole ^V-by-depth approach is wrong), vs (B) a downstair
EXISTS but the agent dies searching for it (survival/search problem -> tune the loop).

Play data can't tell them apart because the agent never covers the level. So use
wizard-mode level reveal (^F / wiz_map): teleport, MAGIC-MAP the entire level, then
check whether a downstair glyph exists at all and whether it is BFS-reachable from
the agent. This settles A vs B in one keypress, no survival needed.

Runs several targets in one process is not possible (one iterative teleport per game),
so one (seed, target) per invocation. Usage: python asc_reveal.py <seed> <target> <out>
"""
import sys, json, gym, nle, lore_patches, lore_scenario
seed = int(sys.argv[1]); target = int(sys.argv[2]); OUT = sys.argv[3]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
from autoascend.glyph import G
from autoascend import utils as _u
from autoascend.exceptions import AgentFinished
import autoascend.agent as _ag
import nle.nethack as _nh
import numpy as np

R = {"seed": seed, "target": target}

# downstair cmap glyph (raw-glyph check, independent of AA's parser)
try:
    DN_GLYPH = _nh.GLYPH_CMAP_OFF + _nh.S_dnstair
    UP_GLYPH = _nh.GLYPH_CMAP_OFF + _nh.S_upstair
except Exception:
    DN_GLYPH = UP_GLYPH = -1


def probe(self):
    agent = self.agent
    if getattr(agent, "_done", False):
        raise AgentFinished()
    agent.__dict__["_done"] = True
    low = agent.env.env.unwrapped.env
    # 1) teleport to target (iterative, force-generating each Gehennom level)
    try:
        lore_scenario._do_teleport(agent, target)
    except Exception as e:
        R["tp_err"] = repr(e)[:80]
    try:
        R["depth"] = int(agent.blstats.depth)
        R["dungeon"] = int(agent.current_level().dungeon_number)
    except Exception as e:
        R["stat_err"] = repr(e)[:60]
    # 2) MAGIC-MAP the whole level: ^F (ASCII 6) = wiz_map. clear prompts.
    try:
        low.step(6)
        for _ in range(3):
            low.step(13)
        agent.step(_ag.A.Command.ESC)
        agent.inventory.update()
    except Exception as e:
        R["map_err"] = repr(e)[:60]
    try:
        R["map_msg"] = bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip()[:80]
    except Exception:
        pass
    # 3) count downstairs -- BOTH via AA's parsed objects and via raw glyphs
    try:
        lvl = agent.current_level()
        bf = agent.bfs()
        walk = lvl.walkable
        reach = (bf != -1) & walk
        # AA-parsed downstairs
        dmask = _u.isin(lvl.objects, G.STAIR_DOWN)
        umask = _u.isin(lvl.objects, G.STAIR_UP)
        dpos = list(zip(*dmask.nonzero()))
        R["aa_downstairs"] = len(dpos)
        R["aa_downstairs_reachable"] = int(sum(1 for y, x in dpos if bf[y, x] != -1))
        R["aa_downstair_yx_reach"] = [[int(y), int(x), int(bf[y, x])] for y, x in dpos][:8]
        R["aa_upstairs"] = int(umask.sum())
        R["reachable_cells"] = int(reach.sum())
        R["total_walkable"] = int(walk.sum())
        R["reach_frac"] = round(int(reach.sum()) / max(1, int(walk.sum())), 3)
        # raw-glyph downstairs (parser-independent)
        g = np.asarray(agent.last_observation['glyphs'])
        R["raw_downstairs"] = int((g == DN_GLYPH).sum()) if DN_GLYPH >= 0 else -1
        R["raw_upstairs"] = int((g == UP_GLYPH).sum()) if UP_GLYPH >= 0 else -1
        rd = list(zip(*(g == DN_GLYPH).nonzero())) if DN_GLYPH >= 0 else []
        R["raw_downstair_yx_reach"] = [[int(y), int(x), int(bf[y, x])] for y, x in rd][:8]
        # ASCII render of the revealed level
        my = (int(agent.blstats.y), int(agent.blstats.x))
        wall = _u.isin(lvl.objects, G.WALL)
        rows = []
        for ry in range(lvl.objects.shape[0]):
            r = ""
            for rx in range(lvl.objects.shape[1]):
                if (ry, rx) == my: r += "@"
                elif dmask[ry, rx]: r += ">"
                elif umask[ry, rx]: r += "<"
                elif wall[ry, rx]: r += "|"
                elif bf[ry, rx] != -1: r += "."
                elif lvl.walkable[ry, rx]: r += ":"   # walkable but NOT reachable
                else: r += " "
            rows.append(r.rstrip())
        R["ascii"] = "\n".join(rows)
        # verdict
        dn_exists = max(R["aa_downstairs"], R.get("raw_downstairs", 0))
        dn_reach = R["aa_downstairs_reachable"] + int(sum(1 for y, x in rd if bf[y, x] != -1))
        if dn_exists == 0:
            R["verdict"] = "NO_DOWNSTAIR"           # degenerate level / Sanctum-like
        elif dn_reach > 0:
            R["verdict"] = "DOWNSTAIR_REACHABLE"     # survival/search problem, not topology
        else:
            R["verdict"] = "DOWNSTAIR_WALLED"        # exists but behind moat/lava/stone
    except Exception as e:
        import traceback
        R["scan_err"] = repr(e)[:80]
        R["scan_tb"] = traceback.format_exc()[-300:]
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
print("DONE", R.get("verdict"), "aa_dn=%s raw_dn=%s reach_frac=%s" %
      (R.get("aa_downstairs"), R.get("raw_downstairs"), R.get("reach_frac")), flush=True)
