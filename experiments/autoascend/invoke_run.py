"""Invocation RITUAL mechanics (endgame-sequence, option b). Wish the four
invocation items, then perform the ritual via the tracked interface, capturing
each step's message: attach 7 candles -> light candelabrum -> ring Bell of
Opening -> read Book of the Dead. The invocation EFFECT only triggers at the
Vibrating Square, but the item-application mechanics are testable anywhere (DL1).
This proves the agent can execute the sequence AutoAscend has no code for."""
import sys, json, gym, nle, lore_patches, lore_scenario, time
seed = int(sys.argv[1]); OUT = sys.argv[2]
lore_scenario.patch_enhance_noop()
from autoascend import global_logic as _gl
import nle.nethack as _nh

WISHES = ["The Candelabrum of Invocation", "7 wax candles",
          "The Bell of Opening", "The Book of the Dead"]
STEPS = []


def _letter(agent, cat, sub, exclude=None):
    for nm, oc, lt in zip(agent.last_observation['inv_strs'],
                          agent.last_observation['inv_oclasses'],
                          agent.last_observation['inv_letters']):
        s = bytes(nm).decode('latin1').strip('\x00').strip().lower()
        if int(oc) == cat and int(lt) != 0 and sub in s and (exclude is None or exclude not in s):
            return chr(int(lt))
    return None


def ritual_global(self):
    agent = self.agent
    import autoascend.agent as _A
    low = agent.env.env.unwrapped.env
    def _msg():
        try:
            return bytes(agent.last_observation['message']).decode('latin1').strip('\x00').strip()
        except Exception:
            return "?"
    if not getattr(agent, "_done", False):
        agent.__dict__["_done"] = True
        for it in WISHES:
            lore_scenario._do_wish(agent, it)
        try:
            agent.step(_A.A.Command.ESC); agent.inventory.update()
        except Exception:
            pass

        # RAW applies: AutoAscend's agent.step ASSERTS on the candelabrum/invocation
        # inventory strings (it has no model for them). low.step bypasses that.
        def apply_item(cmd_char, cat, sub, confirms, exclude=None):
            lt = _letter(agent, cat, sub, exclude=exclude)
            if lt is None:
                STEPS.append("%s: NO ITEM" % sub); return
            def m(r):
                try:
                    o = r[0] if isinstance(r, tuple) else r
                    return bytes(o['message']).decode('latin1').strip('\x00').strip()
                except Exception:
                    return ""
            msgs = []
            msgs.append(m(low.step(ord(cmd_char))))   # 'a' apply / 'r' read
            msgs.append(m(low.step(ord(lt))))         # the item
            for _ in range(confirms):
                msgs.append(m(low.step(ord('y'))))    # answer Attach?/Light? [yn]
            msg = " / ".join(x for x in msgs if x)    # all non-empty step messages
            low.step(13)                       # clear --More--
            try:
                agent.step(_A.A.Command.ESC)
            except Exception:
                pass
            STEPS.append("%s [%s]: %s" % (sub, lt, msg[:90]))

        apply_item('a', _nh.TOOL_CLASS, "candle", 1, exclude="candelabrum")  # attach
        apply_item('a', _nh.TOOL_CLASS, "candelabrum", 1)     # light
        apply_item('a', _nh.TOOL_CLASS, "bell", 0)            # ring Bell of Opening
        apply_item('r', _nh.SPBOOK_CLASS, "papyrus", 0)       # read Book of the Dead
        # final inventory state of the candelabrum
        try:
            for nm in agent.last_observation['inv_strs']:
                s = bytes(nm).decode('latin1').strip('\x00').strip()
                if 'candelabrum' in s.lower():
                    STEPS.append("FINAL: " + s)
        except Exception:
            pass
        lore_patches.COUNTERS["steps"] = STEPS
    raise __import__("autoascend.exceptions", fromlist=["AgentFinished"]).AgentFinished()


_gl.GlobalLogic.global_strategy = ritual_global
env = gym.make("NetHackChallenge-v0", wizard=True, allow_all_modes=True)
try: env.seed(seed, seed)
except Exception: pass
w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: w.end_reason = repr(e)[:60]
json.dump({"seed": seed, "steps": lore_patches.COUNTERS.get("steps")},
          open(OUT, "w"), default=str, indent=1)
print("DONE", flush=True)
