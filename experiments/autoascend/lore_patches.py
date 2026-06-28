"""LORE intervention layer: runtime patches over the frozen AutoAscend base.

We do not edit AutoAscend's source. We wrap it from outside. First intervention:
the Sokoban solver crashes with a bare AssertionError when the real board
diverges from its precomputed solution (boulder destroyed/displaced). That kills
~10% of runs, disproportionately the deepest ones. AutoAscend already has a
graceful abandon path (log 'sokoban_dropped' -> advance milestone -> raise
AgentPanic). We route the desync into that path instead of letting it crash.
"""
import functools
from autoascend.strategy import Strategy
from autoascend import global_logic as _gl
from autoascend.exceptions import AgentPanic

_Milestone = _gl.Milestone

# Reliable telemetry: AutoAscend's StatsLogger.log_event raises KeyError on
# unknown names (fixed dict), so we count intervention firings here instead.
COUNTERS = {}


def _bump(name):
    COUNTERS[name] = COUNTERS.get(name, 0) + 1


def apply_crash_recovery(max_recover=300):
    """PEAK lever: AutoAscend's deepest runs die to its OWN AssertionError crashes
    (5 of the top-10 deepest seeds). Convert any non-panic crash into a recovery
    MOVE -- relocate + advance a turn so the deterministic crash path isn't
    immediately re-selected and the cyclic-panic guard never trips. Keeps the
    peak runs alive through the base's bugs. Cap prevents true infinite spin."""
    import random as _random
    from autoascend.agent import Agent
    from autoascend.exceptions import AgentPanic, AgentFinished

    orig = Agent.handle_exception
    dirs = ["n", "s", "e", "w", "ne", "nw", "se", "sw"]

    def _recover_move(self):
        c = self.__dict__.get("_lore_crash", 0)
        order = dirs[c % 8:] + dirs[:c % 8]  # rotate escape dir by crash count
        for d in order:
            try:
                self.move(d)
                return True
            except Exception:
                continue
        try:
            self.search(1)
            return True
        except Exception:
            return False

    def safe_handle(self, exc):
        if isinstance(exc, (KeyboardInterrupt, AgentFinished, SystemExit)):
            raise exc
        # The env is finished (agent died): stepping it again raises this
        # RuntimeError. Recovering from it just re-steps the dead env -> the
        # cyclic-panic guard trips and masks the real death. Treat as terminal.
        if isinstance(exc, RuntimeError) and "finished" in str(exc):
            raise AgentFinished()
        if isinstance(exc, AgentPanic):
            return orig(self, exc)
        # non-panic crash -> recover instead of dying
        d = int(getattr(self.blstats, "depth", 0)) if getattr(self, "blstats", None) else 0
        if self.__dict__.get("_lore_crash_depth") != d:
            self.__dict__["_lore_crash_depth"] = d
            self.__dict__["_lore_crash"] = 0  # reset per depth (progress made)
        c = self.__dict__.get("_lore_crash", 0) + 1
        self.__dict__["_lore_crash"] = c
        _bump("crash_recover")
        if c > max_recover:
            raise exc
        try:
            self.all_panics.append(exc)
        except Exception:
            pass
        _recover_move(self)

    Agent.handle_exception = safe_handle
    return ["crash_recovery (recover-move, cap=%d)" % max_recover]


def apply():
    # Operate on the Strategy object the original method returns, not its
    # internals -- robust to however many decorators are stacked on it.
    orig_method = _gl.GlobalLogic.solve_sokoban_strategy

    def patched(self, *a, **k):
        orig_strat = orig_method(self, *a, **k)        # a Strategy
        orig_factory = orig_strat.strategy             # callable -> generator

        def safe_factory():
            gen = orig_factory()
            try:
                cond = next(gen)                       # condition yield
            except StopIteration as e:
                return
            yield cond
            if not cond:
                return
            try:
                next(gen)                              # run body
            except StopIteration as e:
                return e.value
            except AssertionError:
                # Sokoban solver desynced from the real board. Abandon via
                # AutoAscend's own recovery: advance milestone, panic.
                try:
                    self.agent.stats_logger.log_event("sokoban_desync_lore")
                except Exception:
                    pass
                try:
                    self.milestone = _Milestone(int(self.milestone) + 1)
                except Exception:
                    pass
                raise AgentPanic("sokoban solver desync (LORE patch)")

        return Strategy(safe_factory, getattr(orig_strat, "config", None))

    _gl.GlobalLogic.solve_sokoban_strategy = patched
    return ["safe_solve_sokoban (Strategy-wrap)"]


# --- Intervention #2: oracle melee veto on knowledge-dependent instadeaths ---
# AutoAscend melees cockatrices (not in its never-melee set) -> petrification
# deaths, including its deepest runs. We intercept melee_monster_priority: when
# the target is a flagged-dangerous monster, query the oracle; if it says not to
# melee, return a strongly negative priority so AutoAscend falls back to ranged
# or avoidance. Only flagged monsters trigger a query, so queries stay rare.

# Knowledge-gated threats: meleeing these barehanded/at-melee is an instant or
# near-instant death the wiki warns about but a damage-greedy bot may walk into.
_DANGEROUS = ("cockatrice", "chickatrice", "floating eye")
_NON_MELEE_ACTIONS = {"RANGED", "AVOID", "FLEE", "ELBERETH", "PRAY"}


def apply_oracle_veto(mock=True, base_url=None, model=None):
    """The LLM oracle gates AutoAscend's melee decision at knowledge-dependent
    threats. When the oracle (given the real threat + the char's ranged/gloves
    state) says anything but FIGHT, veto the melee so AA falls back to ranged/
    avoid. This is the knowledge-gated decision the descent null pointed us to."""
    import oracle as _oracle
    from autoascend.combat import fight_heur as _fh

    orig = _fh.melee_monster_priority

    def _has_gloves(agent):
        try:
            return agent.inventory.items.gloves is not None
        except Exception:
            return False

    def _has_ranged(agent):
        try:
            from autoascend.item import flatten_items as _fi
            import nle.nethack as _nh
            for it in _fi(agent.inventory.items):
                if getattr(it, "category", None) == _nh.WAND_CLASS:
                    return True
                if hasattr(it, "is_launcher") and it.is_launcher():
                    return True
            return False
        except Exception:
            return False

    def patched_melee_priority(agent, monsters, monster):
        try:
            _, y, x, mon, _ = monster
            name = (mon.mname or "").lower()
            if any(p in name for p in _DANGEROUS):
                bl = agent.blstats
                state = {
                    "role": str(getattr(agent, "character", ""))[:3],
                    "xl": int(bl.experience_level), "hp": int(bl.hitpoints),
                    "max_hp": int(bl.max_hitpoints), "depth": int(bl.depth),
                    "threat_name": name, "threat_dist": 1,
                    "has_ranged": _has_ranged(agent), "has_gloves": _has_gloves(agent),
                    "can_elbereth": True,
                }
                d = _oracle.query_threat(state, base_url=base_url, model=model, mock=mock)
                _bump("veto_query")
                _bump("veto_query_" + name.replace(" ", "_"))
                if d.get("action") in _NON_MELEE_ACTIONS:
                    _bump("veto_fired")
                    return -1000  # veto melee; AutoAscend falls back to ranged/avoid
        except Exception:
            pass
        return orig(agent, monsters, monster)

    _fh.melee_monster_priority = patched_melee_priority
    return ["oracle_melee_veto (mock=%s)" % mock]


# --- Intervention #3: oracle-gated descent timing -------------------------
# AutoAscend leaves DL1 at XL8 then rushes to DL4-5 and dies to fast hitters
# (unicorns: all XL8-9 on DL4-5). move() is the single choke point for stair
# traversal. On a '>' descent we ask the oracle DESCEND vs BUILD. BUILD holds
# the descent (the agent levels/explores instead). Re-queried per (level, XL)
# so rising XP eventually flips to DESCEND; a per-level hold cap guarantees no
# deadlock. Queries are at most ~once per level per XP gain, so cheap.

def apply_descent_gate(mock=True, base_url=None, model=None, max_holds_per_level=6):
    import oracle as _oracle
    from autoascend.agent import Agent
    from autoascend.exceptions import AgentPanic

    orig_move = Agent.move

    def gated_move(self, y, x=None):
        try:
            d = self.calc_direction(self.blstats.y, self.blstats.x, y, x) if x is not None else y
        except Exception:
            d = y
        if d == ">":
            try:
                bl = self.blstats
                key = self.current_level().key()
                cache = self.__dict__.setdefault("_lore_descent_cache", {})
                holds = self.__dict__.setdefault("_lore_descent_holds", {})
                ck = (key, int(bl.experience_level))
                dec = cache.get(ck)
                if dec is None:
                    state = {"role": "Val", "xl": int(bl.experience_level),
                             "hp": int(bl.hitpoints), "max_hp": int(bl.max_hitpoints),
                             "depth": int(bl.depth),
                             "hunger": int(getattr(bl, "hunger_state", 1))}
                    dec = _oracle.query_descent(state, base_url=base_url, model=model, mock=mock).get("decision", "DESCEND")
                    cache[ck] = dec
                    _bump("descent_query")
                if dec == "BUILD":
                    holds[key] = holds.get(key, 0) + 1
                    if holds[key] <= max_holds_per_level:
                        _bump("descent_hold")
                        raise AgentPanic("lore: build before descending DL%d (XL%d)"
                                         % (int(bl.depth), int(bl.experience_level)))
                    _bump("descent_cap_reached")
                else:
                    _bump("descent_go")
            except AgentPanic:
                raise
            except Exception:
                pass
        return orig_move(self, y, x)

    Agent.move = gated_move
    return ["oracle_descent_gate (mock=%s, cap=%d)" % (mock, max_holds_per_level)]


# --- Intervention #4: petrifier avoidance via movement heatmap ------------
# ACTION INJECTION (not veto). AutoAscend's movement heatmap drives where the
# agent steps; cockatrice falls into the generic "engage" branch (attraction ->
# walks adjacent -> petrified by the monster's own attack). We rewrite its
# heatmap contribution to pure repulsion (+ ranged if available) so the agent
# keeps distance. The player (speed 12) outruns a cockatrice (speed 6), so
# fleeing actually works. fight_heur imports the draw_* names directly, so we
# patch fight_heur's bound references, not the movement_priority module.

_PETRIFIERS_AV = ("cockatrice", "chickatrice")


def apply_petrifier_avoidance():
    from autoascend.combat import fight_heur as _fh
    from autoascend.combat import movement_priority as _mp

    orig_pos = _fh.draw_monster_priority_positive
    orig_neg = _fh.draw_monster_priority_negative

    def pos(agent, monster, priority, walkable):
        _, y, x, mon, _ = monster
        if any(p in (mon.mname or "").lower() for p in _PETRIFIERS_AV):
            priority[y, x] = float("nan")  # never step onto it
            try:
                if len(agent.inventory.get_ranged_combinations()):
                    _mp._draw_ranged(priority, y, x, 2, walkable, radius=7, operation="max")
            except Exception:
                pass
            _bump("petrifier_avoid")
            return
        return orig_pos(agent, monster, priority, walkable)

    def neg(agent, monster, priority, walkable):
        _, y, x, mon, _ = monster
        if any(p in (mon.mname or "").lower() for p in _PETRIFIERS_AV):
            _mp._draw_around(priority, y, x, -10, radius=1)
            _mp._draw_around(priority, y, x, -6, radius=2)
            _mp._draw_around(priority, y, x, -3, radius=3)
            return
        return orig_neg(agent, monster, priority, walkable)

    _fh.draw_monster_priority_positive = pos
    _fh.draw_monster_priority_negative = neg
    return ["petrifier_avoidance (heatmap repulsion)"]


# --- Intervention #5: generalized oracle override -------------------------
# The contribution, not per-monster patches. One preempting strategy at the top
# of global_strategy. Broad trigger (a non-trivial threat is present); hand the
# full game state to the oracle; execute whatever action it returns. Covers the
# long tail of knowledge-dependent deaths uniformly -- petrification, fast
# hitters, floating eyes, low-HP fights -- with a RANGE of success that is the
# result. EC later tunes the trigger and interface. Real action injection: we
# call AutoAscend's own primitives (pray/engrave/move/fire), not just veto.

_WEAK = ("newt", "lichen", "grid bug", "sewer rat", "jackal", "kobold", "gnome")


def _nearest_threat(agent):
    bl = agent.blstats
    best = None
    for m in agent.get_visible_monsters():
        try:
            _, y, x, mon, _ = m
            name = (mon.mname or "").lower()
            d = max(abs(y - bl.y), abs(x - bl.x))
            if best is None or d < best[0]:
                best = (d, y, x, name, mon)
        except Exception:
            continue
    return best


def _step_away(agent, ty, tx):
    bl = agent.blstats
    dy = (bl.y > ty) - (bl.y < ty)
    dx = (bl.x > tx) - (bl.x < tx)
    d = ("s" if dy > 0 else "n" if dy < 0 else "") + ("e" if dx > 0 else "w" if dx < 0 else "")
    if not d:
        return False
    try:
        agent.move(d)
        return True
    except Exception:
        return False


def apply_oracle_override(mock=True, base_url=None, model=None, trigger_radius=3, hp_trigger=0.5):
    import oracle as _oracle
    try:
        import game_state as _gs
    except Exception:
        _gs = None
    from autoascend.strategy import Strategy
    from autoascend.combat import fight_heur as _fh

    orig_global = _gl.GlobalLogic.global_strategy

    def _risky(agent):
        bl = agent.blstats
        if bl.max_hitpoints and bl.hitpoints < hp_trigger * bl.max_hitpoints:
            t = _nearest_threat(agent)
            return t is not None and t[0] <= trigger_radius + 2
        t = _nearest_threat(agent)
        if t is None:
            return False
        d, _, _, name, _ = t
        return d <= trigger_radius and not any(w in name for w in _WEAK)

    def _make_override(self):
        agent = self.agent

        def factory():
            # CONDITION PHASE: no agent steps allowed here (preempt runs this
            # under disallow_step_calling). Decide whether to take control.
            if not _risky(agent):
                yield False
                return
            # once-per-turn guard: never fire twice on the same game turn, so a
            # non-turn-advancing action can't spin into a turn-inactivity death.
            now = int(agent.blstats.time)
            if agent.__dict__.get("_lore_override_turn") == now:
                yield False
                return
            t = _nearest_threat(agent)
            try:
                if _gs:
                    tm = (0, t[1], t[2], t[4], 0) if t else None
                    state = _gs.extract(agent, threat_monster=tm)
                else:
                    state = {"xl": int(agent.blstats.experience_level), "hp": int(agent.blstats.hitpoints),
                             "max_hp": int(agent.blstats.max_hitpoints), "depth": int(agent.blstats.depth),
                             "threat_name": t[3] if t else "none"}
                act = (_oracle.query_threat(state, base_url=base_url, model=model, mock=mock).get("action") or "FIGHT")
            except Exception:
                act = "FIGHT"
            if act == "FIGHT":
                yield False  # don't preempt; let AutoAscend's own combat handle it
                return
            yield True
            # ACTION PHASE: steps allowed.
            agent.__dict__["_lore_override_turn"] = now
            _bump("override_act_" + act)
            try:
                if act == "PRAY":
                    agent.pray()
                elif act == "ELBERETH":
                    try:
                        _fh.elbereth_action(agent, agent.get_visible_monsters())
                    except Exception:
                        agent.engrave("Elbereth")
                elif act in ("FLEE", "AVOID", "RANGED"):
                    if t is None or not _step_away(agent, t[1], t[2]):
                        try:
                            agent.engrave("Elbereth")
                        except Exception:
                            pass
            except Exception:
                pass

        return Strategy(factory)

    def patched_global(self):
        base = orig_global(self)
        return base.preempt(self.agent, [_make_override(self)])

    _gl.GlobalLogic.global_strategy = patched_global
    return ["oracle_override (mock=%s, radius=%d, hp<%.0f%%)" % (mock, trigger_radius, hp_trigger * 100)]


# --- Intervention #6: anti-starvation (pray/eat at WEAK, before disorientation) -
# Diagnosis (seed 107, 74k run): the #1 killer of deep runs is a starvation
# death-spiral. AutoAscend emergency-eats/prays only at FAINTING, by which point
# it is disoriented ("too disoriented for this") and cannot act -> faint loop ->
# death. We replace emergency_strategy with the same logic fired one hunger stage
# earlier (WEAK), so prayer/quaff succeeds before incapacitation. Keeps the HP
# -potion branch unchanged. Source stays byte-frozen; we own the whole function.

def apply_anti_starvation():
    from autoascend.agent import Agent
    from autoascend.strategy import Strategy
    from autoascend.glyph import Hunger
    import autoascend.agent as _ag
    import nle.nethack as nh
    flatten_items = _ag.flatten_items

    @Strategy.wrap
    def emergency_strategy(self):
        bl = self.blstats
        # HP emergency potions (unchanged from AutoAscend)
        heal = [i for i in flatten_items(self.inventory.items) if i.is_unambiguous()
                and i.category == nh.POTION_CLASS
                and i.object.name in ("healing", "extra healing", "full healing")]
        if (bl.hitpoints < bl.max_hitpoints / 3 or bl.hitpoints < 8) and heal:
            yield True
            self.inventory.quaff(heal[0])
            return

        # LORE: act at WEAK (was FAINTING) so we never reach disorientation
        hungry = int(bl.hunger_state) >= int(Hunger.WEAK)

        juice = [i for i in flatten_items(self.inventory.items) if i.is_unambiguous()
                 and i.category == nh.POTION_CLASS and i.object.name == "fruit juice"]
        if juice and hungry:
            yield True
            _bump("starv_weak_juice")
            self.inventory.quaff(juice[0])
            return

        if ((self.is_safe_to_pray(500) and
             (bl.hitpoints < bl.max_hitpoints / (5 if bl.experience_level < 6 else 6) or bl.hitpoints < 6))
                or (self.is_safe_to_pray(400) and hungry)):
            yield True
            if hungry:
                _bump("starv_weak_pray")
            self.pray()
            return

        yield False

    Agent.emergency_strategy = emergency_strategy
    return ["anti_starvation (emergency at WEAK)"]


def apply_decode_hardening():
    """The deepest runs hit a non-utf8 byte in a message -> bytes.decode() crash
    -> Cyclic Panic (killed the 107k/DL26 run). Sanitize obs byte arrays (clip
    >127 to space) only on the rare decode failure, then re-parse."""
    import numpy as np
    from autoascend.agent import Agent
    orig = Agent.get_message_and_popup

    def safe(self, obs):
        try:
            return orig(self, obs)
        except UnicodeDecodeError:
            o = dict(obs)
            for k in ("message", "tty_chars"):
                if k in o:
                    a = np.array(o[k]); a[a > 127] = 32; o[k] = a
            _bump("decode_sanitize")
            return orig(self, o)

    Agent.get_message_and_popup = safe

    # Per-step status-line decodes (character.py) crash strict-utf8 on a non-utf8
    # byte deep -> this was the real recurring crash (Cyclic Panic ['utf-8']).
    try:
        from autoascend.character import Character
        def _mk(tok):
            return property(lambda self: tok in bytes(
                self.agent.last_observation["tty_chars"][-1]).decode("utf-8", "replace"))
        for nm, tok in [("confusion", "Conf"), ("stun", "Stun"),
                        ("hallu", "Hallu"), ("blind", "Blind")]:
            setattr(Character, nm, _mk(tok))
    except Exception:
        pass
    return ["decode_hardening (msg + status utf8 sanitize)"]


def apply_obs_sanitize():
    """Global decode fix: clip >127 bytes in the obs text arrays at the single
    point they enter the agent (Agent.update), so EVERY downstream bytes.decode()
    (message, tty, inv_strs, glyph char map) is safe. The deep 'utf-8' Cyclic
    Panic came from a non-utf8 byte the per-site patches kept missing."""
    import numpy as np
    from autoascend.agent import Agent
    orig = Agent.update
    KEYS = ("message", "tty_chars", "inv_strs", "chars", "inv_letters")

    def safe_update(self, observation, *a, **k):
        try:
            o = observation
            copied = False
            for key in KEYS:
                arr = o.get(key) if hasattr(o, "get") else None
                if arr is not None and getattr(arr, "dtype", None) == np.uint8 and (arr > 127).any():
                    if not copied:
                        o = dict(o); copied = True
                    a2 = arr.copy(); a2[a2 > 127] = 32; o[key] = a2
            if copied:
                _bump("obs_sanitize")
                observation = o
        except Exception:
            pass
        return orig(self, observation, *a, **k)

    Agent.update = safe_update
    return ["obs_sanitize (clip >127 in obs text arrays)"]
