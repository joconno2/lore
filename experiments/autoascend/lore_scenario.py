"""Wizard-mode scenario harness: drop the AutoAscend agent onto a target deep
level to test late-game capabilities controllably (the reach bottleneck). The
agent inits normally (intro handled), then a one-time ^V level-teleport fires
before the strategy loop, so it plays from the target depth. Requires a
wizard-mode env (wizard=True)."""
import lore_patches


def _do_wish(agent, item):
    """Wizard #wish for one item via raw low-level keypresses (on the current,
    stable level -- call before teleport). Lets us equip a realistic late-game
    kit so the agent survives ordinary threats and reaches the knowledge-
    dependent deaths LORE targets, instead of dying naked to raw damage."""
    low = agent.env.env.unwrapped.env
    low.step(23)                       # ^W = wizard wish (NOT #wish, which is unknown)
    for ch in item:                    # -> "For what do you wish?" getlin
        low.step(ord(ch))
    low.step(13)                       # submit wish
    low.step(13); low.step(13)         # clear --More--


def _eat_for_intrinsics(agent):
    """Eat wished corpses on the safe start level to gain intrinsic resistances
    (real gameplay prep, not a wizard cheat): killer-bee/kobold corpses confer
    poison resistance. The Valley of the Dead kills equipped chars via vampire
    POISONED bites -- poison res is the missing piece. Eats every FOOD corpse in
    inventory via raw 'e' keypresses."""
    import nle.nethack as _nh
    from autoascend.agent import flatten_items
    low = agent.env.env.unwrapped.env
    try:
        agent.step(__import__("autoascend.agent", fromlist=["A"]).A.Command.ESC)
        agent.inventory.update()
    except Exception:
        pass
    eaten = 0
    for _ in range(20):
        corpse = None
        for it in flatten_items(agent.inventory.items):
            if getattr(it, "category", None) == _nh.FOOD_CLASS and getattr(it, "is_corpse", lambda: False)():
                corpse = it
                break
        if corpse is None:
            break
        try:
            letter = agent.inventory.items.get_letter(corpse)
        except Exception:
            break
        low.step(ord('e'))                 # eat
        # if it asks about floor food first ('y'/'n'), decline then pick inventory
        low.step(ord(letter))              # choose the corpse
        low.step(ord('y'))                 # "eat anyway?" / confirm
        low.step(13); low.step(13)         # clear --More--
        eaten += 1
        try:
            agent.step(__import__("autoascend.agent", fromlist=["A"]).A.Command.ESC)
            agent.inventory.update()
        except Exception:
            break
    lore_patches.COUNTERS["corpses_eaten"] = eaten


def _equip_endgame(agent):
    """Equip the wished kit on the SAFE start level before teleporting. AutoAscend
    wears armor (wear_best_stuff) but has NO ring/amulet logic at all -- so the
    wished amulet of reflection + rings (free action, fire resistance) would sit
    unused. Wear armor via AA's routine, then PUT ON amulet + rings via raw 'P'
    keypresses. Without this the char lands in Gehennom at ~AC10, unprotected."""
    import nle.nethack as _nh
    import autoascend.agent as _A
    # 1) armor (gray dragon scale mail etc.) via AutoAscend's own routine
    try:
        agent.inventory.update()
        s = agent.inventory.wear_best_stuff()
        if s.check_condition():
            s.run()
    except Exception:
        pass
    try:
        agent.step(_A.A.Command.ESC)
        agent.inventory.update()
    except Exception:
        pass
    # 2) amulet + rings via the TRACKED interface (atom_operation + type_text),
    #    modelled on AutoAscend's own wear(). The old raw low.step('P')+letter
    #    sequence desynced and CORRUPTED inventory (destroyed wished wands). Read
    #    letters from the raw observation (wished items are unidentified, so
    #    get_letter/flatten_items are unreliable).
    amulets, rings = [], []
    try:
        obs = agent.last_observation
        for oc, lt in zip(obs['inv_oclasses'], obs['inv_letters']):
            if int(lt) == 0:
                continue
            c, l = int(oc), chr(int(lt))
            if c == _nh.AMULET_CLASS:
                amulets.append(l)
            elif c == _nh.RING_CLASS:
                rings.append(l)
    except Exception:
        pass
    worn = 0
    for l in amulets[:1]:                       # one amulet slot (reflection)
        try:
            with agent.atom_operation():
                agent.step(_A.A.Command.PUTON)
                agent.type_text(l)
            worn += 1
        except Exception:
            pass
    lore_patches.COUNTERS["ring_count"] = len(rings)
    # Rings via RAW keypresses: AutoAscend's agent.step ASSERTS on ring messages
    # ("...on right hand") -- it has no ring logic. Raw low.step bypasses that.
    # Correct letters (from obs) mean no inventory corruption; worst case a stray
    # move if a finger prompt doesn't appear. P, letter, finger, clear.
    low = agent.env.env.unwrapped.env
    for l, finger in zip(rings[:2], ['r', 'l']):
        try:
            low.step(ord('P'))             # PUTON
            low.step(ord(l))               # ring letter
            low.step(ord(finger))          # answer "Which ring-finger?" (or stray move)
            low.step(13)                   # clear --More--
            worn += 1
        except Exception as _re:
            lore_patches.COUNTERS["ring_err"] = repr(_re)[:90]
    try:
        agent.step(_A.A.Command.ESC)
        agent.inventory.update()
    except Exception:
        pass
    try:
        agent.step(_A.A.Command.ESC)
        agent.inventory.update()
    except Exception:
        pass
    lore_patches.COUNTERS["equipped_jewelry"] = worn
    try:
        lore_patches.COUNTERS["ac_after_equip"] = int(agent.blstats.armor_class)
    except Exception:
        pass


def _do_teleport(agent, target_depth):
    # ^V isn't in AutoAscend's action space, so issue the wizard level-teleport
    # at the LOW-LEVEL nethack (raw keypresses), then resync the agent with one
    # normal step. low-level env: gym_env.unwrapped.env (NLE issues keypresses
    # via self.env.step(keypress)).
    genv = agent.env.env
    low = genv.unwrapped.env
    low.step(27); low.step(27)         # ESC x2: clear any residual prompt (e.g. post-wish)
    low.step(22)                       # ^V
    for ch in str(int(target_depth)):
        low.step(ord(ch))              # digits
    low.step(13)                       # enter
    low.step(13); low.step(13)         # clear any --More--
    import autoascend.agent as _ag
    agent.step(_ag.A.Command.ESC)      # resync agent state with new level
    try:
        agent.levels.clear()           # wipe stale DL1 stair graph (-> PLANE)
    except Exception:
        pass
    try:
        lore_patches.COUNTERS["scenario_depth"] = int(agent.blstats.depth)
    except Exception:
        pass


def install_teleport(target_depth):
    from autoascend import global_logic as _gl

    orig = _gl.GlobalLogic.global_strategy

    def _do_teleport_OLD(agent):
        genv = agent.env.env
        low = genv.unwrapped.env
        low.step(22)                       # ^V
        for ch in str(int(target_depth)):
            low.step(ord(ch))              # digits
        low.step(13)                       # enter
        low.step(13); low.step(13)         # clear any --More--
        # resync: one normal agent step (ESC) so its state reflects the new level
        import autoascend.agent as _ag
        agent.step(_ag.A.Command.ESC)
        # clear stale level tracking: init recorded a bogus DL1 stair_destination
        # (-> PLANE) before the jump; wipe it so the agent rebuilds fresh for the
        # teleported level and cross-level pathing doesn't reference dead state.
        try:
            agent.levels.clear()
        except Exception:
            pass
        try:
            lore_patches.COUNTERS["scenario_depth"] = int(agent.blstats.depth)
        except Exception:
            pass

    def patched(self):
        if not getattr(self.agent, "_lore_tp_done", False):
            self.agent.__dict__["_lore_tp_done"] = True
            try:
                _do_teleport(self.agent, target_depth)
                lore_patches._bump("scenario_teleport")
                # After teleport the milestone is still BE_ON_FIRST_LEVEL, so the
                # agent tries to navigate BACK UP to DL1 and hangs exhaustively
                # searching the deep level for a path that isn't there. Set the
                # forward (deepest) milestone so it plays FROM the teleported
                # level -- explore + descend -- instead of returning to DL1.
                try:
                    self.milestone = _gl.Milestone.GO_DOWN
                except Exception:
                    pass
            except Exception as e:
                lore_patches._bump("scenario_teleport_fail")
        return orig(self)

    _gl.GlobalLogic.global_strategy = patched
    return ["scenario_teleport(DL%d)" % target_depth]


def install_scenario(target_depth, wishes=()):
    """Replace global_strategy entirely for the scenario. AutoAscend's cross-level
    navigation (go_to_level_strategy) HANGS post-teleport: levels.clear() wipes the
    stair graph, so it explores forever looking for an unmapped path, and
    open_visit_search(search_prio_limit=None) spins. We swap it for BOUNDED local
    play -- explore1(0) (terminates) + the combat/survival preempts (fight2,
    engulfed_fight, eat, emergency). Tests late-game survival without navigation.
    wishes: list of #wish item strings, granted before teleport to equip a
    realistic late-game kit."""
    from autoascend import global_logic as _gl
    from autoascend.strategy import Strategy
    from autoascend.glyph import Hunger

    def scenario_global(self):
        agent = self.agent
        if not getattr(agent, "_lore_tp_done", False):
            agent.__dict__["_lore_tp_done"] = True
            try:
                # ALL setup on the SAFE start level (DL1), THEN teleport. Doing
                # the wish/level-up at the deep target got the agent killed mid-
                # keypress (defenseless) and left flaky XL. Order: wish -> quaff
                # gain-level (level up safely) -> teleport down already strong.
                import autoascend.agent as _ag3
                import nle.nethack as _nh
                for it in wishes:
                    _do_wish(agent, it)
                lore_patches.COUNTERS["wishes"] = len(wishes)
                # Resync inventory, then quaff every wished potion (gain-level,
                # unidentified -> match by POTION category). Re-fetch each time.
                try:
                    agent.step(_ag3.A.Command.ESC)
                    agent.inventory.update()
                except Exception:
                    pass
                for _ in range(40):
                    pot = None
                    for it in _ag3.flatten_items(agent.inventory.items):
                        if getattr(it, "category", None) == _nh.POTION_CLASS:
                            pot = it
                            break
                    if pot is None:
                        break
                    try:
                        agent.inventory.quaff(pot)
                    except Exception:
                        break
                lore_patches.COUNTERS["xl_before_tp"] = int(agent.blstats.experience_level)
                _do_teleport(agent, target_depth)
                lore_patches._bump("scenario_teleport")
                lore_patches.COUNTERS["tp_depth"] = int(agent.blstats.depth)
                lore_patches.COUNTERS["xl_after"] = int(agent.blstats.experience_level)
            except Exception as e:
                lore_patches._bump("scenario_teleport_fail")
                try:
                    import traceback
                    open("/workspace/tp_err.txt", "w").write(repr(e) + "\n" + traceback.format_exc())
                except Exception:
                    pass
        # Base loop must NEVER return -- AutoAscend's main() does `assert 0` after
        # global_strategy().run(). Explore (bounded) when there's work, else
        # search/wait so the loop is infinite. Combat preempts handle monsters.
        @Strategy.wrap
        def survive(s):
            # Never returns. AgentFinished (death) / AgentPanic propagate to
            # main() which handles them -- do NOT swallow (swallowing AgentFinished
            # then stepping a dead env raised "Called step on finished NetHack").
            yield True
            while 1:
                e = agent.exploration.explore1(0)
                if e.check_condition():
                    e.run()
                else:
                    agent.search(1)  # nothing to explore -> wait/search a turn

        return (
            survive(self)
            .preempt(agent, [agent.inventory.wear_best_stuff()])  # equip wished gear
            .preempt(agent, [agent.eat_corpses_from_ground(only_below_me=True)
                             .condition(lambda: agent.blstats.hunger_state >= Hunger.NOT_HUNGRY)])
            .preempt(agent, [agent.fight2()])
            .preempt(agent, [agent.engulfed_fight()])
            .preempt(agent, [agent.emergency_strategy()])
        )

    _gl.GlobalLogic.global_strategy = scenario_global
    return ["scenario_global(DL%d, bounded-local)" % target_depth]


def install_descent(target_depth, wishes=()):
    """Endgame DESCENT planner. AutoAscend's dungeon model (level.py) knows only
    DoD/Mines/Sokoban/Quest -- no Gehennom, no planes -- so its go_to_level
    routing can't reach the endgame. But get_stairs(down=True) works on ANY
    current level. So we drive descent model-free: explore the current level,
    find the downstair, take it, repeat. This pushes the agent DEEPER through
    Gehennom than AutoAscend structurally can (its GO_DOWN is an unimplemented
    TODO). Tracks max depth reached. Setup (wish + level-up) on safe DL1 first."""
    from autoascend import global_logic as _gl
    from autoascend.strategy import Strategy
    from autoascend.glyph import Hunger, G
    from autoascend import utils as _u

    from autoascend.exceptions import AgentPanic, AgentFinished
    from autoascend import exceptions as _gl_exc
    from autoascend.agent import flatten_items
    from autoascend import objects as _O
    import nle.nethack as _nh2

    def descent_global(self):
        agent = self.agent
        if not getattr(agent, "_lore_tp_done", False):
            agent.__dict__["_lore_tp_done"] = True
            try:
                import autoascend.agent as _ag3
                import nle.nethack as _nh
                for it in wishes:
                    _do_wish(agent, it)
                lore_patches.COUNTERS["wishes"] = len(wishes)
                try:
                    agent.step(_ag3.A.Command.ESC)
                    agent.inventory.update()
                except Exception:
                    pass
                for _ in range(40):
                    pot = None
                    for it in _ag3.flatten_items(agent.inventory.items):
                        if getattr(it, "category", None) == _nh.POTION_CLASS:
                            pot = it
                            break
                    if pot is None:
                        break
                    try:
                        agent.inventory.quaff(pot)
                    except Exception:
                        break
                lore_patches.COUNTERS["xl_before_tp"] = int(agent.blstats.experience_level)
                import os as _os
                if _os.environ.get("LORE_NO_EAT") != "1":
                    _eat_for_intrinsics(agent)  # poison res from wished corpses
                if _os.environ.get("LORE_NO_EQUIP") != "1":
                    _equip_endgame(agent)       # wear armor + put on amulet/rings
                _do_teleport(agent, target_depth)
                lore_patches._bump("scenario_teleport")
                lore_patches.COUNTERS["tp_depth"] = int(agent.blstats.depth)
                lore_patches.COUNTERS["xl_after"] = int(agent.blstats.experience_level)
                lore_patches.COUNTERS["max_depth"] = int(agent.blstats.depth)
                lore_patches.COUNTERS["descents"] = 0
            except Exception as e:
                lore_patches._bump("scenario_teleport_fail")
                try:
                    import traceback
                    open("/workspace/tp_err.txt", "w").write(repr(e) + "\n" + traceback.format_exc())
                except Exception:
                    pass

        @Strategy.wrap
        def descend(s):
            yield True
            while 1:
                lore_patches.COUNTERS["descend_iters"] = \
                    lore_patches.COUNTERS.get("descend_iters", 0) + 1
                d = int(agent.blstats.depth)
                if d > lore_patches.COUNTERS.get("max_depth", 0):
                    lore_patches.COUNTERS["max_depth"] = d
                # diagnostic: is a downstair glyph mapped, and how much explored?
                try:
                    lvl = agent.current_level()
                    nd = int(_u.isin(lvl.objects, G.STAIR_DOWN).sum())
                    lore_patches.COUNTERS["downstair_glyphs"] = max(
                        lore_patches.COUNTERS.get("downstair_glyphs", 0), nd)
                    lore_patches.COUNTERS["explored_cells"] = int(lvl.was_on.sum())
                    lore_patches.COUNTERS["dungeon_num"] = int(lvl.dungeon_number)
                except Exception:
                    pass
                # 0) DIG-DOWN: zap a wand of digging downward to fall to the next
                #    level. Bypasses Gehennom's broken stair navigation entirely
                #    (the agent gets stuck in a ~17-cell pocket and never maps the
                #    downstair). Real NetHack descent technique. Primary in Gehennom.
                try:
                    lore_patches.COUNTERS["dig_block_ran"] = \
                        lore_patches.COUNTERS.get("dig_block_ran", 0) + 1
                    try:
                        import autoascend.agent as _agz0
                        agent.step(_agz0.A.Command.ESC)   # clear any stale prompt
                        agent.inventory.update()          # fresh letters post-teleport
                    except Exception:
                        pass
                    # Read the wand letter straight from the raw observation
                    # (inv_oclasses/inv_letters) -- robust to AutoAscend's item
                    # object identity. We only wished digging wands.
                    # find a wand letter from the raw observation (wished wands
                    # show as unidentified; we only wished digging wands).
                    def _wand_letter():
                        lt = None
                        for oc, l in zip(agent.last_observation['inv_oclasses'],
                                         agent.last_observation['inv_letters']):
                            if int(oc) == _nh2.WAND_CLASS and int(l) != 0:
                                lt = chr(int(l))
                        return lt
                    letter = _wand_letter()
                    lore_patches.COUNTERS["wands_seen"] = 1 if letter else 0
                    if letter is not None:
                        before = int(agent.blstats.depth)
                        import autoascend.agent as _agz
                        # Step off any staircase first: a dig beam zapped down from
                        # a stair BOUNCES (it doesn't dig). ^V often lands the agent
                        # ON the up-stair, which is why the first dig failed.
                        try:
                            y0, x0 = agent.blstats.y, agent.blstats.x
                            lvl0 = agent.current_level()
                            if lvl0.objects[y0, x0] in G.STAIR_UP or \
                                    lvl0.objects[y0, x0] in G.STAIR_DOWN:
                                bf = agent.bfs()
                                for ny, nx in agent.neighbors(y0, x0):
                                    if bf[ny, nx] != -1 and lvl0.walkable[ny, nx] and \
                                            lvl0.objects[ny, nx] not in G.STAIR_UP and \
                                            lvl0.objects[ny, nx] not in G.STAIR_DOWN:
                                        agent.move(ny, nx)
                                        lore_patches.COUNTERS["stepped_off_stair"] = \
                                            lore_patches.COUNTERS.get("stepped_off_stair", 0) + 1
                                        break
                        except Exception:
                            pass
                        # Zap a wand of digging downward via the agent's tracked
                        # interface with the KNOWN letter (agent.zap's get_letter
                        # desyncs on wished wands; the direction prompt wants the
                        # '>' CHAR, not the DOWN action). Digging makes a PIT first
                        # then a HOLE -> re-zap until depth drops or charges run out.
                        for _zi in range(6):
                            lt2 = _wand_letter()
                            if lt2 is None:
                                break
                            with agent.atom_operation():
                                agent.step(_agz.A.Command.ZAP)
                                agent.type_text(lt2)
                                agent.type_text('>')
                            lore_patches.COUNTERS["zap_msg"] = str(agent.message)[:120]
                            if int(agent.blstats.depth) > before:
                                break
                        if int(agent.blstats.depth) > before:
                            lore_patches.COUNTERS["digs"] = \
                                lore_patches.COUNTERS.get("digs", 0) + 1
                            lore_patches.COUNTERS["descents"] = \
                                lore_patches.COUNTERS.get("descents", 0) + 1
                            try:
                                agent.levels.clear()  # fresh map for the new level
                            except Exception:
                                pass
                            continue
                        else:
                            # no charge / dig failed -> stop trying the wand
                            lore_patches.COUNTERS["dig_fail"] = \
                                lore_patches.COUNTERS.get("dig_fail", 0) + 1
                except AgentFinished:
                    raise
                except AgentPanic:
                    lore_patches.COUNTERS["dig_panic"] = \
                        lore_patches.COUNTERS.get("dig_panic", 0) + 1
                except RuntimeError:
                    raise
                except _gl_exc.AgentChangeStrategy:
                    raise                       # control-flow signal, must propagate
                except BaseException as _de:
                    lore_patches.COUNTERS["dig_exc"] = \
                        lore_patches.COUNTERS.get("dig_exc", 0) + 1
                    lore_patches.COUNTERS["dig_exc_msg"] = repr(_de)[:120]
                # 1) try AutoAscend's own stair primitive: go to a discovered,
                #    unexplored DOWNstair and take it (battle-tested navigation).
                took = False
                try:
                    st = agent.exploration.explore_stairs(
                        agent.exploration.go_to_strategy, down=True)
                    if st.check_condition():
                        lore_patches.COUNTERS["stairs_seen"] = \
                            lore_patches.COUNTERS.get("stairs_seen", 0) + 1
                        before = agent.current_level().key()
                        st.run()
                        if agent.current_level().key() != before:
                            lore_patches.COUNTERS["descents"] = \
                                lore_patches.COUNTERS.get("descents", 0) + 1
                        took = True
                except AgentFinished:
                    raise                       # death: let main() end the game
                except AgentPanic:
                    # recoverable navigation hiccup (monster, blocked path) --
                    # AutoAscend re-plans on the next loop. Count it, carry on.
                    lore_patches.COUNTERS["descend_err"] = \
                        lore_patches.COUNTERS.get("descend_err", 0) + 1
                except RuntimeError:
                    raise                       # finished-env / cyclic panic: propagate
                if took:
                    continue
                # 2) no known downstair -> explore this level to discover one.
                # explore1/search raise AgentFinished on death -> propagate.
                e = agent.exploration.explore1(0)
                if e.check_condition():
                    lore_patches.COUNTERS["explore1_ran"] = \
                        lore_patches.COUNTERS.get("explore1_ran", 0) + 1
                    e.run()
                else:
                    lore_patches.COUNTERS["search_ran"] = \
                        lore_patches.COUNTERS.get("search_ran", 0) + 1
                    agent.search(1)

        def _count(strat, name):
            """Wrap a Strategy to bump COUNTERS['pre_<name>'] each time it actually
            ACTIVATES (condition True -> body runs), so we see which preempt eats
            control while descend starves."""
            orig_factory = strat.strategy
            def f():
                gen = orig_factory()
                cond = next(gen)
                yield cond
                if not cond:
                    return
                lore_patches.COUNTERS["pre_" + name] = \
                    lore_patches.COUNTERS.get("pre_" + name, 0) + 1
                try:
                    next(gen); assert 0
                except StopIteration as _e:
                    return _e.value
            return Strategy(f, strat.config)

        return (
            descend(self)
            .preempt(agent, [_count(agent.inventory.wear_best_stuff(), "wear")])
            .preempt(agent, [_count(agent.eat_corpses_from_ground(only_below_me=True)
                             .condition(lambda: agent.blstats.hunger_state >= Hunger.NOT_HUNGRY), "eat")])
            .preempt(agent, [_count(agent.fight2(), "fight2")])
            .preempt(agent, [_count(agent.engulfed_fight(), "engulf")])
            .preempt(agent, [_count(agent.emergency_strategy(), "emergency")])
        )

    _gl.GlobalLogic.global_strategy = descent_global
    return ["descent_global(DL%d, model-free descend)" % target_depth]


def patch_enhance_noop():
    """Wizard-mode #enhance view breaks AutoAscend's skill parser at init
    ('bare handed combat' line). Skill-enhancing is secondary for scenario tests
    (we control the kit), so no-op it to let init proceed."""
    from autoascend.character import Character
    Character.parse_enhance_view = lambda self: None
    Character.parse_enhance = lambda self, *a, **k: None if hasattr(Character, "parse_enhance") else None
    return ["enhance_noop"]
