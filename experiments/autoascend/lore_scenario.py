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


def _equip_endgame(agent):
    """Equip the wished kit on the SAFE start level before teleporting. AutoAscend
    wears armor (wear_best_stuff) but has NO ring/amulet logic at all -- so the
    wished amulet of reflection + rings (free action, fire resistance) would sit
    unused. Wear armor via AA's routine, then PUT ON amulet + rings via raw 'P'
    keypresses. Without this the char lands in Gehennom at ~AC10, unprotected."""
    import nle.nethack as _nh
    low = agent.env.env.unwrapped.env
    # 1) armor (gray dragon scale mail etc.) via AutoAscend's own routine
    try:
        agent.inventory.update()
        s = agent.inventory.wear_best_stuff()
        if s.check_condition():
            s.run()
    except Exception:
        pass
    # 2) amulet + rings via raw PUTON. 'P' (80) -> "What do you want to put on?"
    #    -> item letter -> (rings only) "Which finger?" r/l. Re-fetch each time.
    try:
        agent.step(__import__("autoascend.agent", fromlist=["A"]).A.Command.ESC)
        agent.inventory.update()
    except Exception:
        pass
    from autoascend.agent import flatten_items
    worn = 0
    finger = ord('r')
    for cat in (_nh.AMULET_CLASS, _nh.RING_CLASS, _nh.RING_CLASS):
        item = None
        for it in flatten_items(agent.inventory.items):
            if getattr(it, "category", None) == cat and not getattr(it, "equipped", False):
                # skip an already-worn item of same letter by checking 'at use'
                item = it
                break
        if item is None:
            continue
        try:
            letter = agent.inventory.items.get_letter(item)
        except Exception:
            letter = None
        if letter is None:
            continue
        low.step(80)                       # P (put on)
        low.step(ord(letter))              # which item
        if cat == _nh.RING_CLASS:
            low.step(finger)               # right then left finger
            finger = ord('l')
        low.step(13); low.step(13)         # clear prompts/--More--
        worn += 1
    try:
        agent.step(__import__("autoascend.agent", fromlist=["A"]).A.Command.ESC)
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
                _equip_endgame(agent)      # wear armor + put on amulet/rings on safe DL1
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
                    e.run()
                else:
                    agent.search(1)

        return (
            descend(self)
            .preempt(agent, [agent.inventory.wear_best_stuff()])
            .preempt(agent, [agent.eat_corpses_from_ground(only_below_me=True)
                             .condition(lambda: agent.blstats.hunger_state >= Hunger.NOT_HUNGRY)])
            .preempt(agent, [agent.fight2()])
            .preempt(agent, [agent.engulfed_fight()])
            .preempt(agent, [agent.emergency_strategy()])
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
