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


def _spawn_monsters(agent, names):
    """Wizard ^G (create monster, keycode 7) -- spawn named monsters next to the
    agent for a CONTROLLED knowledge-gated threat test. ^G prompts 'Create what
    kind of monster?' (getlin); the monster appears adjacent."""
    import autoascend.agent as _A
    low = agent.env.env.unwrapped.env
    spawned = 0
    for nm in names:
        low.step(7)                      # ^G
        for ch in nm:
            low.step(ord(ch))
        low.step(13)                     # submit name
        low.step(13)                     # clear any --More--
        spawned += 1
    try:
        r = agent.env.env.unwrapped.env
        lore_patches.COUNTERS["spawn_msg"] = bytes(
            agent.last_observation['message']).decode('latin1').strip('\x00').strip()[:80] \
            if False else lore_patches.COUNTERS.get("spawn_msg", "")
    except Exception:
        pass
    try:
        agent.step(_A.A.Command.ESC)
        agent.inventory.update()
    except Exception:
        pass
    lore_patches.COUNTERS["spawned"] = spawned


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
    # Then eat FOOD RATIONS for nutrition, but ONLY while genuinely hungry --
    # eating when already Satiated makes the char CHOKE TO DEATH (killed the whole
    # setup: 'Called step on finished NetHack' at teleport). Stop at hunger_state<2.
    rations = 0
    for _ in range(8):
        try:
            if int(agent.blstats.hunger_state) < 2:   # 0 satiated, 1 normal -> stop
                break
        except Exception:
            break
        ration = None
        for it in flatten_items(agent.inventory.items):
            if getattr(it, "category", None) == _nh.FOOD_CLASS and \
                    not getattr(it, "is_corpse", lambda: False)():
                ration = it
                break
        if ration is None:
            break
        try:
            letter = agent.inventory.items.get_letter(ration)
        except Exception:
            break
        low.step(ord('e')); low.step(ord(letter)); low.step(ord('y'))
        low.step(13); low.step(13)
        rations += 1
        try:
            agent.step(__import__("autoascend.agent", fromlist=["A"]).A.Command.ESC)
            agent.inventory.update()
        except Exception:
            break
    lore_patches.COUNTERS["rations_eaten"] = rations


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
    # Wearing the wished rings corrupts AutoAscend's ring-less inventory model:
    # a later agent.step parses the "...on right hand" message and ASSERTS,
    # crashing the whole descent. Armor (the AC that keeps us alive) is worn
    # separately via AA's own routine, so skip rings when LORE_NO_RINGS=1 to keep
    # descending. (Loses free-action/fire-res; fixable later by patching AA's
    # inventory parser, but rings aren't needed to reach the vibrating square.)
    import os as _osr
    low = agent.env.env.unwrapped.env
    for l, finger in zip([] if _osr.environ.get("LORE_NO_RINGS") == "1" else rings[:2], ['r', 'l']):
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


def install_threat(target_depth, wishes=(), spawn=()):
    """Knowledge-gated THREAT scenario: place a vulnerable char (armor for AC but
    BARE HANDS, no ranged) at a depth, spawn instant-death threats (cockatrice /
    floating eye) via ^G, and run AutoAscend's NATIVE combat. Arms (set externally
    via apply_oracle_veto): base AA | mock-veto | llm-veto. Metric = survival
    (turns alive). This is where wiki knowledge should matter: meleeing a
    cockatrice barehanded = petrification death; floating eye = paralysis death."""
    from autoascend import global_logic as _gl
    from autoascend.strategy import Strategy
    from autoascend.glyph import Hunger

    def threat_global(self):
        agent = self.agent
        if not getattr(agent, "_lore_tp_done", False):
            agent.__dict__["_lore_tp_done"] = True
            try:
                import autoascend.agent as _ag3
                import nle.nethack as _nh
                for it in wishes:
                    _do_wish(agent, it)
                try:
                    agent.step(_ag3.A.Command.ESC); agent.inventory.update()
                except Exception:
                    pass
                for _ in range(40):                 # quaff gain-level potions
                    pot = None
                    for it in _ag3.flatten_items(agent.inventory.items):
                        if getattr(it, "category", None) == _nh.POTION_CLASS:
                            pot = it; break
                    if pot is None:
                        break
                    try:
                        agent.inventory.quaff(pot)
                    except Exception:
                        break
                try:                                # wear armor (AC), no gloves
                    s = agent.inventory.wear_best_stuff()
                    if s.check_condition():
                        s.run()
                except Exception:
                    pass
                lore_patches.COUNTERS["xl_after"] = int(agent.blstats.experience_level)
                _do_teleport(agent, target_depth)
                lore_patches.COUNTERS["tp_depth"] = int(agent.blstats.depth)
                if spawn:
                    _spawn_monsters(agent, spawn)
            except Exception:
                lore_patches._bump("scenario_teleport_fail")

        @Strategy.wrap
        def survive(s):
            yield True
            while 1:
                e = agent.exploration.explore1(0)
                if e.check_condition():
                    e.run()
                else:
                    agent.search(1)

        return (
            survive(self)
            .preempt(agent, [agent.eat_corpses_from_ground(only_below_me=True)
                             .condition(lambda: agent.blstats.hunger_state >= Hunger.NOT_HUNGRY)])
            .preempt(agent, [agent.fight2()])
            .preempt(agent, [agent.engulfed_fight()])
            .preempt(agent, [agent.emergency_strategy()])
        )

    _gl.GlobalLogic.global_strategy = threat_global
    return ["threat_global(DL%d, spawn=%s)" % (target_depth, list(spawn))]


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
    from autoascend.glyph import MON as _MON
    import nle.nethack as _nh2
    import oracle as _oracle
    patch_ring_parse()   # let AA tolerate worn rings/amulet without crashing

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
                try: lore_patches.COUNTERS["t_after_wishes"] = int(agent.blstats.time)
                except Exception: pass
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
                try: lore_patches.COUNTERS["t_after_quaff"] = int(agent.blstats.time)
                except Exception: pass
                import os as _os
                if _os.environ.get("LORE_NO_EAT") != "1":
                    _eat_for_intrinsics(agent)  # poison res from wished corpses
                try: lore_patches.COUNTERS["t_after_eat"] = int(agent.blstats.time)
                except Exception: pass
                if _os.environ.get("LORE_NO_EQUIP") != "1":
                    _equip_endgame(agent)       # wear armor + put on amulet/rings
                try: lore_patches.COUNTERS["t_after_equip"] = int(agent.blstats.time)
                except Exception: pass
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

        # ---- primitives = the LLM's action space (each returns/raises normally) --
        import autoascend.agent as _agz
        import os as _os2

        def _wand_letter():
            for oc, l in zip(agent.last_observation['inv_oclasses'],
                             agent.last_observation['inv_letters']):
                if int(oc) == _nh2.WAND_CLASS and int(l) != 0:
                    return chr(int(l))
            return None

        def _step_off_stair():
            y0, x0 = agent.blstats.y, agent.blstats.x
            lvl0 = agent.current_level()
            if lvl0.objects[y0, x0] in G.STAIR_UP or lvl0.objects[y0, x0] in G.STAIR_DOWN:
                bf = agent.bfs()
                for ny, nx in agent.neighbors(y0, x0):
                    if bf[ny, nx] != -1 and lvl0.walkable[ny, nx] and \
                            lvl0.objects[ny, nx] not in G.STAIR_UP and \
                            lvl0.objects[ny, nx] not in G.STAIR_DOWN:
                        agent.move(ny, nx)
                        break

        def prim_dig_down():
            """zap wand of digging down -> fall a level. returns True if descended."""
            try:
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
            except Exception:
                pass
            if _wand_letter() is None:
                return False
            before = int(agent.blstats.depth)
            try:
                _step_off_stair()
            except Exception:
                pass
            for _zi in range(6):
                lt2 = _wand_letter()
                if lt2 is None:
                    break
                with agent.atom_operation():
                    agent.step(_agz.A.Command.ZAP); agent.type_text(lt2); agent.type_text('>')
                lore_patches.COUNTERS["zap_msg"] = str(agent.message)[:120]
                if 'too hard' in agent.message:
                    lore_patches.COUNTERS["level_no_dig"] = int(agent.blstats.depth)
                    break
                if int(agent.blstats.depth) > before:
                    break
            if int(agent.blstats.depth) > before:
                lore_patches.COUNTERS["digs"] = lore_patches.COUNTERS.get("digs", 0) + 1
                try:
                    agent.levels.clear()
                except Exception:
                    pass
                return True
            lore_patches.COUNTERS["dig_fail"] = lore_patches.COUNTERS.get("dig_fail", 0) + 1
            return False

        def prim_descend_stairs():
            """MODEL-FREE: BFS to a down-stair GLYPH and take it via raw '>'. AA's
            explore_stairs relies on its dungeon model, which has NO Gehennom, so it
            never reaches the maze downstairs -- the endgame-reach blocker. The glyph
            grid + agent.bfs() work regardless of dungeon, so navigate directly."""
            try:
                lvl0 = agent.current_level()
                bf = agent.bfs()
                downs = list(zip(*((_u.isin(lvl0.objects, G.STAIR_DOWN)) & (bf != -1)).nonzero()))
            except Exception:
                return False
            if not downs:
                return False
            before = int(agent.blstats.depth)
            y, x = downs[0]
            try:
                if (int(agent.blstats.y), int(agent.blstats.x)) != (int(y), int(x)):
                    agent.go_to(int(y), int(x))
            except Exception:
                return False
            low = agent.env.env.unwrapped.env
            try:
                low.step(ord('>')); low.step(13); low.step(13)
                agent.step(_agz.A.Command.ESC)
                agent.inventory.update()
            except Exception:
                pass
            if int(agent.blstats.depth) > before:
                lore_patches.COUNTERS["stair_descents"] = \
                    lore_patches.COUNTERS.get("stair_descents", 0) + 1
                try:
                    agent.levels.clear()
                except Exception:
                    pass
                return True
            return False

        def prim_explore():
            """Gehennom maze explorer. AA's explore1 follows corridors well but
            plateaus (leaves reachable frontier + never searches hidden passages,
            which Gehennom mazes have). So: (1) explore1 while it can progress,
            (2) else BFS to the nearest reachable-unexplored cell, (3) else SEARCH
            for hidden corridors. Loop through these until the downstair reveals."""
            # 1) AA's corridor follower (best at normal corridors)
            try:
                e = agent.exploration.explore1(0)
                if e.check_condition():
                    e.run()
                    return
            except Exception:
                pass
            # 2) frontier: go to nearest reachable cell not yet stepped on
            try:
                lvl0 = agent.current_level()
                bf = agent.bfs()
                mask = (bf != -1) & lvl0.walkable & (~lvl0.was_on)
                cand = list(zip(*mask.nonzero()))
                if cand:
                    cand.sort(key=lambda p: bf[p[0], p[1]])
                    for yy, xx in cand[:4]:
                        try:
                            agent.go_to(int(yy), int(xx))
                            return
                        except Exception:
                            continue
            except Exception:
                pass
            # 3) plateau -> DIG horizontally toward the unexplored half to expand
            # past the walled-off reachable pocket (the core Gehennom-reach blocker).
            # Pick the cardinal with the most never-seen cells; rotate on 'too hard'.
            try:
                lvl0 = agent.current_level()
                y, x = int(agent.blstats.y), int(agent.blstats.x)
                was = lvl0.was_on
                cnt = {'h': int((~was[:, :x]).sum()), 'l': int((~was[:, x + 1:]).sum()),
                       'k': int((~was[:y, :]).sum()), 'j': int((~was[y + 1:, :]).sum())}
                tried = agent.__dict__.setdefault("_lore_dig_tried", set())
                order = [d for d, _ in sorted(cnt.items(), key=lambda t: -t[1]) if d not in tried]
                wl = _wand_letter()
                if wl and order:
                    dchar = order[0]
                    b4 = int((agent.bfs() != -1).sum())
                    with agent.atom_operation():
                        agent.step(_agz.A.Command.ZAP); agent.type_text(wl); agent.type_text(dchar)
                    if 'too hard' in str(agent.message):
                        tried.add(dchar)          # undiggable border -> don't retry
                    elif int((agent.bfs() != -1).sum()) <= b4:
                        tried.add(dchar)          # dug but no expansion -> try another
                    else:
                        agent.__dict__["_lore_dig_tried"] = set()   # progress -> reset
                    lore_patches.COUNTERS["dig_expand"] = \
                        lore_patches.COUNTERS.get("dig_expand", 0) + 1
                    return
            except Exception:
                pass
            # last resort: search for hidden passages
            try:
                agent.search(10)
            except Exception:
                pass

        def _eat_if_hungry():
            """Survival reflex: extended Gehennom traversal burns food -> the char
            starves mid-descent. Eat a food item (ration/corpse) when hungry, via
            raw 'e' keypresses (AA has no endgame eat logic in this loop)."""
            try:
                if int(agent.blstats.hunger_state) < 2:   # 0 satiated 1 normal
                    return False
            except Exception:
                return False
            low = agent.env.env.unwrapped.env
            try:
                food_lt = None
                for oc, lt in zip(agent.last_observation['inv_oclasses'],
                                  agent.last_observation['inv_letters']):
                    if int(oc) == _nh2.FOOD_CLASS and int(lt) != 0:
                        food_lt = chr(int(lt)); break
                if food_lt is None:
                    return False
                low.step(ord('e')); low.step(ord(food_lt)); low.step(ord('y'))
                low.step(13); low.step(13)
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
                lore_patches.COUNTERS["descent_eats"] = \
                    lore_patches.COUNTERS.get("descent_eats", 0) + 1
                return True
            except Exception:
                return False

        def prim_fight():
            f = agent.fight2()
            if f.check_condition():
                f.run()
            else:
                prim_explore()

        def prim_flee():
            # move toward the up-stair / away; fall back to a random open step
            try:
                bf = agent.bfs()
                lvl0 = agent.current_level()
                ups = list(zip(*((_u.isin(lvl0.objects, G.STAIR_UP)) & (bf != -1)).nonzero()))
                if ups:
                    agent.go_to(*ups[0]); return
            except Exception:
                pass
            try:
                for ny, nx in agent.neighbors(agent.blstats.y, agent.blstats.x):
                    if agent.current_level().walkable[ny, nx]:
                        agent.move(ny, nx); return
            except Exception:
                pass
            prim_explore()

        def prim_pray():
            try:
                agent.pray()
            except Exception:
                prim_explore()

        def prim_elbereth():
            try:
                agent.engrave_elbereth()
            except Exception:
                prim_explore()

        def _adjacent_threats():
            names = []
            try:
                y, x = agent.blstats.y, agent.blstats.x
                for ny, nx in agent.neighbors(y, x):
                    g = int(agent.glyphs[ny, nx])
                    if _nh2.glyph_is_monster(g):
                        try:
                            names.append(_MON.permonst(g).mname)
                        except Exception:
                            names.append("monster")
            except Exception:
                pass
            return names

        def _build_state():
            lvl = agent.current_level()
            try:
                on_stair = lvl.objects[agent.blstats.y, agent.blstats.x] in G.STAIR_DOWN \
                           or lvl.objects[agent.blstats.y, agent.blstats.x] in G.STAIR_UP
            except Exception:
                on_stair = False
            return {
                "depth": int(agent.blstats.depth),
                "dungeon": int(lvl.dungeon_number),
                "xl": int(agent.blstats.experience_level),
                "hp": int(agent.blstats.hitpoints),
                "max_hp": int(agent.blstats.max_hitpoints),
                "ac": int(agent.blstats.armor_class),
                "hunger": int(agent.blstats.hunger_state),
                "has_dig_wand": _wand_letter() is not None,
                "on_stair": bool(on_stair),
                "have_downstair": int(_u.isin(lvl.objects, G.STAIR_DOWN).sum()) > 0,
                "level_no_dig": lore_patches.COUNTERS.get("level_no_dig") == int(agent.blstats.depth),
                "adjacent_threats": _adjacent_threats(),
            }

        DISPATCH = {
            "DIG_DOWN": prim_dig_down, "DESCEND_STAIRS": prim_descend_stairs,
            "EXPLORE": prim_explore, "FIGHT": prim_fight, "FLEE": prim_flee,
            "PRAY": prim_pray, "ELBERETH": prim_elbereth,
        }
        # ABLATION SWITCH (set LORE_POLICY): which decides the descent action.
        #   llm        - oracle.query_endgame with corpus knowledge (default)
        #   llm_nocorpus - oracle, knowledge="" (isolates whether the wiki helps)
        #   mock       - oracle perfect-knowledge controller (upper bound)
        #   hardcoded  - fixed dig->stairs->explore, NO LLM (the dumb baseline the
        #                LLM must beat; this is what reached DL27-28)
        POLICY = _os2.environ.get("LORE_POLICY", "llm")
        MOCK = POLICY == "mock" or _os2.environ.get("LORE_ORACLE_MOCK", "0") == "1"
        USE_CORPUS = POLICY != "llm_nocorpus"
        lore_patches.COUNTERS["policy"] = POLICY

        def _decide_action():
            if POLICY == "hardcoded":
                # fixed priority, no LLM. returns the action label; dispatch runs it.
                st = _build_state()
                if st["has_dig_wand"] and not st["level_no_dig"]:
                    return "DIG_DOWN"
                if st["have_downstair"]:
                    return "DESCEND_STAIRS"
                return "EXPLORE"
            try:
                state = _build_state()
                knowledge = _oracle.retrieve_knowledge(state) if USE_CORPUS else ""
                decision = _oracle.query_endgame(state, knowledge, mock=MOCK)
                return decision.get("action") or "EXPLORE"
            except AgentFinished:
                raise
            except Exception as _oe:
                lore_patches.COUNTERS["oracle_err"] = repr(_oe)[:90]
                return "EXPLORE"

        @Strategy.wrap
        def descend(s):
            yield True
            while 1:
                lore_patches.COUNTERS["descend_iters"] = \
                    lore_patches.COUNTERS.get("descend_iters", 0) + 1
                # Hard cap: a fed char in a safe Gehennom pocket never dies, so
                # without this the loop runs forever (the frontier-explorer hang).
                # Bail cleanly so results are written and the run can't wedge.
                if lore_patches.COUNTERS["descend_iters"] > int(_os2.environ.get("LORE_MAX_ITERS", 600)):
                    lore_patches.COUNTERS["hit_iter_cap"] = 1
                    raise AgentFinished()
                d = int(agent.blstats.depth)
                if d > lore_patches.COUNTERS.get("max_depth", 0):
                    lore_patches.COUNTERS["max_depth"] = d
                # SURVIVAL-WEIGHTED metric: record the game-turn when each depth is
                # FIRST reached. Downstream scalar = deepest depth still alive >=K
                # turns later (final_turns - first_reach[d] >= K). Suicide-digging
                # to a deep level then dying fast does NOT count that depth;
                # efficient descent that survives DOES. Not gameable by either
                # reckless digging or by loitering shallow.
                try:
                    fr = "firstreach_%d" % d
                    if fr not in lore_patches.COUNTERS:
                        lore_patches.COUNTERS[fr] = int(agent.blstats.time)
                except Exception:
                    pass
                try:
                    lore_patches.COUNTERS["explored_cells"] = int(agent.current_level().was_on.sum())
                    lore_patches.COUNTERS["dungeon_num"] = int(agent.current_level().dungeon_number)
                except Exception:
                    pass
                # DIAGNOSTIC (bug #3): once, report whether a down-stair GLYPH is
                # known on this Gehennom level and whether it is BFS-reachable, so
                # we distinguish "not revealed" from "revealed but behind moat/lava".
                if lore_patches.COUNTERS.get("descend_iters", 0) == 400 and \
                        "down_diag" not in lore_patches.COUNTERS:
                    try:
                        _lvl = agent.current_level()
                        _down = _u.isin(_lvl.objects, G.STAIR_DOWN)
                        _bf = agent.bfs()
                        _st = [(int(yy), int(xx), int(_bf[yy, xx]))
                               for yy, xx in zip(*_down.nonzero())]
                        _bfreach = int((_bf != -1).sum())            # cells reachable now
                        _walk = int(_lvl.walkable.sum())             # known walkable cells
                        # frontier: walkable-but-unexplored reachable cells (explorer
                        # stopped early if this is large)
                        _frontier = int(((_bf != -1) & _lvl.walkable & (~_lvl.was_on)).sum())
                        # ASCII render so we can SEE the level (sealed pocket vs
                        # hidden stair vs glyph-detection issue): @ me, > down, < up,
                        # | wall, } water, . reachable, , explored-elsewhere, space unknown.
                        _rows = []
                        _wall = _u.isin(_lvl.objects, G.WALL)
                        _up = _u.isin(_lvl.objects, G.STAIR_UP)
                        _wa = _u.isin(_lvl.objects, G.WATER) if hasattr(G, "WATER") else _down & False
                        _my = (int(agent.blstats.y), int(agent.blstats.x))
                        for ry in range(_lvl.objects.shape[0]):
                            _r = ""
                            for rx in range(_lvl.objects.shape[1]):
                                if (ry, rx) == _my: _r += "@"
                                elif _down[ry, rx]: _r += ">"
                                elif _up[ry, rx]: _r += "<"
                                elif _wa[ry, rx]: _r += "}"
                                elif _wall[ry, rx]: _r += "|"
                                elif _bf[ry, rx] != -1: _r += "."
                                elif _lvl.was_on[ry, rx]: _r += ","
                                else: _r += " "
                            _rows.append(_r.rstrip())
                        lore_patches.COUNTERS["down_diag"] = {
                            "n_downstairs_known": len(_st), "stairs_yx_reach": _st,
                            "explored": int(_lvl.was_on.sum()), "reachable_now": _bfreach,
                            "known_walkable": _walk, "frontier_unexplored_reachable": _frontier,
                            "my_pos": [int(agent.blstats.y), int(agent.blstats.x)],
                            "ascii": "\n".join(_rows)}
                    except Exception as _de:
                        lore_patches.COUNTERS["down_diag"] = "err %r" % _de
                # survival reflex: eat before starving (extended traversal burns food)
                try:
                    if _eat_if_hungry():
                        continue
                except Exception:
                    pass
                # --- decide the action (policy-dependent), then execute it ---
                try:
                    action = _decide_action()
                except AgentFinished:
                    raise
                lore_patches.COUNTERS["oracle_" + str(action)] = \
                    lore_patches.COUNTERS.get("oracle_" + str(action), 0) + 1
                lore_patches.COUNTERS["last_action"] = str(action)
                # execute the LLM's chosen action via the primitive (action space)
                try:
                    descended = DISPATCH.get(action, prim_explore)()
                    if descended:
                        lore_patches.COUNTERS["descents"] = \
                            lore_patches.COUNTERS.get("descents", 0) + 1
                except AgentFinished:
                    import traceback as _tb
                    lore_patches.COUNTERS["agentfinished_action"] = str(action)
                    lore_patches.COUNTERS["agentfinished_tb"] = _tb.format_exc()[-500:]
                    lore_patches.COUNTERS["agentfinished_iter"] = \
                        lore_patches.COUNTERS.get("descend_iters")
                    raise
                except _gl_exc.AgentChangeStrategy:
                    raise
                except RuntimeError:
                    raise
                except AgentPanic:
                    lore_patches.COUNTERS["act_panic"] = \
                        lore_patches.COUNTERS.get("act_panic", 0) + 1
                except BaseException as _ae:
                    lore_patches.COUNTERS["act_exc"] = repr(_ae)[:90]

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


def patch_ring_parse():
    """AutoAscend's item_manager.parse_text asserts (assert 0, 'on right hand') on
    a WORN ring/amulet/boots -- it has no model for those worn-location suffixes,
    so inventory.update() crashes the whole run whenever the endgame kit's rings
    are on. AA can't reach the endgame anyway, so it never hit these. Normalize the
    worn-location parenthetical to '(being worn)' (which AA parses as equipped)
    BEFORE parsing. Keeps AA source frozen; unblocks the equipped deep descent."""
    from autoascend.item import item_manager as _im
    import re
    _WORN = re.compile(r"\((?:on (?:right|left) (?:hand|foot|paw)|around (?:neck|left claw)|"
                       r"on left hand|embedded in your skin)\)")
    _orig = _im.ItemManager.parse_text.__func__ if hasattr(_im.ItemManager.parse_text, "__func__") \
        else _im.ItemManager.parse_text

    def _patched(text, category=None, glyph=None):
        return _orig(_WORN.sub("(being worn)", text), category, glyph)

    _im.ItemManager.parse_text = staticmethod(_patched)
    return ["ring_parse_fix"]


def patch_enhance_noop():
    """Wizard-mode #enhance view breaks AutoAscend's skill parser at init
    ('bare handed combat' line). Skill-enhancing is secondary for scenario tests
    (we control the kit), so no-op it to let init proceed."""
    from autoascend.character import Character
    Character.parse_enhance_view = lambda self: None
    Character.parse_enhance = lambda self, *a, **k: None if hasattr(Character, "parse_enhance") else None
    return ["enhance_noop"]
