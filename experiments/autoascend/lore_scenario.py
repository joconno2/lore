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


def _do_genocide(agent, classes):
    """Read blessed scrolls of genocide (in inventory) to genocide deadly monster
    CLASSES globally -- a real ascension technique. Read on the safe start level so
    the deep Gehennom levels (else lethal-on-arrival) become survivable. Raw low.step:
    'r' -> scroll letter -> 'What class...?' getlin -> class symbol -> enter."""
    import nle.nethack as _nh
    import autoascend.agent as _A
    low = agent.env.env.unwrapped.env
    done = []
    for cls in classes:
        slt = None
        try:
            for oc, lt in zip(agent.last_observation['inv_oclasses'],
                              agent.last_observation['inv_letters']):
                # wished scrolls are UNIDENTIFIED (display as "scroll labeled XXX"),
                # so detect by CLASS -- the kit's only scrolls are genocide scrolls.
                if int(oc) == _nh.SCROLL_CLASS and int(lt) != 0 and chr(int(lt)) not in done:
                    slt = chr(int(lt)); break
        except Exception:
            break
        if slt is None:
            break
        try:
            low.step(ord('r')); low.step(ord(slt))
            for ch in cls:
                low.step(ord(ch))
            low.step(13); low.step(13); low.step(13)
            done.append(cls)
            agent.step(_A.A.Command.ESC); agent.inventory.update()
        except Exception:
            break
    lore_patches.COUNTERS["genocided"] = done


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


def _is_sick(agent):
    """True if the status line shows a fatal illness (Sick/FoodPois/Ill). Parsed
    off the bottom tty rows -- robust across NLE blstats layouts."""
    try:
        tc = agent.last_observation['tty_chars']
        bottom = "\n".join(bytes(row).decode('latin1') for row in tc[-3:]).lower()
        return any(k in bottom for k in ("foodpois", "sick", "ill "))
    except Exception:
        return False


# --- intrinsic (resistance) tracking + eating: the legit ascension-prep mechanic ---
# Eat corpses on the way DOWN to gain the resistances a real ascension char needs.
# NLE exposes each monster's conveyed resistances (permonst.mconveys) and level live,
# so the reflex reads exactly what a corpse grants and how reliably (grant odds scale
# with monster level -- eat.c givit()). No wizard grants: the char eats what it kills.
_MR_INTR = {0x01: 'fire', 0x02: 'cold', 0x04: 'sleep',
            0x08: 'disint', 0x10: 'shock', 0x20: 'poison'}
_ALL_INTR = set(_MR_INTR.values())
# givit() success messages (src/eat.c) -> the intrinsic just gained. Parsed off the
# message line to KNOW what we already have (so we stop eating for it).
_INTR_MSG = (
    ('momentary chill', 'fire'), ("be chillin'", 'fire'),
    ('full of hot air', 'cold'),
    ('wide awake', 'sleep'),
    ('very firm', 'disint'), ('totally together', 'disint'),
    ('feels amplified', 'shock'), ('grounded in reality', 'shock'),
    ('feel healthy', 'poison'), ('especially healthy', 'poison'),
)


def _raw_msg(o):
    """Message string from a low-level Nethack.step return, which is
    (obs_tuple, done) -- obs is a TUPLE of arrays, message is the (256,) one.
    Best-effort: never raises (a capture error must not abort a keypress seq)."""
    try:
        for a in o[0]:
            if getattr(a, 'shape', None) == (256,):
                return bytes(a).decode('latin1').lower()
    except Exception:
        pass
    return ''


def _intr_mark(intr):
    lst = lore_patches.COUNTERS.setdefault('intr_have', [])
    if intr not in lst:
        lst.append(intr)


def _intr_scan_msg(agent):
    """Record any intrinsic the char just gained (parses givit's grant message)."""
    try:
        m = bytes(agent.last_observation['message']).decode('latin1').lower()
    except Exception:
        return
    for phrase, intr in _INTR_MSG:
        if phrase in m:
            _intr_mark(intr)


def _intr_have():
    return set(lore_patches.COUNTERS.get('intr_have', []))


def _corpse_conveys_missing(nh, MON, glyph, missing):
    """(want_set, mlevel) for a BODY glyph whose monster conveys a still-missing
    resistance, else None. Read live from permonst -- no hardcoded table."""
    try:
        if not nh.glyph_is_body(int(glyph)):
            return None
        pm = MON.permonst(int(glyph))
        mc = int(getattr(pm, 'mconveys', 0))
        want = {nm for bit, nm in _MR_INTR.items() if mc & bit} & missing
        return (want, int(getattr(pm, 'mlevel', 0))) if want else None
    except Exception:
        return None


def _name_conveys_missing(nh, MON, name, missing):
    """Same, for a CARRIED corpse identified by its item-name string."""
    s = name.lower()
    if 'corpse' not in s:
        return None
    s = s.split('corpse')[0]
    for w in ('blessed', 'uncursed', 'cursed', 'partly eaten', 'rotted', 'tainted',
              'very old', 'old', 'slimy'):
        s = s.replace(w, ' ')
    mon = ' '.join(s.split()).strip()
    try:
        mid = MON.id_from_name(mon)
        pm = nh.permonst(mid)
        mc = int(getattr(pm, 'mconveys', 0))
        want = {nm for bit, nm in _MR_INTR.items() if mc & bit} & missing
        return (want, int(getattr(pm, 'mlevel', 0))) if want else None
    except Exception:
        return None


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
    def _conv_missing(it):
        """want-set for a corpse item conveying a still-missing resistance, + mlevel."""
        try:
            if not it.is_corpse():
                return None
            pm = _nh.permonst(int(it.monster_id))
            mc = int(getattr(pm, 'mconveys', 0))
            want = {nm for bit, nm in _MR_INTR.items() if mc & bit} - _intr_have()
            return (want, int(getattr(pm, 'mlevel', 0))) if want else None
        except Exception:
            return None

    # one-time inventory dump: what corpses exist and where (below-me vs carried)
    try:
        _inv = lore_patches.COUNTERS.setdefault("setup_corpse_inv", {})
        _bm = []
        for it in list(agent.inventory.items_below_me):
            if getattr(it, 'is_corpse', lambda: False)():
                _bm.append("%s(mid=%s)" % (str(getattr(it, 'text', '?'))[:24],
                                           getattr(it, 'monster_id', '?')))
        _cr = []
        for it in flatten_items(agent.inventory.items):
            if getattr(it, 'is_corpse', lambda: False)():
                _cr.append("%s(mid=%s)" % (str(getattr(it, 'text', '?'))[:24],
                                           getattr(it, 'monster_id', '?')))
        _inv["below_me"] = _bm[:30]
        _inv["carried"] = _cr[:30]
    except Exception as _ie:
        lore_patches.COUNTERS["setup_corpse_inv"] = "err %r" % _ie

    # CHOKE-SAFE loop: eat wished resistance corpses (below-me pile from the wish, then
    # carried) via AA's own eat -- which FINISHES the corpse so cpostfx grants the
    # intrinsic and routes through the hooked env.step so the grant is captured. Stop
    # at Satiated (eating then chokes to death); the descent reflex eats the rest from
    # fresh kills on the way down.
    eaten = 0
    for _ in range(30):
        try:
            if int(agent.blstats.hunger_state) == 0:      # Satiated -> choke, stop
                break
        except Exception:
            pass
        target = None
        try:
            for it in list(agent.inventory.items_below_me):
                if _conv_missing(it) is not None:
                    target = it
                    break
        except Exception:
            pass
        if target is None:                                 # else a carried corpse, highest level
            try:
                best = None
                for it in flatten_items(agent.inventory.items):
                    r = _conv_missing(it)
                    if r is None:
                        continue
                    if best is None or r[1] > best[1]:
                        best = (it, r[1])
                if best is not None:
                    target = best[0]
            except Exception:
                pass
        if target is None:
            break
        t0 = int(agent.blstats.time)
        _dbg = lore_patches.COUNTERS.setdefault("setup_eat_dbg", [])
        try:
            _tnm = str(getattr(target, 'text', '?'))[:30]
            _tmid = int(getattr(target, 'monster_id', -1))
            _th0 = int(agent.blstats.hunger_state)
        except Exception:
            _tnm, _tmid, _th0 = '?', -1, -1
        try:
            agent.inventory.eat(target)
            agent.inventory.update()
        except Exception as _ee:
            if len(_dbg) < 20:
                _dbg.append("EATERR %s: %r" % (_tnm, _ee))
            break
        _intr_scan_msg(agent)
        try:
            _pm = bytes(agent.last_observation['message']).decode('latin1').strip()[:40]
            _ph1 = int(agent.blstats.hunger_state)
            if len(_dbg) < 20:
                _dbg.append("ate %s mid=%d dt=%d hun%d->%d msg=%s"
                            % (_tnm, _tmid, int(agent.blstats.time) - t0, _th0, _ph1, _pm))
        except Exception:
            pass
        if int(agent.blstats.time) <= t0:                  # nothing consumed -> avoid spin
            break
        eaten += 1
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
    # 1) armor (gray DSM, shield of reflection, cloak, helm, boots, gloves) via
    #    AutoAscend's own routine
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
    # 1b) WIELD the silver saber (wear_best_stuff is armor only; AA won't auto-wield
    #     the wished silver weapon that demons/vampires/undead are vulnerable to).
    try:
        agent.wield_best_melee_weapon()
        agent.step(_A.A.Command.ESC)
        agent.inventory.update()
        lore_patches.COUNTERS["wielded"] = 1
    except Exception as _we:
        lore_patches.COUNTERS["wield_err"] = repr(_we)[:60]
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


def _put_on_blindfold(agent):
    """Put on the towel so the char is Blind -> the worn helm of telepathy then senses
    ALL minded monsters through walls (ESP). The skilled-player way to detect mind
    flayers / giant eels before contact. Navigation must then run on mapped/remembered
    terrain (blind can't see NEW terrain), so pair with magic mapping / prior reveal."""
    import autoascend.agent as _A
    low = agent.env.env.unwrapped.env
    try:
        lt = None
        for nm, ltr in zip(agent.last_observation['inv_strs'],
                           agent.last_observation['inv_letters']):
            if int(ltr) == 0:
                continue
            s = bytes(nm).decode('latin1').lower()
            if 'towel' in s or 'blindfold' in s:
                lt = chr(int(ltr)); break
        if lt is None:
            lore_patches.COUNTERS['blindfold_err'] = 'no towel in inv'
            return
        low.step(ord('P')); low.step(ord(lt)); low.step(13); low.step(13)
        agent.step(_A.A.Command.ESC); agent.inventory.update()
        lore_patches.COUNTERS['blindfolded'] = 1
    except Exception as _e:
        lore_patches.COUNTERS['blindfold_err'] = repr(_e)[:50]


def patch_perstep_survival():
    """PER-STEP survival interrupt (the architectural fix the ladder doc flagged).
    AA's multi-turn primitives (explore1/go_to/fight2) call Agent.step per game-turn;
    the descend loop only re-checks survival BETWEEN primitives, so a swarm that chips
    the tank down DURING a primitive kills it before any reflex fires (Gehennom ~200-
    turn deaths, panic_reads=0). Wrap Agent.step to RAISE AgentPanic the moment HP goes
    critical or a swarm forms -> unwinds the primitive back to the descend loop, whose
    next iteration fires heal/panic-escape. Active only after setup teleport (gated on
    agent._lore_descent_active) so setup/equip/eat aren't interrupted."""
    from autoascend.agent import Agent
    from autoascend.exceptions import AgentPanic
    if getattr(Agent, "_lore_perstep_patched", False):
        return
    _orig = Agent.step

    def _step(self, action, additional_action_iterator=None):
        _orig(self, action, additional_action_iterator)
        if not getattr(self, "_lore_descent_active", False):
            return
        # debounce: don't re-panic every single step (the escape/heal reflex needs a
        # few steps to resolve). Interrupt at most once per 3 steps.
        last = getattr(self, "_lore_last_panic", -99)
        if self.step_count - last < 3:
            return
        try:
            hp = int(self.blstats.hitpoints)
            mhp = max(1, int(self.blstats.max_hitpoints))
        except Exception:
            return
        # true emergencies only: aborting AA's primitive is costly, so interrupt only
        # when death is imminent (HP<30%) or a real swarm (>=4 within 3), not routine.
        crit = hp < 0.30 * mhp
        swarm = False
        if not crit:
            try:
                mons = self.get_visible_monsters()
                swarm = sum(1 for m in mons if int(m[0]) <= 3) >= 4
            except Exception:
                swarm = False
        if crit or swarm:
            self._lore_last_panic = self.step_count
            lore_patches.COUNTERS["perstep_panics"] = \
                lore_patches.COUNTERS.get("perstep_panics", 0) + 1
            raise AgentPanic("lore-survival: hp=%d/%d swarm=%s" % (hp, mhp, swarm))

    Agent.step = _step
    Agent._lore_perstep_patched = True


def patch_water_walkable():
    """AA's update_level (agent.py:596) has NO water glyph set and excludes water
    from `walkable`, so a downstair across a MOAT is unreachable and the char is
    stranded (the real descent 'pocket'). With water-walking boots the char crosses
    moats safely -> monkeypatch update_level to ALSO mark S_pool/S_water cells
    walkable after AA's own update. Persistent (AA re-runs update_level every step,
    so a transient per-iteration marking gets overwritten -- this doesn't). Lava is
    NOT marked (needs fire res + levitation). AA stays frozen; this is a scenario
    patch enabling the wished water-walking kit to actually be used by the pathing."""
    from autoascend.agent import Agent
    import nle.nethack as _nh
    if getattr(Agent, "_lore_water_patched", False):
        return
    POOL = _nh.GLYPH_CMAP_OFF + 32
    WATER = _nh.GLYPH_CMAP_OFF + 41
    _orig = Agent.update_level

    def _patched(self):
        _orig(self)
        try:
            wm = (self.glyphs == POOL) | (self.glyphs == WATER)
            if wm.any():
                self.current_level().walkable[wm] = True
        except Exception:
            pass
    Agent.update_level = _patched
    Agent._lore_water_patched = True
    return ["water_walkable"]


def _do_teleport(agent, target_depth):
    # ^V isn't in AutoAscend's action space, so issue the wizard level-teleport
    # at the LOW-LEVEL nethack (raw keypresses), then resync the agent with one
    # normal step. low-level env: gym_env.unwrapped.env (NLE issues keypresses
    # via self.env.step(keypress)).
    genv = agent.env.env
    low = genv.unwrapped.env
    import autoascend.agent as _ag
    # ITERATIVE teleport: a single ^V clamps at the deepest GENERATED level (~DL29
    # from a fresh game), but stepping to deepest+1 repeatedly FORCE-GENERATES each
    # next level -- reaching deep Gehennom (DL49, the invocation-level range) with
    # zero maze navigation. So step down one level at a time toward the target.
    def _tp(d):
        low.step(27); low.step(27)     # ESC x2: clear any residual prompt
        low.step(22)                   # ^V
        for ch in str(int(d)):
            low.step(ord(ch))          # digits
        low.step(13)                   # enter
        low.step(13); low.step(13)     # clear any --More--
    try:
        cur = int(agent.blstats.depth)
    except Exception:
        cur = 1
    if int(target_depth) <= cur + 1:
        _tp(target_depth)              # shallow / single step
    else:
        for d in range(cur + 1, int(target_depth) + 1):
            _tp(d)                     # step down, force-generating each level
            try:
                if int(agent.blstats.depth) >= int(target_depth):
                    break
            except Exception:
                pass
    agent.step(_ag.A.Command.ESC)      # resync agent state with new level
    try:
        agent.levels.clear()           # wipe stale DL1 stair graph (-> PLANE)
    except Exception:
        pass
    try:
        lore_patches.COUNTERS["scenario_depth"] = int(agent.blstats.depth)
    except Exception:
        pass


def _teleport_to_valley(agent):
    """Place at the VALLEY OF THE DEAD (Gehennom L1) for a PROPER descent. ^V-by-depth
    stays in the current branch, and the DoD length varies (25-29), so a FIXED target
    lands in the main dungeon on most seeds (dnum=0). Descend one level at a time and
    STOP at the first Gehennom level (dungeon_number==1 == the Valley). Verified:
    Valley target is 26/28/29 on seeds 45/42/43 -- adaptive, not fixed."""
    import autoascend.agent as _ag
    low = agent.env.env.unwrapped.env
    def _tp(d):
        low.step(27); low.step(27); low.step(22)
        for ch in str(int(d)): low.step(ord(ch))
        low.step(13); low.step(13); low.step(13)
    last = None
    for d in range(24, 40):
        _tp(d)
        try:
            agent.step(_ag.A.Command.ESC)
        except Exception:
            pass
        try:
            dn = int(getattr(agent.blstats, "dungeon_number", 0))
            last = int(agent.blstats.depth)
            if dn == 1:                       # first Gehennom level == the Valley
                try: agent.levels.clear()
                except Exception: pass
                lore_patches.COUNTERS["valley_depth"] = last
                lore_patches.COUNTERS["valley_lnum"] = int(getattr(agent.blstats, "level_number", -1))
                return last
        except Exception:
            pass
    lore_patches.COUNTERS["valley_fail"] = last
    return last


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
                        if getattr(it, "category", None) != _nh.POTION_CLASS:
                            continue
                        # SKIP healing potions -> keep them for the heal reflex.
                        # Identify via raw inv_strs name (the AA item object's str
                        # is NOT the name). Default to quaffing if lookup fails.
                        try:
                            _lt = agent.inventory.items.get_letter(it)
                            _nm = ""
                            for nm, ltr in zip(agent.last_observation['inv_strs'],
                                               agent.last_observation['inv_letters']):
                                if int(ltr) != 0 and chr(int(ltr)) == _lt:
                                    _nm = bytes(nm).decode('latin1').lower(); break
                            if 'healing' in _nm:
                                continue
                        except Exception:
                            pass
                        pot = it
                        break
                    if pot is None:
                        break
                    try:
                        agent.inventory.quaff(pot)
                    except Exception:
                        break
                lore_patches.COUNTERS["xl_before_tp"] = int(agent.blstats.experience_level)
                try:
                    _hk = 0
                    for nm, oc, ltr in zip(agent.last_observation['inv_strs'],
                                           agent.last_observation['inv_oclasses'],
                                           agent.last_observation['inv_letters']):
                        if int(oc) == _nh.POTION_CLASS and int(ltr) != 0 and \
                                'healing' in bytes(nm).decode('latin1').lower():
                            _hk += 1
                    lore_patches.COUNTERS["healing_kept"] = _hk
                except Exception:
                    pass
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
    patch_water_walkable()  # cross moats (water-walking boots) -> reach walled-off downstairs

    def descent_global(self):
        agent = self.agent
        if not getattr(agent, "_lore_tp_done", False):
            agent.__dict__["_lore_tp_done"] = True
            try:
                import autoascend.agent as _ag3
                import nle.nethack as _nh
                # Wished items come in UNIDENTIFIED (a "potion of full healing" shows
                # as e.g. "black potions"), so a name-based 'healing' skip does NOT
                # work -- the old code drank the healing potions during the level-up
                # loop, leaving none for Gehennom (heal reflex never fired). FIX:
                # split the kit. Quaff gain-level potions FIRST (before any healing
                # potion exists), THEN wish the rest and capture the healing stack's
                # LETTER (stable per stack) for the reflex to use.
                gain = [w for w in wishes if 'gain level' in w.lower()]
                rest = [w for w in wishes if 'gain level' not in w.lower()]
                for it in gain:
                    _do_wish(agent, it)
                try:
                    agent.step(_ag3.A.Command.ESC); agent.inventory.update()
                except Exception: pass
                for _ in range(80):        # drain every gain-level potion -> tank XL
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
                lore_patches.COUNTERS["xl_before_tp"] = int(agent.blstats.experience_level)
                for it in rest:            # now the survival kit incl. healing potions
                    _do_wish(agent, it)
                lore_patches.COUNTERS["wishes"] = len(wishes)
                try: lore_patches.COUNTERS["t_after_wishes"] = int(agent.blstats.time)
                except Exception: pass
                try:
                    agent.step(_ag3.A.Command.ESC); agent.inventory.update()
                except Exception:
                    pass
                # capture the healing potion (the only potion stack left after the
                # gain-level quaff) by LETTER + appearance -- names are unidentified.
                try:
                    _hk = 0; _hlt = None; _happ = None
                    for nm, oc, ltr in zip(agent.last_observation['inv_strs'],
                                           agent.last_observation['inv_oclasses'],
                                           agent.last_observation['inv_letters']):
                        if int(oc) == _nh.POTION_CLASS and int(ltr) != 0:
                            s = bytes(nm).decode('latin1').strip('\x00').strip()
                            _hlt = chr(int(ltr)); _happ = s.lower()
                            try: _hk += int(s.split()[0])
                            except Exception: _hk += 1
                    agent.__dict__["_lore_heal_lt"] = _hlt
                    agent.__dict__["_lore_heal_app"] = _happ
                    lore_patches.COUNTERS["healing_kept"] = _hk
                    lore_patches.COUNTERS["heal_letter"] = _hlt
                    lore_patches.COUNTERS["heal_appearance"] = _happ
                except Exception as _he:
                    lore_patches.COUNTERS["heal_capture_err"] = repr(_he)[:60]
                try: lore_patches.COUNTERS["t_after_quaff"] = int(agent.blstats.time)
                except Exception: pass
                import os as _os
                # GENOCIDE deadly Gehennom classes on the safe start level (global
                # effect) so the deep levels are survivable for the endgame demo.
                if _os.environ.get("LORE_GENOCIDE"):
                    _do_genocide(agent, list(_os.environ.get("LORE_GENOCIDE")))
                # EQUIP FIRST (armor + jewelry + weapon), THEN eat. Eating a
                # poisonous corpse can choke/hurt the char and abort setup on some
                # seeds -> equip skipped -> lands at AC ~3 and dies fast (2-3/8 seeds).
                # Equipping first guarantees the char always lands fully armored
                # (AC-20+); poison-res eating is a bonus that can't cost the armor.
                if _os.environ.get("LORE_NO_EQUIP") != "1":
                    _equip_endgame(agent)       # wear armor + put on amulet/rings
                try: lore_patches.COUNTERS["t_after_equip"] = int(agent.blstats.time)
                except Exception: pass
                # BLINDFOLD NAV (LORE_BLINDFOLD=1): blind the char so the worn helm of
                # telepathy senses monsters through walls (mind flayers / eels). Tests
                # whether blind navigation on revealed terrain still works.
                if _os.environ.get("LORE_BLINDFOLD") == "1":
                    _put_on_blindfold(agent)
                if _os.environ.get("LORE_NO_EAT") != "1":
                    _eat_for_intrinsics(agent)  # poison res from wished corpses
                    # Eating a poisonous corpse while not yet poison-resistant can
                    # self-inflict illness; cure it NOW with prayer (fresh game, safe).
                    if _is_sick(agent):
                        try:
                            agent.pray()
                            lore_patches.COUNTERS["setup_prayed_sick"] = 1
                        except Exception as _pe:
                            lore_patches.COUNTERS["setup_pray_err"] = repr(_pe)[:50]
                    lore_patches.COUNTERS["sick_after_eat"] = int(_is_sick(agent))
                try: lore_patches.COUNTERS["t_after_eat"] = int(agent.blstats.time)
                except Exception: pass
                # LORE_VALLEY=1: place at the Valley of the Dead (Gehennom L1) for a
                # PROPER descent, instead of a fixed ^V depth that lands in the main
                # dungeon (dnum=0) on most seeds. Adaptive: descend to first dnum=1.
                if _os.environ.get("LORE_VALLEY") == "1":
                    _teleport_to_valley(agent)
                else:
                    _do_teleport(agent, target_depth)
                lore_patches._bump("scenario_teleport")
                # per-step survival interrupt is now safe to arm (setup done)
                agent.__dict__["_lore_descent_active"] = True
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

        def _reveal_level():
            """Wizard ^F (wiz_map) reveals the whole level so the downstair GLYPH is
            known immediately -> beeline/dig straight to it instead of a slow,
            oscillating frontier crawl (which explored 46 cells in 418 iters and hung
            one seed to the docker timeout). This is option-(b) capability tooling: a
            SCENARIO demo of the endgame sequence (wizard-placed + wished kit); the
            LLM contribution is the invocation ritual, not maze-solving. Fires once
            per level (keyed on depth). wiz_map takes no game turn."""
            d = int(agent.blstats.depth)
            if agent.__dict__.get("_lore_revealed_depth") == d:
                return
            agent.__dict__["_lore_revealed_depth"] = d
            low = agent.env.env.unwrapped.env
            try:
                low.step(27)                      # ESC any pending prompt FIRST
                low.step(6)                       # ^F = wiz_map (reveals terrain)
                low.step(27)                      # ESC to dismiss the map view
                agent.step(_agz.A.Command.ESC)    # resync AA to normal play view
                agent.inventory.update()
                lore_patches.COUNTERS["reveals"] = \
                    lore_patches.COUNTERS.get("reveals", 0) + 1
                # capture the revealed downstair(s) NOW (lvl.objects has them right
                # after ^F) into a persistent per-depth set. AA's later update_level
                # drops revealed-but-unseen stairs -> have_downstair goes False and the
                # policy never routes to the stair-nav (the walled-pocket blocker).
                try:
                    _rd = list(zip(*_u.isin(agent.current_level().objects, G.STAIR_DOWN).nonzero()))
                    if _rd:
                        agent.__dict__["_lore_downs_%d" % d] = [(int(yy), int(xx)) for yy, xx in _rd]
                except Exception:
                    pass
            except Exception as _re:
                lore_patches.COUNTERS["reveal_err"] = repr(_re)[:60]

        def prim_descend_stairs():
            """Go to the known down-stair GLYPH and take it via raw '>'. AA's
            explore_stairs relies on its Gehennom-blind dungeon model, so navigate
            directly off the glyph grid. If the downstair is KNOWN (revealed) but
            walled off (not bfs-reachable -- the ~2/7 WALLED case), dig one cardinal
            toward it and step in; the tunnel connects it over a few iterations."""
            try:
                lvl0 = agent.current_level()
                bf = agent.bfs()
                all_downs = list(zip(*_u.isin(lvl0.objects, G.STAIR_DOWN).nonzero()))
            except Exception:
                return False
            if not all_downs:
                # FALLBACK: AA dropped the ^F-revealed stair -> use the persistent
                # per-depth capture from _reveal_level so the walled-pocket nav runs.
                all_downs = agent.__dict__.get("_lore_downs_%d" % int(agent.blstats.depth), [])
            if not all_downs:
                return False
            reach = [(y, x) for y, x in all_downs if bf[y, x] != -1]
            if not reach and _os2.environ.get("LORE_REVEAL") == "1":
                # WALLED (wall-pocket, non-moat): L-DIG to the known downstair.
                # Single-axis digging at an offset stair never connects. Instead:
                # PHASE 1 align to the stair's ROW (dig vertical toward ty, step into
                # the tunnel); PHASE 2 once on row ty, dig HORIZONTAL toward tx and
                # walk the tunnel. Re-runs each iteration so it tunnels an L-path.
                # WALLED: 8-DIR DIG-AWARE Dijkstra to the known downstair. WALKABLE
                # neighbors (8-dir incl. diagonal -- preserves the DL37 peak) cost 1;
                # DIGGABLE WALL neighbors (cardinal only -- can't dig diagonally) cost
                # 6; stone/boundary impassable. Walk open, DIG walls along the route,
                # ONE move/dig per call. (An earlier cardinal-ONLY version regressed
                # DL37->DL30; diagonals matter.)
                import numpy as _npb2
                import heapq as _hq2
                ty, tx = min(all_downs, key=lambda p: abs(p[0] - agent.blstats.y) + abs(p[1] - agent.blstats.x))
                ty, tx = int(ty), int(tx)
                y0w, x0w = int(agent.blstats.y), int(agent.blstats.x)
                try:
                    walkw = lvl0.walkable
                    Hw, Ww = walkw.shape
                    INF = 1 << 30
                    cst = _npb2.full((Hw, Ww), INF, dtype=_npb2.int64)
                    cst[ty, tx] = 0
                    pqw = [(0, ty, tx)]
                    DIAG = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

                    def _diggable(yy, xx):
                        # a wand of digging bores through WALLS and solid ROCK (^F
                        # doesn't reveal rock, so the barrier is 'unknown'), but NOT
                        # the level boundary ring (undiggable stone). Cardinal only.
                        return 1 <= yy < Hw - 1 and 1 <= xx < Ww - 1
                    while pqw:
                        c, cy, cx = _hq2.heappop(pqw)
                        if c > cst[cy, cx]:
                            continue
                        for dy, dx in DIAG:
                            ny, nx = cy + dy, cx + dx
                            if not (0 <= ny < Hw and 0 <= nx < Ww):
                                continue
                            if walkw[ny, nx]:
                                nc = c + 1
                            elif (dy == 0 or dx == 0) and _diggable(ny, nx):   # dig cardinal
                                nc = c + 6
                            else:
                                continue
                            if nc < cst[ny, nx]:
                                cst[ny, nx] = nc
                                _hq2.heappush(pqw, (nc, ny, nx))
                    if cst[y0w, x0w] < INF:
                        bestw = None
                        for dy, dx in DIAG:
                            ny, nx = y0w + dy, x0w + dx
                            if not (0 <= ny < Hw and 0 <= nx < Ww):
                                continue
                            steppable = walkw[ny, nx] or ((dy == 0 or dx == 0) and _diggable(ny, nx))
                            if steppable and cst[ny, nx] < cst[y0w, x0w]:
                                if bestw is None or cst[ny, nx] < cst[bestw[0], bestw[1]]:
                                    bestw = (ny, nx)
                        if bestw is not None:
                            by, bx = bestw
                            if walkw[by, bx]:
                                agent.move(int(by), int(bx))
                                lore_patches.COUNTERS["stair_steps"] = \
                                    lore_patches.COUNTERS.get("stair_steps", 0) + 1
                            else:
                                wl = _wand_letter()
                                if wl:
                                    dchar = {(1, 0): 'j', (-1, 0): 'k', (0, 1): 'l', (0, -1): 'h'}[(by - y0w, bx - x0w)]
                                    with agent.atom_operation():
                                        agent.step(_agz.A.Command.ZAP); agent.type_text(wl); agent.type_text(dchar)
                                    lore_patches.COUNTERS["walled_digs"] = \
                                        lore_patches.COUNTERS.get("walled_digs", 0) + 1
                                    if 'too hard' not in str(agent.message) and \
                                            agent.current_level().walkable[by, bx]:
                                        agent.move(int(by), int(bx))
                except AgentFinished:
                    raise
                except Exception as _we:
                    lore_patches.COUNTERS["walled_err"] = repr(_we)[:60]
                return False
            before = int(agent.blstats.depth)
            y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
            ty, tx = int(reach[0][0]), int(reach[0][1])
            if (y0, x0) == (ty, tx):
                # ON the stair -> take it
                low = agent.env.env.unwrapped.env
                try:
                    low.step(ord('>')); low.step(13); low.step(13)
                    agent.step(_agz.A.Command.ESC); agent.inventory.update()
                except Exception:
                    return False
                if int(agent.blstats.depth) > before:
                    lore_patches.COUNTERS["stair_descents"] = \
                        lore_patches.COUNTERS.get("stair_descents", 0) + 1
                    try:
                        agent.levels.clear()
                    except Exception:
                        pass
                    return True
                return False
            # ONE robust step toward the stair. AA's go_to(full path) throws on
            # revealed-but-unvisited terrain (ValueError "'in' is not in list",
            # AgentPanic 'position do not match') and CRASHES the game -- the true
            # stair-take blocker (seed 47 descended fine, 44/45/48 crashed here).
            # Compute the path myself and move ONE cell, tolerating errors; the
            # descend loop re-runs survival reflexes between steps.
            # PURE-NUMPY BFS to the stair + single agent.move step. AA's path()/
            # go_to() throw on revealed maze terrain (the #1 recurring descent
            # blocker); this computes the route myself over lvl.walkable (incl. the
            # water patch) and takes ONE adjacent step, never touching AA's crashing
            # pathfinder. Re-runs each iter -> walks the whole route one step at a time.
            try:
                import numpy as _npb
                from collections import deque as _deque
                walk = agent.current_level().walkable
                H, W = walk.shape
                dist = _npb.full((H, W), -1, dtype=_npb.int32)
                dq = _deque([(ty, tx)]); dist[ty, tx] = 0
                _nb = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
                while dq:
                    cy, cx = dq.popleft()
                    for dy, dx in _nb:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] == -1 \
                                and (walk[ny, nx] or (ny, nx) == (y0, x0)):
                            dist[ny, nx] = dist[cy, cx] + 1; dq.append((ny, nx))
                if dist[y0, x0] != -1:
                    best = None
                    for dy, dx in _nb:
                        ny, nx = y0 + dy, x0 + dx
                        if 0 <= ny < H and 0 <= nx < W and walk[ny, nx] and dist[ny, nx] != -1:
                            if best is None or dist[ny, nx] < dist[best[0], best[1]]:
                                best = (ny, nx)
                    if best is not None:
                        agent.move(int(best[0]), int(best[1]))
                        lore_patches.COUNTERS["stair_steps"] = \
                            lore_patches.COUNTERS.get("stair_steps", 0) + 1
            except AgentFinished:
                raise
            except Exception as _pe:
                lore_patches.COUNTERS["stairnav_err"] = repr(_pe)[:60]
            return False

        def prim_explore():
            """SINGLE-STEP Gehennom maze explorer. Takes ONE step toward the nearest
            reachable-unexplored cell, so the descend loop re-runs its survival
            reflexes (fight/heal/pray) every game turn instead of after a whole
            multi-turn traversal (during which the tank took unanswered hits -- the
            chip-death mode). Falls through to horizontal dig (breach a walled-off
            region) then a single search (hidden passages) when no frontier remains."""
            # 1) frontier: ONE step toward the nearest reachable un-stepped cell
            try:
                lvl0 = agent.current_level()
                bf = agent.bfs()
                mask = (bf != -1) & lvl0.walkable & (~lvl0.was_on)
                cand = list(zip(*mask.nonzero()))
                if cand:
                    cand.sort(key=lambda p: bf[p[0], p[1]])
                    for yy, xx in cand[:4]:
                        try:
                            agent.go_to(int(yy), int(xx), max_steps=1)  # single step
                            return
                        except Exception:
                            continue
            except Exception:
                pass
            # 2) LLM NAV JUDGMENT (LORE_LLM_NAV): no frontier -> walled into a pocket.
            # The symbolic dig below picks 'most unseen cells', NOT toward the
            # downstair, and fails. Hand the LLM the revealed ASCII and dig the
            # cardinal it points toward the (walled-off) downstair -- the novel LLM
            # angle: spatial pocket-escape judgment the symbolic explorer can't make.
            if _os2.environ.get("LORE_LLM_NAV") == "1":
                try:
                    lvlA = agent.current_level(); bfA = agent.bfs()
                    myA = (int(agent.blstats.y), int(agent.blstats.x))
                    downA = _u.isin(lvlA.objects, G.STAIR_DOWN)
                    upA = _u.isin(lvlA.objects, G.STAIR_UP)
                    wallA = _u.isin(lvlA.objects, G.WALL)
                    rowsA = []
                    for ry in range(lvlA.objects.shape[0]):
                        rr = ""
                        for rx in range(lvlA.objects.shape[1]):
                            if (ry, rx) == myA: rr += "@"
                            elif downA[ry, rx]: rr += ">"
                            elif upA[ry, rx]: rr += "<"
                            elif wallA[ry, rx]: rr += "|"
                            elif bfA[ry, rx] != -1: rr += "."
                            elif lvlA.walkable[ry, rx]: rr += ":"
                            else: rr += " "
                        rowsA.append(rr.rstrip())
                    act = _oracle.query_nav_dig("\n".join(rowsA))
                    lore_patches.COUNTERS["llm_nav_q"] = lore_patches.COUNTERS.get("llm_nav_q", 0) + 1
                    lore_patches.COUNTERS["llm_nav_" + str(act)] = \
                        lore_patches.COUNTERS.get("llm_nav_" + str(act), 0) + 1
                    dmap = {"DIG_NORTH": ("k", -1, 0), "DIG_SOUTH": ("j", 1, 0),
                            "DIG_EAST": ("l", 0, 1), "DIG_WEST": ("h", 0, -1)}
                    wl = _wand_letter()
                    if act in dmap and wl:
                        dc, dy, dx = dmap[act]
                        y0, x0 = int(agent.blstats.y), int(agent.blstats.x)
                        with agent.atom_operation():
                            agent.step(_agz.A.Command.ZAP); agent.type_text(wl); agent.type_text(dc)
                        lore_patches.COUNTERS["llm_digs"] = \
                            lore_patches.COUNTERS.get("llm_digs", 0) + 1
                        if 'too hard' not in str(agent.message):
                            ny, nx = y0 + dy, x0 + dx
                            try:
                                if lvlA.objects.shape[0] > ny >= 0 <= nx < lvlA.objects.shape[1] \
                                        and agent.current_level().walkable[ny, nx]:
                                    agent.move(ny, nx)
                            except Exception:
                                pass
                        return
                except AgentFinished:
                    raise
                except Exception as _le:
                    lore_patches.COUNTERS["llm_nav_err"] = repr(_le)[:60]
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
            # last resort: search for hidden passages -- ONE turn only, so the loop
            # returns to the survival reflexes (heal/fight) between searches instead
            # of taking 10 free hits mid-search (the tank-chip death mode).
            try:
                agent.search(1)
            except Exception:
                pass

        def _heal_if_low():
            """Survival reflex: extended Gehennom traversal accumulates damage
            faster than regen; prayer has a ~1000-turn cooldown so it can't sustain.
            Quaff a potion of (full) healing when HP drops, to keep exploring to the
            downstair. Raw 'q' keypresses (unidentified/blessed healing in kit)."""
            try:
                if agent.blstats.hitpoints > 0.45 * max(1, agent.blstats.max_hitpoints):
                    return False
            except Exception:
                return False
            low = agent.env.env.unwrapped.env
            try:
                heal_lt = None
                for nm, oc, lt in zip(agent.last_observation['inv_strs'],
                                      agent.last_observation['inv_oclasses'],
                                      agent.last_observation['inv_letters']):
                    s = bytes(nm).decode('latin1').strip('\x00').strip().lower()
                    if int(oc) == _nh2.POTION_CLASS and int(lt) != 0 and 'healing' in s:
                        heal_lt = chr(int(lt)); break
                if heal_lt is None:
                    return False
                low.step(ord('q')); low.step(ord(heal_lt)); low.step(13); low.step(13)
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
                lore_patches.COUNTERS["descent_heals"] = \
                    lore_patches.COUNTERS.get("descent_heals", 0) + 1
                return True
            except Exception:
                return False

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

        def _item_conveys_missing(it, missing):
            """(want_set, mlevel) if corpse item's monster conveys a missing intrinsic.
            monster_id IS the mnum -> read conveys/level live from permonst."""
            try:
                if not it.is_corpse():
                    return None
                pm = _nh2.permonst(int(it.monster_id))
                mc = int(getattr(pm, 'mconveys', 0))
                want = {nm for bit, nm in _MR_INTR.items() if mc & bit} & missing
                return (want, int(getattr(pm, 'mlevel', 0))) if want else None
            except Exception:
                return None

        def _do_eat(it):
            """Eat a corpse via AA's own eat (handles floor/inventory prompts + the
            multi-turn occupation, so the corpse FINISHES and cpostfx grants the
            intrinsic). Returns True if a game turn passed."""
            t0 = int(agent.blstats.time)
            try:
                agent.inventory.eat(it)
            except AgentFinished:
                raise
            except Exception:
                pass
            try:
                agent.inventory.update()
            except Exception:
                pass
            _intr_scan_msg(agent)
            return int(agent.blstats.time) > t0

        def _eat_intrinsic_reflex():
            """Eat a fresh kill (corpse below me) or a carried resistance corpse for a
            resistance the char still LACKS -- the legit 'eat intrinsics on the way
            down' mechanic (no wizard grants). Uses AA's eat so the corpse finishes."""
            lore_patches.COUNTERS["reflex_calls"] = \
                lore_patches.COUNTERS.get("reflex_calls", 0) + 1
            _intr_scan_msg(agent)
            missing = _ALL_INTR - _intr_have()
            lore_patches.COUNTERS["reflex_missing_n"] = len(missing)
            if not missing:
                return False
            try:
                if int(agent.blstats.hunger_state) == 0:   # Satiated -> choke risk
                    return False
            except Exception:
                pass
            if _adjacent_threats():                        # don't eat under attack
                return False
            # 1) a corpse UNDER me (fresh kill or wished-onto-floor) that grants a
            #    missing resistance -> eat it now.
            try:
                for it in list(agent.inventory.items_below_me):
                    if _item_conveys_missing(it, missing) is not None:
                        if _do_eat(it):
                            lore_patches.COUNTERS["intr_eats"] = \
                                lore_patches.COUNTERS.get("intr_eats", 0) + 1
                            return True
            except AgentFinished:
                raise
            except Exception:
                pass
            # 2) backstop: when hungry, eat the CARRIED resistance corpse (wished
            #    dragons) with the highest monster level (most reliable grant).
            try:
                if int(agent.blstats.hunger_state) >= 2:
                    from autoascend.agent import flatten_items
                    best = None
                    for it in flatten_items(agent.inventory.items):
                        r = _item_conveys_missing(it, missing)
                        if r is None:
                            continue
                        if best is None or r[1] > best[1][1]:
                            best = (it, r)
                    if best is not None and _do_eat(best[0]):
                        lore_patches.COUNTERS["intr_eats_carried"] = \
                            lore_patches.COUNTERS.get("intr_eats_carried", 0) + 1
                        return True
            except AgentFinished:
                raise
            except Exception:
                pass
            return False

        _INSTADEATH_ESP = ('mind flayer', 'giant eel', 'kraken', 'lich', 'nalfeshnee',
                           'marilith', 'pit fiend', 'orcus', 'demogorgon', 'nalfeshnee',
                           'vampire lord', 'vlad', 'wizard of yendor', 'green slime')

        def _telepathy_scan():
            """LORE_BLINDFOLD blindfold-navigation: put on the towel (blind) so the
            worn helm of telepathy gives ESP -- sensing EVERY minded monster on the
            level (through walls), which line-of-sight misses. List them via the game's
            own monster list, record instadeath threats (mind flayers / eels / covetous
            liches), then take the towel OFF so vision + scroll-reading work again.
            Runs once per level. Legit: real telepathy, real gameplay, no wizard peek."""
            try:
                lvl = agent.current_level()
                lvl_id = (int(lvl.dungeon_number), int(lvl.level_number))
            except Exception:
                return False
            seen = agent.__dict__.setdefault("_lore_esp_levels", set())
            if lvl_id in seen:
                return False
            towel = None
            try:
                for nm, oc, ltr in zip(agent.last_observation['inv_strs'],
                                       agent.last_observation['inv_oclasses'],
                                       agent.last_observation['inv_letters']):
                    if int(ltr) == 0:
                        continue
                    if 'towel' in bytes(nm).decode('latin1').lower():
                        towel = chr(int(ltr)); break
            except Exception:
                return False
            if towel is None:
                return False
            seen.add(lvl_id)
            low = agent.env.env.unwrapped.env
            try:
                # baseline: monsters visible by SIGHT before blinding (for the A/B)
                try:
                    _sighted = len(agent.monster_tracker.take_all_monsters())
                except Exception:
                    _sighted = -1
                low.step(ord('P')); low.step(ord(towel)); low.step(13)     # put on -> blind + ESP
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
                mons = agent.monster_tracker.take_all_monsters()           # {(y,x): name}, ESP-wide
                lore_patches.COUNTERS["esp_sighted_total"] = \
                    lore_patches.COUNTERS.get("esp_sighted_total", 0) + max(0, _sighted)
                threats = [(int(y), int(x), name.lower()) for (y, x), name in mons.items()
                           if any(t in name.lower() for t in _INSTADEATH_ESP)]
                agent.__dict__["_lore_esp_threats"] = threats
                lore_patches.COUNTERS["esp_scans"] = lore_patches.COUNTERS.get("esp_scans", 0) + 1
                lore_patches.COUNTERS["esp_mons_total"] = \
                    lore_patches.COUNTERS.get("esp_mons_total", 0) + len(mons)
                lore_patches.COUNTERS["esp_threats_total"] = \
                    lore_patches.COUNTERS.get("esp_threats_total", 0) + len(threats)
                _tn = lore_patches.COUNTERS.setdefault("esp_threat_names", [])
                for _, _, n in threats:
                    if n not in _tn and len(_tn) < 20:
                        _tn.append(n)
            except AgentFinished:
                raise
            except Exception as _e:
                lore_patches.COUNTERS["esp_err"] = repr(_e)[:80]
            finally:
                try:
                    low.step(ord('R')); low.step(ord(towel)); low.step(13)  # take off -> unblind
                    agent.step(_agz.A.Command.ESC); agent.inventory.update()
                except Exception:
                    pass
            return False

        def _heal_reflex():
            """Quaff a potion of healing when HP is low, to sustain the long
            Gehennom traversal (prayer alone can't -- ~1000-turn cooldown). Returns
            True ONLY if a game turn actually passed (TURN-GUARD: a reflex that
            returns True without a step makes the descend loop spin to the cap)."""
            try:
                # top up at 65% -- Gehennom deaths are BURSTS (multiple strong
                # monsters), so heal before HP is low enough for one to be lethal.
                if agent.blstats.hitpoints > 0.65 * max(1, agent.blstats.max_hitpoints):
                    return False
                t0 = int(agent.blstats.time)
            except Exception:
                return False
            low = agent.env.env.unwrapped.env
            # healing potions are UNIDENTIFIED ("black potions"), so match by the
            # LETTER captured at setup (stable per stack), then by stored appearance,
            # then by an identified 'healing' name (post-first-quaff). Verify the
            # candidate letter still points to a potion before quaffing.
            hlt = agent.__dict__.get("_lore_heal_lt")
            happ = agent.__dict__.get("_lore_heal_app")
            try:
                lt = None
                for nm, oc, ltr in zip(agent.last_observation['inv_strs'],
                                       agent.last_observation['inv_oclasses'],
                                       agent.last_observation['inv_letters']):
                    if int(oc) != _nh2.POTION_CLASS or int(ltr) == 0:
                        continue
                    c = chr(int(ltr))
                    s = bytes(nm).decode('latin1').strip('\x00').strip().lower()
                    if c == hlt or 'healing' in s or (happ and s.split(' ', 1)[-1] in happ):
                        lt = c; break
                if lt is None:
                    return False
                low.step(ord('q')); low.step(ord(lt)); low.step(13); low.step(13)
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
            except Exception:
                return False
            try:
                if int(agent.blstats.time) > t0:      # TURN-GUARD
                    lore_patches.COUNTERS["descent_heals"] = \
                        lore_patches.COUNTERS.get("descent_heals", 0) + 1
                    return True
            except Exception:
                pass
            return False

        def _apply_unicorn_horn():
            """Apply the blessed unicorn horn to cure sickness/blind/confuse/stun/
            stat-drain -- the Gehennom substitute for prayer (which goes to Moloch).
            Find the horn by name (wished -> identified as 'unicorn horn'). Raw 'a'
            apply keypresses; TURN-GUARDED so a no-op can't spin the loop."""
            try:
                t0 = int(agent.blstats.time)
                lt = None
                for nm, oc, ltr in zip(agent.last_observation['inv_strs'],
                                       agent.last_observation['inv_oclasses'],
                                       agent.last_observation['inv_letters']):
                    if int(ltr) == 0 or int(oc) != _nh2.TOOL_CLASS:
                        continue
                    if 'unicorn horn' in bytes(nm).decode('latin1').lower():
                        lt = chr(int(ltr)); break
                if lt is None:
                    return False
                low = agent.env.env.unwrapped.env
                low.step(ord('a')); low.step(ord(lt)); low.step(13); low.step(13)
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
                if int(agent.blstats.time) > t0:
                    lore_patches.COUNTERS["horn_applies"] = \
                        lore_patches.COUNTERS.get("horn_applies", 0) + 1
                    return True
            except Exception:
                pass
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

        def _panic_escape():
            """KB Gehennom panic button (playbook s4 escape priority): when swarmed or
            low HP, READ an escape scroll -- scroll of scare monster freezes the swarm,
            scroll of teleportation whisks the char away. Scrolls are unidentified, so
            prefer one already LEARNED to be scare/teleport; else read an unknown one
            and learn its type from the effect (8/14 kit scrolls are escapes). Returns
            True if a scroll was read (a game turn passed)."""
            low = agent.env.env.unwrapped.env
            learned = agent.__dict__.setdefault("_lore_scroll_type", {})
            scrolls = []
            try:
                for oc, ltr in zip(agent.last_observation['inv_oclasses'],
                                   agent.last_observation['inv_letters']):
                    if int(oc) == _nh2.SCROLL_CLASS and int(ltr) != 0:
                        scrolls.append(chr(int(ltr)))
            except Exception:
                return False
            if not scrolls:
                return False
            # priority: known escape > unknown (learn it) > known magic-map (last resort)
            pref = ([c for c in scrolls if learned.get(c) in ('scare', 'teleport')]
                    + [c for c in scrolls if c not in learned]
                    + [c for c in scrolls if learned.get(c) == 'magicmap'])
            lt = pref[0] if pref else scrolls[0]
            try:
                t0 = int(agent.blstats.time)
                py, px = int(agent.blstats.y), int(agent.blstats.x)
                msgs = []
                low.step(ord('r'))
                msgs.append(_raw_msg(low.step(ord(lt))))
                for _ in range(3):
                    msgs.append(_raw_msg(low.step(13)))
                agent.step(_agz.A.Command.ESC); agent.inventory.update()
                full = ' '.join(msgs)
                # classify to learn the letter's type for next time
                if 'coalesces' in full or 'map of' in full:
                    learned[lt] = 'magicmap'
                elif 'scare' in full or 'flee' in full or 'maniacal' in full \
                        or 'sense of loss' in full:
                    learned[lt] = 'scare'
                elif 'teleport' in full or 'disorient' in full \
                        or (int(agent.blstats.y), int(agent.blstats.x)) != (py, px):
                    learned[lt] = 'teleport'
                lore_patches.COUNTERS["panic_reads"] = \
                    lore_patches.COUNTERS.get("panic_reads", 0) + 1
                lore_patches.COUNTERS["scroll_types"] = dict(learned)
                return int(agent.blstats.time) > t0
            except Exception:
                return False

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
                "have_downstair": (int(_u.isin(lvl.objects, G.STAIR_DOWN).sum()) > 0
                                   or bool(agent.__dict__.get("_lore_downs_%d" % int(agent.blstats.depth)))),
                "downstair_reachable": bool((_u.isin(lvl.objects, G.STAIR_DOWN) & (agent.bfs() != -1)).any()),
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
                # fixed priority, no LLM. PROPER-DESCENT ORDER (LORE_STAIRS_FIRST):
                # take the STAIRS when a downstair is known -- stairs arrive on the
                # next level's UPSTAIR, which NetHack guarantees is connected to that
                # level's downstair, so no walled pockets. Dig-down (falling to a
                # random cell) can drop you into a sealed pocket -- use it only as a
                # fallback when no downstair is known. With LORE_REVEAL the ^F map
                # makes the downstair known immediately on each level.
                # DEFAULT = STAIRS-FIRST (proper descent). The goal is ASCENSION, not
                # max depth. Dig-down-fast reaches deeper per-run only because it
                # recklessly plummets into danger and dies -- a local minimum, the
                # OPPOSITE of "strong and prepared". Taking stairs arrives on the next
                # level's UPSTAIR (guaranteed connected to its downstair -> no pockets,
                # no random drops into a swarm); the char stays intact and can survive
                # the whole way. Depth-per-run is a red herring. LORE_STAIRS_FIRST=0
                # opts back into dig-plummet for comparison only.
                st = _build_state()
                if _os2.environ.get("LORE_STAIRS_FIRST", "1") == "1":
                    # take the stair when it is REACHABLE (proper descent). When the
                    # downstair is known but UNREACHABLE (the ^V lands at a random cell
                    # walled off from it -- e.g. across the Valley's moat), don't loop
                    # on it forever: DIG_DOWN falls straight past the barrier to the
                    # next level. In a real descent (arrive on the connected upstair)
                    # the stair is reachable, so this fallback only fixes the placement
                    # artifact -- not reckless plummeting.
                    if st["downstair_reachable"]:
                        return "DESCEND_STAIRS"
                    # downstair KNOWN (via ^F reveal) but UNREACHABLE: ^F reveals the
                    # glyph but AA's walkable/bfs doesn't traverse revealed-unwalked
                    # corridors, so the stair only becomes reachable once we EXPLORE
                    # (walk) the connecting corridors of the (real, connected) level.
                    # Retrying the stair or digging (no-dig Valley) loops forever ->
                    # EXPLORE to actually connect the path to the downstair.
                    if st["has_dig_wand"] and not st["level_no_dig"]:
                        return "DIG_DOWN"
                    return "EXPLORE"
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
                # (in-loop wiz_map reveal disabled: firing ^F mid-descend corrupts
                # AA's observation state -- inventory.update fails, _wand_letter
                # returns None, agent freezes at explored_cells=1. Reveal works in a
                # one-shot probe but not here; needs a proper resync primitive before
                # re-enabling. Downstairs are found by exploration meanwhile.)
                if _os2.environ.get("LORE_REVEAL") == "1":
                    try: _reveal_level()
                    except AgentFinished: raise
                    except Exception: pass
                # VIBRATING-SQUARE detection: this loop runs INSIDE AA's main loop
                # (observations update), so watch for the invocation square's message
                # while exploring. Found => we've reached the invocation level.
                try:
                    _vm = str(agent.message).lower()
                    if "vibrat" in _vm and "vibration_found" not in lore_patches.COUNTERS:
                        lore_patches.COUNTERS["vibration_found"] = int(agent.blstats.depth)
                        lore_patches.COUNTERS["vibration_pos"] = [int(agent.blstats.y), int(agent.blstats.x)]
                        raise AgentFinished()
                except AgentFinished:
                    raise
                except Exception:
                    pass
                d = int(agent.blstats.depth)
                if d > lore_patches.COUNTERS.get("max_depth", 0):
                    lore_patches.COUNTERS["max_depth"] = d
                # WATER-AWARE nav: the char wears water-walking boots, so moats/pools
                # are crossable -- but AA has NO water glyph set and excludes water
                # from `walkable`, so a downstair across a moat is unreachable and the
                # char is stranded (the real 'pocket'). Detect water from the RAW
                # glyphs (S_pool/S_water cmap) and mark it walkable so bfs/go_to path
                # across the moat to the downstair. (Water-walking makes this safe;
                # lava excluded.) This is the ascension-focused deterministic fix.
                try:
                    import nle.nethack as _nhw
                    _g = agent.glyphs
                    _water = (_g == _nhw.GLYPH_CMAP_OFF + 32) | (_g == _nhw.GLYPH_CMAP_OFF + 41)
                    if bool(_water.any()):
                        agent.current_level().walkable[_water] = True
                        lore_patches.COUNTERS["water_marked"] = \
                            max(lore_patches.COUNTERS.get("water_marked", 0), int(_water.sum()))
                except Exception:
                    pass
                # LOOP-LEVEL LLM NAV (LORE_LLM_NAV): if DEPTH stalls for many iters we
                # are walled into a pocket. Fires regardless of which primitive the
                # policy picks (stairs-first routes walled-downstair to prim_descend_
                # stairs, not prim_explore, so the in-primitive hook never ran). Hand
                # the LLM the revealed ASCII and dig the cardinal it points toward the
                # walled-off downstair; re-query each stall so it tunnels an L-path.
                _di2 = int(lore_patches.COUNTERS.get("descend_iters", 0))
                if agent.__dict__.get("_lln_depth") != d:
                    agent.__dict__["_lln_depth"] = d
                    agent.__dict__["_lln_iter"] = _di2
                elif _os2.environ.get("LORE_LLM_NAV") == "1" and \
                        _di2 - agent.__dict__.get("_lln_iter", _di2) > 40:
                    agent.__dict__["_lln_iter"] = _di2
                    try:
                        lvlA = agent.current_level(); bfA = agent.bfs()
                        myA = (int(agent.blstats.y), int(agent.blstats.x))
                        downA = _u.isin(lvlA.objects, G.STAIR_DOWN)
                        upA = _u.isin(lvlA.objects, G.STAIR_UP)
                        wallA = _u.isin(lvlA.objects, G.WALL)
                        rowsA = []
                        for ry in range(lvlA.objects.shape[0]):
                            rr = ""
                            for rx in range(lvlA.objects.shape[1]):
                                if (ry, rx) == myA: rr += "@"
                                elif downA[ry, rx]: rr += ">"
                                elif upA[ry, rx]: rr += "<"
                                elif wallA[ry, rx]: rr += "|"
                                elif bfA[ry, rx] != -1: rr += "."
                                elif lvlA.walkable[ry, rx]: rr += ":"
                                else: rr += " "
                            rowsA.append(rr.rstrip())
                        act = _oracle.query_nav_dig("\n".join(rowsA))
                        lore_patches.COUNTERS["llm_nav_q"] = lore_patches.COUNTERS.get("llm_nav_q", 0) + 1
                        lore_patches.COUNTERS["llm_nav_" + str(act)] = \
                            lore_patches.COUNTERS.get("llm_nav_" + str(act), 0) + 1
                        # GROUND-TRUTH check: does the LLM's cardinal match the actual
                        # direction to the nearest known downstair? (dominant axis).
                        try:
                            _dn = list(zip(*downA.nonzero()))
                            if _dn:
                                ty, tx = min(_dn, key=lambda p: abs(p[0] - myA[0]) + abs(p[1] - myA[1]))
                                if abs(ty - myA[0]) >= abs(tx - myA[1]):
                                    truth = "DIG_SOUTH" if ty > myA[0] else "DIG_NORTH"
                                else:
                                    truth = "DIG_EAST" if tx > myA[1] else "DIG_WEST"
                                key = "llm_nav_correct" if act == truth else "llm_nav_wrong"
                                lore_patches.COUNTERS[key] = lore_patches.COUNTERS.get(key, 0) + 1
                        except Exception:
                            pass
                        dmap = {"DIG_NORTH": ("k", -1, 0), "DIG_SOUTH": ("j", 1, 0),
                                "DIG_EAST": ("l", 0, 1), "DIG_WEST": ("h", 0, -1)}
                        wl = _wand_letter()
                        if act in dmap and wl:
                            dc, dy, dx = dmap[act]
                            y0, x0 = myA
                            with agent.atom_operation():
                                agent.step(_agz.A.Command.ZAP); agent.type_text(wl); agent.type_text(dc)
                            lore_patches.COUNTERS["llm_digs"] = \
                                lore_patches.COUNTERS.get("llm_digs", 0) + 1
                            if 'too hard' not in str(agent.message):
                                ny, nx = y0 + dy, x0 + dx
                                try:
                                    if lvlA.objects.shape[0] > ny >= 0 <= nx < lvlA.objects.shape[1] \
                                            and agent.current_level().walkable[ny, nx]:
                                        agent.move(ny, nx)
                                except Exception:
                                    pass
                            continue
                    except AgentFinished:
                        raise
                    except Exception as _le:
                        lore_patches.COUNTERS["llm_nav_err"] = repr(_le)[:60]
                # STRENGTH-RETENTION metric (Jim, Jul 13): a STRONG surviving char is
                # the goal, not raw depth. Track peak vs current XL (drain = getting
                # weaker: vampires/mind-flayers/wraiths drain levels) and min HP frac.
                # xl_drained = peak_xl - current_xl (0 = stayed strong).
                try:
                    _xl = int(agent.blstats.experience_level)
                    _pk = max(lore_patches.COUNTERS.get("peak_xl", 0), _xl)
                    lore_patches.COUNTERS["peak_xl"] = _pk
                    lore_patches.COUNTERS["cur_xl"] = _xl
                    lore_patches.COUNTERS["xl_drained"] = _pk - _xl
                    _hpf = agent.blstats.hitpoints / max(1, agent.blstats.max_hitpoints)
                    lore_patches.COUNTERS["min_hp_frac"] = round(
                        min(lore_patches.COUNTERS.get("min_hp_frac", 1.0), _hpf), 2)
                    lore_patches.COUNTERS["cur_ac"] = int(agent.blstats.armor_class)
                except Exception:
                    pass
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
                if lore_patches.COUNTERS.get("descend_iters", 0) == 100 and \
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
                # LORE_BLINDFOLD: blindfold-navigation ESP scan (once per level) --
                # sense mind flayers / eels / covetous liches through walls before
                # walking into them. Self-limits per level; returns False (passive).
                if _os2.environ.get("LORE_BLINDFOLD") == "1":
                    try:
                        _telepathy_scan()
                    except AgentFinished:
                        raise
                    except Exception:
                        pass
                # PANIC FIRST (before heal): if swarmed, the KB says break the swarm
                # (scare/teleport) rather than heal in place -- healing while 4 monsters
                # keep hitting just trades potions for a death. Escape, THEN heal.
                try:
                    _pm = agent.get_visible_monsters()
                    if _pm and sum(1 for m in _pm if int(m[0]) <= 3) >= 3:
                        if _panic_escape():
                            continue
                except AgentFinished:
                    raise
                except Exception:
                    pass
                # survival reflex: heal when hurt (TURN-GUARDED so it can't spin --
                # only continues if a real game step happened).
                try:
                    if _heal_reflex():
                        continue
                except Exception:
                    pass
                # sickness reflex: illness (food-poison) is a top Gehennom killer
                # that HP potions can't cure. PRAYER IS DEAD in Gehennom (goes to
                # Moloch) -- the correct cure is the UNICORN HORN (playbook s4).
                # Apply the blessed horn to cure sick/blind/confuse/stun/stat-drain.
                try:
                    if _is_sick(agent) and _apply_unicorn_horn():
                        continue
                except AgentFinished:
                    raise
                except Exception:
                    pass
                # intrinsic reflex: eat fresh kills / carried corpses on the way down
                # to gain the resistances the char still LACKS (fire/cold/shock/
                # poison/sleep/disint). The legit ascension-prep mechanic -- no wizard
                # grants. Runs before the hunger reflex so a useful corpse is eaten for
                # its resistance rather than a nutrition-only lizard.
                try:
                    if _eat_intrinsic_reflex():
                        continue
                except AgentFinished:
                    raise
                except Exception:
                    pass
                # hunger reflex: eat CARRIED safe food (lizard corpse / ration) when
                # hungry -- replaces the removed ground-corpse eating (no tainted
                # corpses, no prayer-for-food in Gehennom).
                try:
                    if _eat_if_hungry():
                        continue
                except Exception:
                    pass
                # threat reflex: engage any hostile within range BEFORE navigating,
                # so the tank fights instead of walking/searching into free hits (the
                # chip-death mode). Bare adjacency missed approaching monsters (they
                # hit DURING a multi-turn move); get_visible_monsters() is range-aware
                # and BFS-reachable, so we catch them at distance <= 6 and fight2
                # clears the local swarm. This is why the FIGHT-policy runs survived
                # ~5x longer than the EXPLORE-locked ones.
                try:
                    mons = agent.get_visible_monsters()
                    if mons:
                        lore_patches.COUNTERS["mons_seen"] = \
                            lore_patches.COUNTERS.get("mons_seen", 0) + 1
                        lore_patches.COUNTERS["nearest_mon_min"] = min(
                            lore_patches.COUNTERS.get("nearest_mon_min", 999), int(mons[0][0]))
                    # HP-trajectory: record min HP fraction seen (chip vs burst death)
                    try:
                        _hpf = agent.blstats.hitpoints / max(1, agent.blstats.max_hitpoints)
                        lore_patches.COUNTERS["min_hp_frac"] = round(min(
                            lore_patches.COUNTERS.get("min_hp_frac", 1.0), _hpf), 2)
                    except Exception:
                        pass
                    # LORE_AVOIDDRAIN: NEVER melee level-drainers / bursters. The
                    # DL26-30 deaths are XP drain (vampires/wraiths gut XL30->3 -> max
                    # HP collapses) and xorn bursts (phase through walls). Our kit has
                    # NO drain resistance (artifact-only, unwishable for a Valkyrie), so
                    # the fix is behavioral: flee/panic-escape them, don't trade blows.
                    if _os2.environ.get("LORE_AVOIDDRAIN") == "1" and mons and mons[0][0] <= 6:
                        _DRAINERS = ('vampire', 'wraith', 'xorn', 'lich', 'mind flayer',
                                     'energy vortex', 'shade', 'ghost', 'wight')
                        _dn = None
                        try:
                            for _m in mons:
                                if int(_m[0]) > 6:
                                    continue
                                _mn = getattr(_m[3], 'mname', '').lower()
                                if any(t in _mn for t in _DRAINERS):
                                    _dn = _mn; break
                        except Exception:
                            _dn = None
                        if _dn is not None:
                            if not _panic_escape():
                                prim_flee()
                            lore_patches.COUNTERS["drain_flees"] = \
                                lore_patches.COUNTERS.get("drain_flees", 0) + 1
                            _dtn = lore_patches.COUNTERS.setdefault("drain_threat_names", [])
                            if _dn not in _dtn and len(_dtn) < 15:
                                _dtn.append(_dn)
                            continue
                    if mons and mons[0][0] <= 6:
                        # PLAYBOOK: flee by default, fight only when cornered. A
                        # strong char that stands and fights Gehennom SWARMS still
                        # dies (dragons/nagas gang up). Count near threats; if
                        # OVERWHELMED (>=3 within 4) or hurt with threats near,
                        # FLEE + heal instead of trading blows to death. Fight only
                        # 1-2 threats (clear the path).
                        near = sum(1 for m in mons if int(m[0]) <= 4)
                        try:
                            hpf = agent.blstats.hitpoints / max(1, agent.blstats.max_hitpoints)
                        except Exception:
                            hpf = 1.0
                        if near >= 3 or (hpf < 0.5 and near >= 1):
                            # KB panic button FIRST: read scare-monster/teleport to
                            # break the swarm (prim_flee just walks -- the swarm
                            # follows and chips the tank to death, the DL27-30 ~200-turn
                            # death mode). Fall back to flee only if no scroll fired.
                            if not _panic_escape():
                                prim_flee()
                            lore_patches.COUNTERS["reflex_flees"] = \
                                lore_patches.COUNTERS.get("reflex_flees", 0) + 1
                        else:
                            prim_fight()
                            lore_patches.COUNTERS["reflex_fights"] = \
                                lore_patches.COUNTERS.get("reflex_fights", 0) + 1
                        continue
                except AgentFinished:
                    raise
                except _gl_exc.AgentChangeStrategy:
                    raise
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
            # NO ground-corpse eating in Gehennom: tainted/poisonous corpses cause
            # fatal illness and there is no prayer to cure it. The char eats its
            # carried SAFE food (lizard corpses / rations) via the hunger reflex.
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
    # Candelabrum-of-Invocation state parentheticals also assert (AA has no model
    # for the invocation items): "(no candles attached)", "(7 candles attached)",
    # "(7 candles, lit)". Strip them so parse_text treats it as a plain carried item.
    _CAND = re.compile(r"\((?:no candles attached|\d+ candles(?: attached|, lit| attached, lit)?)\)")
    _orig = _im.ItemManager.parse_text.__func__ if hasattr(_im.ItemManager.parse_text, "__func__") \
        else _im.ItemManager.parse_text

    def _patched(text, category=None, glyph=None):
        text = _CAND.sub("", text)
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
