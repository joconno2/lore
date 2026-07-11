"""Structural fix for the Elbereth-engraving-loop death (kills ~24% of games,
34% of combat deaths). AA's elbereth_action engraves Elbereth (dust) vs adjacent
threats when HP<30, then waits to heal (melee suppressed -100). But @ humans
(mlet==5: soldiers, watchmen, lieutenants, elves) and minotaurs are Elbereth-
IMMUNE -- they keep attacking, the dust engraving degrades, AA re-engraves in a
loop ("What do you want to write in the dust here?") and dies. Matches the
'wand of striking' Yendorian-soldier deaths (the #1 'a wand' killer).

Fix: don't engrave if the only adjacent dangerous monsters are Elbereth-immune.
Then AA never engraves against them -> the -100 melee suppression never triggers
-> AA melees/flees instead. Scarable threats keep the original behavior.

apply() monkeypatches fight_heur.elbereth_action (called by name as a module
global in get_available_actions, so replacing the module attr takes effect)."""

IMMUNE_MLET = {5}  # @ human class: soldier/watchman/lieutenant/captain/elf/human
COUNTERS = {"elb_suppressed": 0}


def _is_immune(mon):
    try:
        if getattr(mon, "mlet", None) in IMMUNE_MLET:
            return True
        if "minotaur" in getattr(mon, "mname", ""):
            return True
    except Exception:
        pass
    return False


def apply():
    from autoascend.combat import fight_heur as _fh
    adjacent = _fh.adjacent
    orig = _fh.elbereth_action

    def patched(agent, monsters):
        scarable_adjacent = False
        for m in monsters:
            _, my, mx, mon, _ = m
            if not adjacent((my, mx), (agent.blstats.y, agent.blstats.x)):
                continue
            if _is_immune(mon):
                continue
            scarable_adjacent = True
            break
        if not scarable_adjacent:
            COUNTERS["elb_suppressed"] += 1
            return []  # only Elbereth-immune threats adjacent -> engraving is useless
        return orig(agent, monsters)

    _fh.elbereth_action = patched
    return ["elbereth_fix (immune @/minotaur -> no engrave)"]
