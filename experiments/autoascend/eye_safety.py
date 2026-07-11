"""Fix for the 'frozen by a floating eye's gaze' deaths (~3% of games). Meleeing a
floating eye triggers its passive gaze -> paralysis for many turns -> other adjacent
monsters kill the frozen bot. AA already penalizes floating-eye melee (-110 priority),
but that soft penalty gets overridden in multi-monster fights (everything else scores
even lower), so it still melees. Fix (diagnosed + LLM-confirmed): make floating-eye
melee an ABSOLUTE no when any OTHER hostile monster is near (paralysis is only lethal
when something else can hit you; a lone floating eye at full HP is survivable, so we
only hard-block when others are present)."""


def apply():
    from autoascend.combat import fight_heur as _fh
    orig = _fh.melee_monster_priority

    def patched(agent, monsters, monster):
        p = orig(agent, monsters, monster)
        try:
            if monster[3].mname == 'floating eye':
                ay, ax = int(agent.blstats.y), int(agent.blstats.x)
                for m in agent.get_visible_monsters():
                    if m[3].mname == 'floating eye':
                        continue
                    if max(abs(int(m[1]) - ay), abs(int(m[2]) - ax)) <= 2:
                        return float('-inf')  # other hostile near -> never melee the eye
        except Exception:
            pass
        return p

    _fh.melee_monster_priority = patched
    return ["eye_safety (no floating-eye melee with another hostile near)"]
