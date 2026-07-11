"""Elbereth-loop fix v2 (loop-cap, broader than the @-immune v1 which was a
no-op). The 24%-of-games loop is dust-degradation: AA engraves Elbereth, a monster
attack degrades the dust ('Elbereth'->'Elberet'), engraving_below_me != 'elbereth'
so elbereth_action fires again, AA re-engraves instead of fighting/fleeing, and
dies while looping. v1 (skip @-immune) never fired because the loops are scarable
monsters/pets, not @ humans.

v2: cap consecutive Elbereth engraves per encounter. Track engrave turns; if AA
has engraved Elbereth >= CAP times within WINDOW turns, suppress elbereth_action
(return []) so AA's melee/flee/zap actions take over and it stops the futile loop.
Pure return-[] (no AgentPanic), so it can't crash like the descent_gate."""

CAP = 3
WINDOW = 25
COUNTERS = {"elb_engraves": 0, "elb_loop_suppressed": 0}


def apply():
    from autoascend.combat import fight_heur as _fh
    from autoascend.agent import Agent

    orig_engrave = Agent.engrave
    def engrave(self, text):
        r = orig_engrave(self, text)
        try:
            if text.lower() == "elbereth":
                self.__dict__.setdefault("_lore_elb_turns", []).append(int(self.blstats.time))
                COUNTERS["elb_engraves"] += 1
        except Exception:
            pass
        return r
    Agent.engrave = engrave

    orig_elb = _fh.elbereth_action
    def patched(agent, monsters):
        try:
            turns = agent.__dict__.get("_lore_elb_turns", [])
            if turns:
                now = int(agent.blstats.time)
                recent = [t for t in turns if now - t <= WINDOW]
                agent.__dict__["_lore_elb_turns"] = recent
                if len(recent) >= CAP:
                    COUNTERS["elb_loop_suppressed"] += 1
                    return []  # break the futile re-engrave loop -> force fight/flee
        except Exception:
            pass
        return orig_elb(agent, monsters)
    _fh.elbereth_action = patched
    return ["elbereth_fix2 (cap=%d/%dturns -> break loop)" % (CAP, WINDOW)]
