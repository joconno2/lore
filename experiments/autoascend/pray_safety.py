"""Fix for the 'killed while praying' deaths (~3% of games, all: an adjacent
hostile monster kills the bot mid-prayer). AA's is_safe_to_pray only checks the
prayer TIMEOUT (god's mood), not adjacent threats -- but praying takes ~4-10 turns
of no action, so an adjacent attacker kills the low-HP bot before the prayer
completes. Fix (diagnosed + independently confirmed by the LLM analyzer): don't
treat praying as safe when a hostile monster is adjacent. get_visible_monsters()
is already hostile-only (peaceful excluded)."""


def apply():
    from autoascend.agent import Agent
    orig = Agent.is_safe_to_pray

    def patched(self, limit=500):
        if not orig(self, limit):
            return False
        try:
            ay, ax = int(self.blstats.y), int(self.blstats.x)
            for m in self.get_visible_monsters():
                _, my, mx, mon, _ = m
                if max(abs(int(my) - ay), abs(int(mx) - ax)) <= 1:
                    return False  # adjacent hostile -> would die mid-prayer
        except Exception:
            pass
        return True

    Agent.is_safe_to_pray = patched
    return ["pray_safety (no pray with adjacent hostile monster)"]
