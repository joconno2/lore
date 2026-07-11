"""Test whether the DL5 wall is SENSITIVE to AA's own emergency-reaction HP
thresholds (the last untested lever class, and the parameter space an option-(b)
ES/QD would search). AA's emergency_strategy heals at hp<1/3 max and prays at
hp<1/5 (XL<6) / 1/6 (XL>=6). Hypothesis: AA reacts too late in fast mid-game
swarms. This replaces emergency_strategy with a faithful copy whose thresholds
are raised (react earlier), so we can measure depth sensitivity.

If raising thresholds shifts median depth -> the wall has parametric room (option
b via param-search viable). If null/worse -> AA's thresholds are near-optimal and
the wall is at AA's optimum (fundamental). Env: LORE_HEALFRAC, LORE_PRAYFRAC."""
import os


def apply(heal_frac=None, pray_frac=None):
    from autoascend.strategy import Strategy
    from autoascend.agent import Agent
    from autoascend.item import flatten_items
    from autoascend.glyph import Hunger
    import nle.nethack as nh

    HF = float(os.environ.get("LORE_HEALFRAC", heal_frac if heal_frac is not None else 0.5))
    PF = float(os.environ.get("LORE_PRAYFRAC", pray_frac if pray_frac is not None else 0.4))

    def emergency(self):
        items = [it for it in flatten_items(self.inventory.items) if it.is_unambiguous() and
                 it.category == nh.POTION_CLASS and it.object.name in ['healing', 'extra healing', 'full healing']]
        if (self.blstats.hitpoints < HF * self.blstats.max_hitpoints or self.blstats.hitpoints < 8) and items:
            yield True
            self.inventory.quaff(items[0]); return

        items = [it for it in flatten_items(self.inventory.items) if it.is_unambiguous() and
                 it.category == nh.POTION_CLASS and it.object.name in ['fruit juice']]
        if items and self.blstats.hunger_state >= Hunger.FAINTING:
            yield True
            self.inventory.quaff(items[0]); return

        if (
            (self.is_safe_to_pray(500) and
             (self.blstats.hitpoints < PF * self.blstats.max_hitpoints or self.blstats.hitpoints < 6))
            or (self.is_safe_to_pray(400) and self.blstats.hunger_state >= Hunger.FAINTING)
        ):
            yield True
            self.pray(); return

        yield False

    Agent.emergency_strategy = Strategy.wrap(emergency)
    return ["emergency_boost (heal<%.2f, pray<%.2f)" % (HF, PF)]
