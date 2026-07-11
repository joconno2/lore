"""Test the container-recheck fix. Root cause: InventoryItems.update re-checks
already-known-empty sacks every update because _recheck_containers persists --
the recursive re-update at line 119 returns before line 129 clears it, and
re-applying the empty sack perturbs inv_strs, re-triggering forever. AA never
descends (~50% of seeds stick at DL1).

Fix (one-shot semantics): capture _recheck_containers into a local at the top of
the update body and clear the instance flag immediately, so the recursive
re-entry does not re-check known containers. use_container still refreshes
modified containers via its own explicit check_container_content pass, so bag
functionality is preserved.

Run the 3 known-stuck seeds; report depth. Fix works if they descend past DL1."""
import sys, json, gym, nle
import nle.nethack as nh
from autoascend import objects as O
from autoascend.item.inventory_items import InventoryItems

seed = int(sys.argv[1]); OUT = sys.argv[2]; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 8000

def update_fixed(self, force=False):
    if force:
        self._recheck_containers = True
    # ONE-SHOT: consume the recheck intent now so the recursive re-update below
    # cannot re-trigger a re-check of already-known containers (the DL1 loop).
    do_recheck = self._recheck_containers
    self._recheck_containers = False

    if force or self._previous_inv_strs is None or \
            (self.agent.last_observation['inv_strs'] != self._previous_inv_strs).any():
        self._clear()
        self._previous_inv_strs = self.agent.last_observation['inv_strs']
        previous_inv_strs = self._previous_inv_strs

        iterable = set()
        for item_name, category, glyph, letter in zip(
                self.agent.last_observation['inv_strs'],
                self.agent.last_observation['inv_oclasses'],
                self.agent.last_observation['inv_glyphs'],
                self.agent.last_observation['inv_letters']):
            item_name = bytes(item_name).decode().strip('\0')
            letter = chr(letter)
            if not item_name:
                continue
            iterable.add((item_name, category, glyph, letter))
        iterable = sorted(iterable, key=lambda x: x[-1])

        assert len(iterable) == len(set(map(lambda x: x[-1], iterable))), \
            'letters in inventory are not unique'

        for item_name, category, glyph, letter in iterable:
            item = self.agent.inventory.item_manager.get_item_from_text(
                item_name, category=category,
                glyph=glyph if not nh.glyph_is_body(glyph) and not nh.glyph_is_statue(glyph) else None,
                position=None)

            self.all_items.append(item)
            self.all_letters.append(letter)

            if item.equipped:
                for types, sub, name in [
                    ((O.Weapon, O.WepTool), None, 'main_hand'),
                    (O.Armor, O.ARM_SHIELD, 'off_hand'),
                    (O.Armor, O.ARM_SUIT, 'suit'),
                    (O.Armor, O.ARM_HELM, 'helm'),
                    (O.Armor, O.ARM_GLOVES, 'gloves'),
                    (O.Armor, O.ARM_BOOTS, 'boots'),
                    (O.Armor, O.ARM_CLOAK, 'cloak'),
                    (O.Armor, O.ARM_SHIRT, 'shirt'),
                ]:
                    if isinstance(item.objs[0], types) and (sub is None or sub == item.objs[0].sub):
                        assert getattr(self, name) is None, ((name, getattr(self, name), item), str(self), iterable)
                        setattr(self, name, item)
                        break

            if item.is_possible_container() or (item.is_container() and do_recheck):
                self.agent.inventory.check_container_content(item)

            if (self.agent.last_observation['inv_strs'] != previous_inv_strs).any():
                self.update()
                return

            self.total_weight += item.weight()

InventoryItems.update = update_fixed

env = gym.make("NetHackChallenge-v0")
try: env.seed(seed, seed)
except Exception: pass
STEPS = [0]; DEPTH = [1]
_os = env.step
def _h(a):
    r = _os(a); STEPS[0] += 1
    try: DEPTH[0] = max(DEPTH[0], int(r[0]["blstats"][12]))
    except Exception: pass
    if STEPS[0] > CAP:
        raise KeyboardInterrupt("cap")
    return r
env.step = _h

w = __import__("autoascend.env_wrapper", fromlist=["EnvWrapper"]).EnvWrapper(
    env, agent_args=dict(panic_on_errors=False, verbose=False))
try: w.main()
except BaseException as e: pass
s = w.get_summary()
json.dump({"seed": seed, "steps": STEPS[0], "max_depth": DEPTH[0],
           "turns": s.get("turns"), "xl": s.get("experience_level"),
           "end_reason": str(s.get("end_reason"))[:120]}, open(OUT, "w"), default=str)
print("DONE seed", seed, "depth", DEPTH[0], "steps", STEPS[0], flush=True)
