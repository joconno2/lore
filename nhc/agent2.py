"""LORE Expert Agent v2: Clean room rewrite based on AutoAscend architecture.

Key differences from v1 (expert_agent.py):
- Agent DRIVES the env via step(), not the other way around
- Recursive prompt handling (step -> update -> step)
- Strategy-based game loop with preemptible priorities
- Proper state tracking per level
- All systems integrated (food, equipment, combat, navigation)

Usage:
    agent = AgentV2(env, verbose=True)
    agent.main()  # runs until episode ends
"""
from __future__ import annotations

import contextlib
import numpy as np
from collections import namedtuple, Counter
from typing import Optional

import nle.nethack as nh
from nle.nethack import actions as A

from nhc.food import FoodManager, NO_CORPSE, UNDEAD_FRAGMENTS
from nhc.equipment import EquipmentManager, WEAPON_DATA
from nhc.strategy import StrategyManager, Milestone
from nhc.fight import (
    assess_monster, should_elbereth, pick_melee_target, should_flee,
    NEVER_MELEE, INSTAKILL, PEACEFUL_IDS, PEACEFUL_NAMES, WEAK
)

# BLStats field indices (from NLE)
BLStats = namedtuple('BLStats',
    'x y str_pct str dex con int wis cha score '
    'hp max_hp depth gold energy max_energy ac monster_level '
    'xl xp time hunger carrying_capacity dungeon_number level_number prop_mask')

# Glyph constants
GLYPH_MON_OFF = 0
GLYPH_PET_OFF = 381
GLYPH_BODY_OFF = 1144
GLYPH_OBJ_OFF = 1906
GLYPH_CMAP_OFF = 2359
NUMMONS = 381
MAP_H, MAP_W = 21, 79

# CMAP indices
_WALKABLE_CMAPS = frozenset({12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
_CLOSED_DOOR_CMAPS = frozenset({15, 16})
_WALL_CMAPS = frozenset(range(1, 12))
_DOOR_CMAPS = frozenset({12, 13, 14, 15, 16})
_BOULDER_GLYPH = GLYPH_OBJ_OFF + 447


class AgentFinished(Exception):
    """Raised when the episode ends."""
    pass


class AgentV2:
    """Expert system agent based on AutoAscend's architecture."""

    def __init__(self, env, verbose=False, seed=None):
        self.env = env
        self.verbose = verbose
        self.seed = seed
        self.actions = list(nh.ACTIONS)

        # Action lookup
        self._act_by_name = {}
        self._act_by_val = {}
        for i, a in enumerate(self.actions):
            name = a.name if hasattr(a, 'name') else str(a)
            if name not in self._act_by_name:
                self._act_by_name[name] = i
            val = int(a)
            if val not in self._act_by_val:
                self._act_by_val[val] = i

        # Subsystems
        self.food = FoodManager()
        self.equip = EquipmentManager()
        self.strategy = StrategyManager()

        # State
        self.obs = None
        self.blstats = None
        self.glyphs = None
        self.message = ''
        self.score = 0.0
        self.step_count = 0
        self._last_turn = -1
        self._inactivity = 0

        # Per-level state
        self.seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self.search_count = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self.door_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)

        # Tracking
        self.resistances = {"cold resistance"}
        self.has_excalibur = False
        self._on_elbereth = False
        self._elbereth_cooldown = 0
        self._eat_cooldown = 0
        self._prev_dlevel = 1
        self.inventory = {}
        self.inv_oclasses = {}

    # ==========================================================
    # Core: step / update (AutoAscend pattern)
    # ==========================================================

    def step(self, action, response_iter=None):
        """Send action to env, handle prompts recursively."""
        if isinstance(action, str):
            action = self._act_by_val[ord(action)]
        elif isinstance(action, int) and action >= 1000:
            # Raw action index
            action = action - 1000
        else:
            # NLE action object
            action = self._act_by_val.get(int(action), action)

        obs, reward, done, truncated, info = self.env.step(action)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        self.score += reward
        self.step_count += 1

        if done or truncated:
            raise AgentFinished()
        if self.step_count > 15000:
            raise AgentFinished()

        self._update(response_iter)

    _update_depth = 0

    def _update(self, response_iter=None):
        """Handle prompts, then update game state."""
        self._update_depth += 1
        if self._update_depth > 50:
            self._update_depth = 0
            self._update_state()
            return

        obs = self.obs
        misc = obs.get('misc', [0, 0, 0])
        msg = bytes(obs['message']).rstrip(b'\x00').decode('latin-1', errors='replace').strip()
        self.message = msg

        # If we have a response iterator (multi-step action), use it
        if response_iter is not None:
            try:
                next_action = next(response_iter)
                self.step(next_action, response_iter)
                return
            except StopIteration:
                pass

        # Handle prompts (AutoAscend update() pattern)
        # xwaitingforspace
        if misc[0] and not misc[1] and not misc[2]:
            self.step(A.TextCharacters.SPACE)
            return

        # --More--
        if '--More--' in msg:
            self.step(A.TextCharacters.SPACE)
            return

        # Text entry (getlin)
        if misc[1]:
            self.step(A.Command.ESC)
            return

        # yn prompt
        if misc[2]:
            if 'Really attack' in msg:
                self.step('n')
                return
            if '[yn]' in msg or '[ynq]' in msg:
                # Default: answer yes
                self.step('y')
                return

        # No prompts: update game state
        self._update_depth = 0
        self._update_state()

    def _update_state(self):
        """Parse observation into game state."""
        obs = self.obs
        bl = obs['blstats']
        self.blstats = BLStats(*bl[:26]) if len(bl) >= 26 else None
        self.glyphs = obs['glyphs']

        if self.blstats is None:
            return

        # Track inactivity
        if self.blstats.time == self._last_turn:
            self._inactivity += 1
            if self._inactivity > 200:
                raise RuntimeError("Stuck: 200 steps without turn advance")
        else:
            self._inactivity = 0
            self._last_turn = self.blstats.time

        # Level change
        if self.blstats.depth != self._prev_dlevel:
            self._prev_dlevel = self.blstats.depth
            self.seen = np.zeros((MAP_H, MAP_W), dtype=bool)
            self.walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
            self.objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
            self.search_count = np.zeros((MAP_H, MAP_W), dtype=np.int32)
            self.door_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)
            self.food.on_level_change()

        # Update maps
        self._update_maps()

        # Parse inventory
        self._parse_inventory()

        # Parse messages
        self._parse_messages()

        # Cooldowns
        if self._elbereth_cooldown > 0:
            self._elbereth_cooldown -= 1
        if self._eat_cooldown > 0:
            self._eat_cooldown -= 1

    # ==========================================================
    # Map tracking
    # ==========================================================

    def _update_maps(self):
        """Update seen/walkable/objects from glyphs."""
        glyphs = self.glyphs
        py, px = self.blstats.y, self.blstats.x

        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(glyphs[r, c])
                cm = g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1

                if cm in _WALKABLE_CMAPS:
                    self.seen[r, c] = True
                    self.walkable[r, c] = True
                    self.objects[r, c] = g
                elif cm in _WALL_CMAPS:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif cm in _CLOSED_DOOR_CMAPS:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif g == _BOULDER_GLYPH:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF:
                    self.seen[r, c] = True
                    if self.objects[r, c] == -1:
                        self.walkable[r, c] = True
                elif cm == 0:  # stone
                    pass  # unseen

        # Mark adjacent stone as seen
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nr, nc = py + dy, px + dx
                if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                    g = int(glyphs[nr, nc])
                    if g - GLYPH_CMAP_OFF == 0:
                        self.seen[nr, nc] = True
                        self.walkable[nr, nc] = False

    def _parse_inventory(self):
        """Parse inventory from observation."""
        inv_strs = self.obs.get('inv_strs')
        inv_letters = self.obs.get('inv_letters')
        inv_oclasses = self.obs.get('inv_oclasses')
        if inv_strs is None or inv_letters is None:
            return

        self.inventory = {}
        self.inv_oclasses = {}
        for i, letter_val in enumerate(inv_letters):
            letter = int(letter_val)
            if letter == 0:
                continue
            letter_chr = chr(letter)
            raw = inv_strs[i]
            try:
                item_str = bytes(np.asarray(raw, dtype=np.uint8)).decode(
                    'ascii', errors='replace').rstrip('\x00').strip()
            except Exception:
                item_str = ''
            if item_str:
                self.inventory[letter_chr] = item_str
                if inv_oclasses is not None:
                    self.inv_oclasses[letter_chr] = int(inv_oclasses[i])

        # Check for Excalibur
        for item_str in self.inventory.values():
            if 'Excalibur' in item_str:
                self.has_excalibur = True

    def _parse_messages(self):
        """Parse messages for game events."""
        msg = self.message.lower()

        # Resistance gains
        resist_map = {
            "you feel especially healthy": "poison resistance",
            "you feel a momentary chill": "cold resistance",
            "you feel warm": "fire resistance",
            "you feel full of energy": "shock resistance",
            "you feel very firm": "disintegration resistance",
            "you feel wide awake": "sleep resistance",
        }
        for fragment, resist in resist_map.items():
            if fragment in msg:
                self.resistances.add(resist)

        # Kill tracking
        for prefix in ["you kill the ", "you kill ", "you destroy the ", "you destroy "]:
            if prefix in msg:
                name = msg.split(prefix, 1)[1].split("!")[0].split(".")[0].strip()
                py, px = self.blstats.y, self.blstats.x
                # Record corpse at approximate kill position
                # (kills happen adjacent, but we don't track direction here)
                self.food.on_kill(name, py, px, self.blstats.time, self.resistances)
                break

        # Excalibur
        if "your sword has a bright" in msg or "excalibur" in msg:
            self.has_excalibur = True

    # ==========================================================
    # Navigation helpers
    # ==========================================================

    def bfs(self):
        """BFS from player position. Returns distance array."""
        from collections import deque
        py, px = self.blstats.y, self.blstats.x
        dis = np.full((MAP_H, MAP_W), -1, dtype=np.int32)
        dis[py, px] = 0
        queue = deque([(py, px)])
        while queue:
            y, x = queue.popleft()
            d = dis[y, x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < MAP_H and 0 <= nx < MAP_W and dis[ny, nx] == -1:
                        if not self.walkable[ny, nx]:
                            # Allow through closed doors for pathfinding
                            g = int(self.glyphs[ny, nx])
                            cm = g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1
                            if cm not in _CLOSED_DOOR_CMAPS:
                                continue
                        # No diagonal through doors
                        if abs(dy) + abs(dx) > 1:
                            g_src = int(self.glyphs[y, x])
                            g_dst = int(self.glyphs[ny, nx])
                            cm_src = g_src - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g_src < GLYPH_CMAP_OFF + 87 else -1
                            cm_dst = g_dst - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g_dst < GLYPH_CMAP_OFF + 87 else -1
                            if cm_src in _DOOR_CMAPS or cm_dst in _DOOR_CMAPS:
                                continue
                        dis[ny, nx] = d + 1
                        queue.append((ny, nx))
        return dis

    def go_to(self, ty, tx, dis=None):
        """Navigate to (ty, tx) using BFS. Takes multiple steps."""
        if dis is None:
            dis = self.bfs()
        py, px = self.blstats.y, self.blstats.x
        if dis[ty, tx] == -1:
            return False

        # Take one step toward target using BFS gradient
        best_ny, best_nx = -1, -1
        best_d = 999999
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                if ddy == 0 and ddx == 0:
                    continue
                ny, nx = py + ddy, px + ddx
                if 0 <= ny < MAP_H and 0 <= nx < MAP_W and dis[ny, nx] != -1:
                    if dis[ny, nx] < dis[py, px]:
                        target_d = abs(ny - ty) + abs(nx - tx)
                        if target_d < best_d:
                            best_d = target_d
                            best_ny, best_nx = ny, nx

        if best_ny == -1:
            return False

        self._move_direction(best_ny - py, best_nx - px)
        return True

    def get_visible_monsters(self):
        """Get list of visible monsters from glyphs."""
        monsters = []
        glyphs = self.glyphs
        py, px = self.blstats.y, self.blstats.x
        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(glyphs[r, c])
                if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                    mon_id = g - GLYPH_MON_OFF
                    name = nh.permonst(mon_id).mname if mon_id < NUMMONS else f"mon_{mon_id}"
                    dist = max(abs(r - py), abs(c - px))
                    monsters.append((dist, r, c, name, mon_id))
                elif GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS:
                    pass  # pet, skip
        return sorted(monsters)

    # ==========================================================
    # High-level strategies
    # ==========================================================

    def fight(self):
        """Fight visible monsters. AutoAscend's fight2()."""
        monsters = self.get_visible_monsters()
        # Filter peacefuls
        monsters = [(d, r, c, name, mid) for d, r, c, name, mid in monsters
                     if name not in PEACEFUL_NAMES and mid not in PEACEFUL_IDS]
        if not monsters:
            return False

        py, px = self.blstats.y, self.blstats.x

        # Check for instakill adjacent
        for d, r, c, name, mid in monsters:
            if d <= 1 and name in INSTAKILL:
                if not self._on_elbereth and self._elbereth_cooldown == 0:
                    self.engrave_elbereth()
                    return True
                self.step(A.MiscDirection.WAIT)
                return True

        # Check for never-melee adjacent
        for d, r, c, name, mid in monsters:
            if d <= 1 and name in NEVER_MELEE:
                # Flee
                dy = py - r
                dx = px - c
                self._move_direction(dy, dx)
                return True

        # Elbereth check
        adj = [(d, r, c, name, mid) for d, r, c, name, mid in monsters if d <= 1]
        if adj and self.blstats.hp < 30 and not self._on_elbereth and self._elbereth_cooldown == 0:
            # Calculate elbereth priority inline (fight module expects different format)
            adj_weight = 0.0
            for d, r, c, name, mid in adj:
                info = assess_monster(name, mid)
                if info["peaceful"]:
                    continue
                hp_mult = min(20.0 / max(self.blstats.hp, 1), 2.0)
                w = 0.2 if info["weak"] else (3.0 if info["danger"] >= 7 else 1.0)
                adj_weight += w * hp_mult
            hp_ratio = (self.blstats.hp / max(self.blstats.max_hp, 1)) ** 0.5
            if -5 + 20 * adj_weight * (1 - hp_ratio) > 0:
                self.engrave_elbereth()
                return True

        # On Elbereth: wait if HP low, otherwise move off
        if self._on_elbereth:
            if self.blstats.hp < self.blstats.max_hp * 0.8:
                self.step(A.Command.SEARCH)
                return True
            self._on_elbereth = False

        # Melee adjacent target
        if adj:
            # Pick weakest non-peaceful, non-never-melee
            targets = [(d, r, c, name, mid) for d, r, c, name, mid in adj
                        if name not in NEVER_MELEE and name not in PEACEFUL_NAMES
                        and mid not in PEACEFUL_IDS]
            if targets:
                _, tr, tc, tname, _ = targets[0]
                dy = tr - py
                dx = tc - px
                # Wield best weapon before fighting
                wep = self.equip.find_best_weapon(self.inventory)
                if wep:
                    self.step(A.Command.WIELD, iter(wep))
                    return True
                prev = self.step_count
                self._move_direction(dy, dx)
                if self.step_count > prev:
                    # After kill: step onto corpse and eat
                    if 'kill' in self.message.lower() or 'destroy' in self.message.lower():
                        self._try_eat_corpse(tname, tr, tc)
                    return True

        # Approach nearest if within 7 tiles
        if monsters and monsters[0][0] <= 7 and self.blstats.hp > self.blstats.max_hp * 0.3:
            d, tr, tc, name, mid = monsters[0]
            dis = self.bfs()
            if dis[tr, tc] != -1:
                if self.go_to(tr, tc, dis):
                    return True

        return False

    def eat_corpses(self):
        """Eat safe corpses on the ground."""
        # Check if there's a corpse message
        msg = self.message.lower()
        if 'corpse' in msg and ('you see here' in msg or 'there is' in msg):
            # Extract name
            for pattern in ['you see here a ', 'you see here an ', 'there is a ', 'there is an ']:
                if pattern in msg and 'corpse' in msg:
                    after = msg.split(pattern, 1)[1]
                    if 'corpse' in after:
                        name = after.split(' corpse')[0].strip()
                        if self.food.is_corpse_safe(name, self.resistances) and name != 'floating eye':
                            self.step(A.Command.EAT)
                            return True
        return False

    def eat_from_inventory(self):
        """Eat food from inventory when hungry."""
        hunger = self.blstats.hunger
        if hunger < 2:  # not hungry
            return False

        FOOD_CLASS = 7
        for letter, item_str in self.inventory.items():
            if self.inv_oclasses.get(letter) != FOOD_CLASS:
                continue
            lower = item_str.lower()
            if 'cursed' in lower:
                continue
            # Eat it
            self.step(A.Command.EAT, iter(letter))
            return True
        return False

    def explore(self):
        """Explore current level: find frontier tiles, doors, stairs."""
        py, px = self.blstats.y, self.blstats.x
        dis = self.bfs()

        # Adjacent closed door: walk into it
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = py + dy, px + dx
            if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                g = int(self.glyphs[nr, nc])
                cm = g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1
                if cm in _CLOSED_DOOR_CMAPS and self.door_attempts[nr, nc] < 5:
                    self.door_attempts[nr, nc] += 1
                    self._move_direction(dy, dx)
                    # If locked, kick
                    if 'locked' in self.message.lower():
                        self.step(A.Command.KICK)
                        self._move_direction(dy, dx)  # direction for kick
                    return

        # Find frontier (walkable tile adjacent to unseen)
        best_frontier = None
        best_d = 999999
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or d >= best_d or not self.walkable[r, c]:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not self.seen[nr, nc]:
                            best_d = d
                            best_frontier = (r, c)
                            break
                    if best_frontier and dis[best_frontier[0], best_frontier[1]] == best_d:
                        break

        if best_frontier:
            if self.go_to(best_frontier[0], best_frontier[1], dis):
                return
            # go_to failed, fall through to search

        # Find stairs down
        for r in range(MAP_H):
            for c in range(MAP_W):
                g = int(self.glyphs[r, c])
                cm = g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1
                if cm in (24, 26):  # dnstair, dnladder
                    total_s = int(self.search_count.sum())
                    if self.strategy.should_descend(
                        self.blstats.depth, self.blstats.xl,
                        self.blstats.hp, self.blstats.max_hp,
                        True, total_s, 0) or \
                       self.strategy.should_force_descend(
                        self.blstats.depth, self.blstats.xl,
                        self.blstats.hp, self.blstats.max_hp, total_s):
                        if self.go_to(r, c, dis):
                            if (self.blstats.y, self.blstats.x) == (r, c):
                                self.step(A.MiscDirection.DOWN)
                            return
                        # go_to failed, fall through

        # Search (always advances a turn)
        self.search_count[self.blstats.y, self.blstats.x] += 1
        search_idx = self._act_by_name.get('SEARCH', 75)
        obs, r, done, trunc, info = self.env.step(search_idx)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        self.score += r
        self.step_count += 1
        if done or trunc or self.step_count > 15000:
            raise AgentFinished()
        self._update_state()

    def emergency(self):
        """Handle emergencies: low HP, starvation."""
        bl = self.blstats
        # HP critical
        if bl.hp <= max(5, bl.max_hp // 3):
            # Try prayer
            if bl.time >= 300:  # rough prayer safety
                self.step(A.Command.PRAY)
                return True
            # Quaff healing potion
            POTION_CLASS = 8
            for letter, item_str in self.inventory.items():
                if self.inv_oclasses.get(letter) != POTION_CLASS:
                    continue
                if any(h in item_str.lower() for h in ['healing', 'extra healing', 'full healing']):
                    self.step(A.Command.QUAFF, iter(letter))
                    return True

        # Starvation
        if bl.hunger >= 4:  # fainting
            if bl.time >= 300:
                self.step(A.Command.PRAY)
                return True
            self.step(A.MiscDirection.WAIT)
            return True

        if bl.hunger >= 2:  # hungry
            if self.eat_from_inventory():
                return True

        return False

    def dip_for_excalibur(self):
        """Dip long sword in fountain for Excalibur."""
        if self.has_excalibur or self.blstats.xl < 7:
            return False

        # Check if on fountain
        py, px = self.blstats.y, self.blstats.x
        g = int(self.objects[py, px])
        cm = g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1
        if cm != 31:  # not fountain
            return False

        # Find long sword
        for letter, item_str in self.inventory.items():
            if 'long sword' in item_str.lower() and 'cursed' not in item_str.lower():
                self.step(A.Command.DIP, iter(letter))
                return True

        return False

    def rest(self):
        """Rest to recover HP."""
        bl = self.blstats
        if bl.hp >= bl.max_hp * 0.6:
            return False
        if bl.hunger >= 2:  # don't rest while hungry
            return False
        # Write Elbereth before resting
        if not self._on_elbereth and self._elbereth_cooldown == 0:
            self.engrave_elbereth()
        self.step(A.Command.SEARCH)
        return True

    # ==========================================================
    # Multi-step actions
    # ==========================================================

    def engrave_elbereth(self):
        """Write Elbereth in the dust."""
        def gen():
            yield '-'  # write with fingers
            # Handle prompts in update()
            yield from 'Elbereth'
            yield '\r'

        self.step(A.Command.ENGRAVE, gen())
        self._on_elbereth = True

    def _try_eat_corpse(self, name, row, col):
        """Try to eat a corpse after killing."""
        if self._eat_cooldown > 0:
            return
        if self.blstats.hunger == 0:  # satiated
            return
        if not self.food.is_corpse_safe(name, self.resistances):
            return
        if name == 'floating eye':
            return

        # Step onto corpse tile and eat
        py, px = self.blstats.y, self.blstats.x
        if (py, px) != (row, col):
            dy = row - py
            dx = col - px
            if abs(dy) <= 1 and abs(dx) <= 1:
                self._move_direction(dy, dx)

        self.step(A.Command.EAT)

    def _do_search(self):
        """Guaranteed step: direct env.step(SEARCH)."""
        search_idx = self._act_by_name.get('SEARCH', 75)
        obs, r, done, trunc, info = self.env.step(search_idx)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        self.score += r
        self.step_count += 1
        if done or trunc or self.step_count > 15000:
            raise AgentFinished()
        self._update_state()

    def _move_direction(self, dy, dx):
        """Move in a direction."""
        direction_map = {
            (-1, 0): 'N', (1, 0): 'S', (0, 1): 'E', (0, -1): 'W',
            (-1, 1): 'NE', (1, 1): 'SE', (1, -1): 'SW', (-1, -1): 'NW',
        }
        name = direction_map.get((dy, dx))
        if name:
            idx = self._act_by_name.get(name)
            if idx is not None:
                self.step(1000 + idx)

    # ==========================================================
    # Main loop
    # ==========================================================

    def main(self):
        """Main game loop. Runs until episode ends."""
        try:
            # Initial setup: reset and clear any initial prompts
            obs, info = self.env.reset(seed=self.seed)
            self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            self._update_state()  # just parse state, no prompt handling
            print(f"  INIT: steps={self.step_count} walkable={int(self.walkable.sum())} seen={int(self.seen.sum())}")

            # Main loop
            while True:
                if self.step_count > 15000:
                    raise AgentFinished()
                try:
                    total_s = int(self.search_count.sum())
                    self.strategy.update(
                        self.blstats.depth, self.blstats.xl,
                        self.blstats.hp, self.blstats.max_hp,
                        self.has_excalibur, False, total_s, True)

                    prev = self.step_count
                    if self.emergency():
                        if self.step_count == prev:
                            self._do_search()
                        continue
                    if self.fight():
                        if self.step_count == prev:
                            self._do_search()
                        continue
                    if self.eat_corpses():
                        if self.step_count == prev:
                            self._do_search()
                        continue
                    if self.dip_for_excalibur():
                        if self.step_count == prev:
                            self._do_search()
                        continue
                    if self.rest():
                        if self.step_count == prev:
                            self._do_search()
                        continue
                    self.explore()

                except RuntimeError as e:
                    if 'Stuck' in str(e):
                        self._inactivity = 0
                        search_idx = self._act_by_name.get('SEARCH', 75)
                        obs, r, done, trunc, info = self.env.step(search_idx)
                        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
                        self.score += r
                        self.step_count += 1
                        if done or trunc:
                            raise AgentFinished()
                        self._update_state()
                    else:
                        raise

        except AgentFinished:
            pass

        return self.score
