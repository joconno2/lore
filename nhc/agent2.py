"""LORE Agent v2: Clean room rewrite of AutoAscend architecture.

The agent DRIVES the env. step() calls env.step() and recursively
handles all prompts via update(). The main loop runs strategies
in priority order. Each strategy calls step() which handles the
full NLE interaction.

This is a faithful reimplementation of AutoAscend's core loop
(agent.py:365-430, 1517-1567) adapted for our modules.
"""
from __future__ import annotations

import sys
sys.setrecursionlimit(10000)

import numpy as np
from collections import namedtuple, deque
from typing import Optional

import nle.nethack as nh
from nle.nethack import actions as A

from nhc.food import FoodManager
from nhc.equipment import EquipmentManager
from nhc.fight import assess_monster, NEVER_MELEE, INSTAKILL, PEACEFUL_NAMES, PEACEFUL_IDS

BLStats = namedtuple('BLStats',
    'x y str_pct str dex con int wis cha score '
    'hp max_hp depth gold energy max_energy ac monster_level '
    'xl xp time hunger carrying_capacity dungeon_number level_number prop_mask')

GLYPH_MON_OFF = 0
GLYPH_PET_OFF = 381
GLYPH_BODY_OFF = 1144
GLYPH_OBJ_OFF = 1906
GLYPH_CMAP_OFF = 2359
NUMMONS = 381
MAP_H, MAP_W = 21, 79

_WALKABLE = frozenset({12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31})
_CLOSED_DOOR = frozenset({15, 16})
_WALL = frozenset(range(1, 12))
_DOOR = frozenset({12, 13, 14, 15, 16})


class AgentFinished(Exception):
    pass


def _cmap(g):
    return g - GLYPH_CMAP_OFF if GLYPH_CMAP_OFF <= g < GLYPH_CMAP_OFF + 87 else -1


class AgentV2:
    def __init__(self, env, seed=None, verbose=False):
        self.env = env
        self.seed = seed
        self.verbose = verbose

        # Build action lookups
        self.actions = list(nh.ACTIONS)
        self._val2idx = {}
        self._name2idx = {}
        for i, a in enumerate(self.actions):
            v = int(a)
            if v not in self._val2idx:
                self._val2idx[v] = i
            n = a.name if hasattr(a, 'name') else str(a)
            if n not in self._name2idx:
                self._name2idx[n] = i

        # Subsystems
        self.food = FoodManager()
        self.equip = EquipmentManager()

        # State
        self.obs = None
        self.blstats = None
        self.glyphs = None
        self.message = ''
        self.score = 0.0
        self.step_count = 0
        self._last_turn = -1

        # Per-level maps
        self.seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self.search_count = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self.door_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)

        # Tracking
        self.resistances = {"cold resistance"}
        self.has_excalibur = False
        self._prev_depth = 1
        self._raw_bl = None
        self.inventory = {}
        self.inv_oclasses = {}

    # ================================================================
    # Core: step / update (AutoAscend agent.py:365-430)
    # ================================================================

    def _env_step(self, idx):
        """Raw env.step with blstats copy."""
        obs, reward, done, truncated, info = self.env.step(idx)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        # Read blstats from COPIED obs - only update if game has started (time > 0)
        bl = self.obs.get('blstats')
        if bl is not None and int(bl[20]) > 0:
            self._raw_bl = np.array(bl, dtype=np.int64)
        self.score += reward
        self.step_count += 1
        return done or truncated

    def step(self, action, gen=None):
        """Send action to env, handle prompts iteratively.

        action: NLE action object, string char, or int action value
        gen: optional generator yielding responses for multi-step actions
        """
        if isinstance(action, str):
            assert len(action) == 1
            idx = self._val2idx.get(ord(action))
        elif isinstance(action, int) and action < len(self.actions):
            idx = action
        else:
            idx = self._val2idx.get(int(action))

        if idx is None:
            return

        if self._env_step(idx):
            self._parse_blstats()
            raise AgentFinished()

        # Save initial message before prompt handling clears it
        raw_msg = self.obs.get('message', b'')
        self.initial_message = bytes(raw_msg).decode('latin-1', errors='replace').replace('\x00', '').strip()

        # Handle prompts iteratively (replaces recursive _update)
        prompt_count = 0
        for _ in range(200):  # safety limit
            prompt_count += 1
            msg_raw = self.obs.get('message', b'')
            self.message = bytes(msg_raw).decode('latin-1', errors='replace').replace('\x00', '').strip()
            misc = self.obs.get('misc', [0, 0, 0])

            # Multi-step action generator
            if gen is not None:
                try:
                    next_action = next(gen)
                    if isinstance(next_action, str):
                        next_idx = self._val2idx.get(ord(next_action))
                    elif isinstance(next_action, int) and next_action < len(self.actions):
                        next_idx = next_action
                    else:
                        next_idx = self._val2idx.get(int(next_action))
                    if next_idx is not None:
                        if self._env_step(next_idx):
                            self._parse_blstats()
                            raise AgentFinished()
                        continue
                except StopIteration:
                    gen = None

            # xwaitingforspace
            if misc[0]:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # --More--
            if '--More--' in self.message:
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # Text entry: ESC
            if misc[1]:
                if self._env_step(self._val2idx.get(27, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # yn prompt
            if misc[2]:
                if 'Really attack' in self.message:
                    resp = self._val2idx.get(ord('n'))
                else:
                    resp = self._val2idx.get(ord('y'))
                if resp is not None:
                    obs, reward, done, truncated, info = self.env.step(resp)
                    self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
                    self.score += reward
                    self.step_count += 1
                    if done or truncated:
                        self._parse_blstats()
                        raise AgentFinished()
                continue

            # No prompts: done
            break

        if prompt_count > 5:
            print(f"  PROMPT LOOP: {prompt_count} iterations, msg={self.message[:50]}")
        self._update_game_state()

    _ugs_count = 0

    def _update_game_state(self):
        """Parse observation into full game state."""
        self._ugs_count += 1
        self._parse_blstats()
        if self.blstats is None:
            return

        # Level change
        if self.blstats.depth != self._prev_depth:
            self._prev_depth = self.blstats.depth
            self.seen[:] = False
            self.walkable[:] = False
            self.objects[:] = -1
            self.search_count[:] = 0
            self.door_attempts[:] = 0
            self.food.on_level_change()

        self.glyphs = self.obs['glyphs']
        self._update_maps()
        self._parse_inventory()
        self._parse_messages()

    def _parse_blstats(self):
        bl = self._raw_bl
        if bl is not None and len(bl) >= 26:
            self.blstats = BLStats(*[int(v) for v in bl[:26]])

    def _update_maps(self):
        g = self.glyphs
        py, px = self.blstats.y, self.blstats.x
        for r in range(MAP_H):
            for c in range(MAP_W):
                v = int(g[r, c])
                cm = _cmap(v)
                if cm in _WALKABLE:
                    self.seen[r, c] = True
                    self.walkable[r, c] = True
                    self.objects[r, c] = v
                elif cm in _WALL:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif cm in _CLOSED_DOOR:
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif v == GLYPH_OBJ_OFF + 447:  # boulder
                    self.seen[r, c] = True
                    self.walkable[r, c] = False
                elif GLYPH_MON_OFF <= v < GLYPH_CMAP_OFF:
                    self.seen[r, c] = True
                    if self.objects[r, c] == -1:
                        self.walkable[r, c] = True
                elif cm == 0:
                    if abs(r - py) <= 1 and abs(c - px) <= 1:
                        self.seen[r, c] = True
                        self.walkable[r, c] = False
        self.walkable[py, px] = True
        self.seen[py, px] = True

    def _parse_inventory(self):
        inv_strs = self.obs.get('inv_strs')
        inv_letters = self.obs.get('inv_letters')
        inv_oclasses = self.obs.get('inv_oclasses')
        if inv_strs is None or inv_letters is None:
            return
        self.inventory = {}
        self.inv_oclasses = {}
        for i, lv in enumerate(inv_letters):
            letter = int(lv)
            if letter == 0:
                continue
            ch = chr(letter)
            raw = inv_strs[i]
            try:
                s = bytes(np.asarray(raw, dtype=np.uint8)).decode('ascii', errors='replace').rstrip('\x00').strip()
            except Exception:
                s = ''
            if s:
                self.inventory[ch] = s
                if inv_oclasses is not None:
                    self.inv_oclasses[ch] = int(inv_oclasses[i])
        if any('Excalibur' in s for s in self.inventory.values()):
            self.has_excalibur = True

    def _parse_messages(self):
        msg = self.message.lower()
        resists = {
            "you feel especially healthy": "poison resistance",
            "you feel a momentary chill": "cold resistance",
            "you feel warm": "fire resistance",
            "you feel full of energy": "shock resistance",
        }
        for frag, r in resists.items():
            if frag in msg:
                self.resistances.add(r)
        if "your sword has a bright" in msg or "excalibur" in msg:
            self.has_excalibur = True
        # Kill tracking
        for prefix in ["you kill the ", "you kill ", "you destroy the ", "you destroy "]:
            if prefix in msg:
                name = msg.split(prefix, 1)[1].split("!")[0].split(".")[0].strip()
                py, px = self.blstats.y, self.blstats.x
                self.food.on_kill(name, py, px, self.blstats.time, self.resistances)
                break

    # ================================================================
    # Navigation
    # ================================================================

    def bfs(self):
        py, px = self.blstats.y, self.blstats.x
        dis = np.full((MAP_H, MAP_W), -1, dtype=np.int32)
        dis[py, px] = 0
        q = deque([(py, px)])
        while q:
            y, x = q.popleft()
            d = dis[y, x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < MAP_H and 0 <= nx < MAP_W) or dis[ny, nx] != -1:
                        continue
                    g = int(self.glyphs[ny, nx])
                    cm = _cmap(g)
                    ok = self.walkable[ny, nx] or cm in _CLOSED_DOOR or (GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF)
                    if not ok:
                        continue
                    # No diagonal through doors
                    if abs(dy) + abs(dx) > 1:
                        if _cmap(int(self.glyphs[y, x])) in _DOOR or cm in _DOOR:
                            continue
                    dis[ny, nx] = d + 1
                    q.append((ny, nx))
        return dis

    def step_toward(self, ty, tx, dis):
        """Take one BFS-optimal step toward (ty, tx). Returns True if stepped."""
        if dis[ty, tx] == -1:
            return False
        py, px = self.blstats.y, self.blstats.x
        # Trace back from target
        path = []
        cy, cx = ty, tx
        for _ in range(300):
            if (cy, cx) == (py, px):
                break
            path.append((cy, cx))
            best = None
            bd = dis[cy, cx]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < MAP_H and 0 <= nx < MAP_W and dis[ny, nx] != -1 and dis[ny, nx] < bd:
                        bd = dis[ny, nx]
                        best = (ny, nx)
            if best is None:
                return False
            cy, cx = best
        if not path:
            return False
        # First step is last element of reversed path
        ny, nx = path[-1]
        dy, dx = ny - py, nx - px
        py, px = self.blstats.y, self.blstats.x
        # Don't step into boulders or walls
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            if g == GLYPH_OBJ_OFF + 447:  # boulder
                return False
            cm = _cmap(g)
            if cm in _WALL or cm == 0:  # wall or stone
                return False

        # Check for diagonal through door (not allowed in NetHack)
        if abs(dy) + abs(dx) > 1:  # diagonal
            src_cm = _cmap(int(self.glyphs[py, px]))
            dst_cm = _cmap(int(self.glyphs[ny, nx]))
            src_obj = _cmap(int(self.objects[py, px])) if self.objects[py, px] != -1 else -1
            dst_obj = _cmap(int(self.objects[ny, nx])) if self.objects[ny, nx] != -1 else -1
            if src_cm in _DOOR or dst_cm in _DOOR or src_obj in _DOOR or dst_obj in _DOOR:
                # Use cardinal step instead
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W and (self.walkable[cr, cc] or _cmap(int(self.glyphs[cr, cc])) in _CLOSED_DOOR):
                        dy, dx = cdy, cdx
                        ny, nx = cr, cc
                        break

        # Final validation: don't step into walls/boulders
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            cm = _cmap(g)
            if cm in _WALL or cm == 0 or g == GLYPH_OBJ_OFF + 447:
                return False

        self._move_dir(dy, dx)
        # If diagonal failed (door), retry with cardinal
        if 'diagonally' in self.message.lower() and abs(dy) + abs(dx) > 1:
            for cdy, cdx in [(dy, 0), (0, dx)]:
                if cdy == 0 and cdx == 0:
                    continue
                self._move_dir(cdy, cdx)
                return True
        return True

    def _move_dir(self, dy, dx):
        """Send a compass direction through step()."""
        dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W',
                (-1,1):'NE',(1,1):'SE',(1,-1):'SW',(-1,-1):'NW'}
        name = dmap.get((dy, dx))
        if name and name in self._name2idx:
            self.step(self._name2idx[name])

    def get_monsters(self):
        """Visible non-pet monsters with distance."""
        py, px = self.blstats.y, self.blstats.x
        mons = []
        for r in range(MAP_H):
            for c in range(MAP_W):
                if r == py and c == px:
                    continue
                g = int(self.glyphs[r, c])
                if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                    mid = g - GLYPH_MON_OFF
                    name = nh.permonst(mid).mname if mid < NUMMONS else f"mon{mid}"
                    d = max(abs(r - py), abs(c - px))
                    mons.append((d, r, c, name, mid))
        return sorted(mons)

    # ================================================================
    # Strategies
    # ================================================================

    def fight(self):
        """Fight or approach visible hostile monsters."""
        mons = [(d,r,c,n,m) for d,r,c,n,m in self.get_monsters()
                if n not in PEACEFUL_NAMES and m not in PEACEFUL_IDS]
        if not mons:
            return False
        py, px = self.blstats.y, self.blstats.x

        # Adjacent monsters: melee them
        adj = [(d,r,c,n,m) for d,r,c,n,m in mons if d <= 1]
        for d, r, c, n, m in adj:
            if n in INSTAKILL:
                dy, dx = py - r, px - c
                self._move_dir(dy, dx)
                return True
            if n in NEVER_MELEE:
                continue
            # Melee
            dy, dx = r - py, c - px
            self._move_dir(dy, dx)
            return True

        # Approach nearest within 15 tiles
        if mons[0][0] <= 15 and self.blstats.hp > self.blstats.max_hp * 0.3:
            dis = self.bfs()
            d, r, c, n, m = mons[0]
            if dis[r, c] != -1:
                if self.step_toward(r, c, dis):
                    return True
        return False

    def emergency(self):
        """Handle HP critical and starvation."""
        bl = self.blstats
        if bl.hp <= max(5, bl.max_hp // 3) and bl.time >= 300:
            self.step(A.Command.PRAY)
            return True
        if bl.hunger >= 4 and bl.time >= 300:  # fainting
            self.step(A.Command.PRAY)
            return True
        if bl.hunger >= 2:  # hungry
            FOOD = 7
            for letter, item in self.inventory.items():
                if self.inv_oclasses.get(letter) == FOOD and 'cursed' not in item.lower():
                    self.step(A.Command.EAT, iter(letter))
                    return True
        return False

    def explore(self):
        """Explore: open doors, go to frontier, search near walls, descend."""
        py, px = self.blstats.y, self.blstats.x
        dis = self.bfs()

        # 1. Adjacent closed doors: open them
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = py+dy, px+dx
            if 0<=nr<MAP_H and 0<=nc<MAP_W:
                g = int(self.glyphs[nr, nc])
                if _cmap(g) in _CLOSED_DOOR and self.door_attempts[nr,nc] < 5:
                    self.door_attempts[nr,nc] += 1
                    dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W'}
                    name = dmap.get((dy,dx))
                    if name and name in self._name2idx:
                        self.step(self._name2idx[name])
                        # If locked, skip this door (kick doesn't work reliably)
                        if 'locked' in self.message.lower():
                            self.door_attempts[nr,nc] = 5  # mark as exhausted
                        return

        # 2. Go to nearest closed door
        best_door, best_dd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if _cmap(int(self.glyphs[r,c])) in _CLOSED_DOOR and self.door_attempts[r,c]<5:
                    d = dis[r,c]
                    if d != -1 and d < best_dd:
                        best_dd = d
                        best_door = (r,c)
        if best_door and best_dd > 1:
            if self.step_toward(best_door[0], best_door[1], dis):
                return

        # 3. Go to nearest frontier
        best_f, best_fd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r,c]
                if d == -1 or d >= best_fd or not self.walkable[r,c]:
                    continue
                for dy2 in (-1,0,1):
                    for dx2 in (-1,0,1):
                        if dy2==0 and dx2==0: continue
                        nr, nc = r+dy2, c+dx2
                        if 0<=nr<MAP_H and 0<=nc<MAP_W and not self.seen[nr,nc]:
                            best_fd = d
                            best_f = (r,c)
                            break
                    if best_f and dis[best_f[0],best_f[1]] == best_fd:
                        break
        if best_f:
            if self.step_toward(best_f[0], best_f[1], dis):
                return

        # 4. Descend: when explored OR after enough time on this level
        total_s = int(self.search_count.sum())
        should_descend = (
            (best_f is None and best_door is None) or  # fully explored
            total_s > 50  # searched enough
        )
        if should_descend and self.blstats.hp > self.blstats.max_hp * 0.5:
            for r in range(MAP_H):
                for c in range(MAP_W):
                    cm = _cmap(int(self.glyphs[r,c]))
                    if cm in (24, 26) and dis[r,c] != -1:  # stairs down
                        if self.step_toward(r, c, dis):
                            if (self.blstats.y, self.blstats.x) == (r, c):
                                self.step(A.MiscDirection.DOWN)
                            return

        # 5. Search near walls (AutoAscend search priority)
        best_s, best_sp = None, -999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if not self.walkable[r,c] or dis[r,c] == -1:
                    continue
                adj = 0
                for dy2 in (-1,0,1):
                    for dx2 in (-1,0,1):
                        if dy2==0 and dx2==0: continue
                        nr, nc = r+dy2, c+dx2
                        if 0<=nr<MAP_H and 0<=nc<MAP_W:
                            cm2 = _cmap(int(self.glyphs[nr,nc]))
                            if cm2 in _WALL or cm2 == 0 or not self.seen[nr,nc]:
                                adj += 1
                if adj == 0:
                    continue
                p = adj * 10 - self.search_count[r,c]**2 * 2 - dis[r,c]
                if p > best_sp:
                    best_sp = p
                    best_s = (r,c)

        if best_s and best_s != (py,px):
            if self.step_toward(best_s[0], best_s[1], dis):
                return

        # 6. Search at current position
        self.search_count[py, px] += 1
        self.step(A.Command.SEARCH)

    # ================================================================
    # Main loop (AutoAscend agent.py:1517-1567)
    # ================================================================

    def main(self):
        try:
            obs, info = self.env.reset(seed=self.seed)
            self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            bl = obs.get('blstats')
            if bl is not None:
                self._raw_bl = bl.copy()
            self._update_game_state()
            # Clear any initial prompts
            try:
                self.step(A.Command.ESC)
                self.step(A.Command.ESC)
            except AgentFinished:
                raise

            while True:
                try:
                    if self.emergency():
                        continue
                    # Eat corpse on ground if initial_message mentions one
                    if not getattr(self, '_eat_cooldown_v2', 0):
                        msg_check = getattr(self, 'initial_message', '') or ''
                        if 'corpse' in msg_check.lower() and \
                           ('you see here' in msg_check.lower() or 'there is' in msg_check.lower()):
                            if self.blstats.hunger != 0:  # not satiated
                                self._eat_cooldown_v2 = 20  # skip next 20 iterations
                                self.step(A.Command.EAT)
                                continue
                    else:
                        self._eat_cooldown_v2 -= 1
                    if self.fight():
                        continue
                    self.explore()
                except RuntimeError:
                    self.step(A.Command.SEARCH)

        except AgentFinished:
            pass
        return self.score
