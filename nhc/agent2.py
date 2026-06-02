"""LORE Agent v2: Clean room rewrite based on AutoAscend architecture.

The agent DRIVES the env. step() calls env.step() then handles prompts
iteratively. The main loop runs strategies in priority order:
emergency > fight > eat > explore.

Based on AutoAscend (agent.py:365-430, 1517-1567, global_logic.py:608-644).
"""
from __future__ import annotations

import numpy as np
from collections import namedtuple, deque

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
_STAIRS_DOWN = frozenset({24, 26})  # dnstair, dnladder
_STAIRS_UP = frozenset({22, 25})    # upstair, upladder
_FOUNTAIN = frozenset({31})

# Hunger states
SATIATED = 0
NOT_HUNGRY = 1
HUNGRY = 2
WEAK = 3
FAINTING = 4

# Object classes
FOOD_CLASS = 7
WEAPON_CLASS = 3
POTION_CLASS = 10


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
        self.initial_message = ''
        self.score = 0.0
        self.step_count = 0
        self._last_turn = -1

        # Per-level maps
        self.seen = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.walkable = np.zeros((MAP_H, MAP_W), dtype=bool)
        self.objects = np.full((MAP_H, MAP_W), -1, dtype=np.int16)
        self.search_count = np.zeros((MAP_H, MAP_W), dtype=np.int32)
        self.door_attempts = np.zeros((MAP_H, MAP_W), dtype=np.int32)

        # Level tracking
        self._prev_depth = 1
        self._raw_bl = None
        self._stairs_down = set()  # (y, x) positions of known downstairs
        self._fountains = set()   # (y, x) positions of known fountains
        self._level_turns = 0   # turns spent on current level

        # Inventory
        self.inventory = {}
        self.inv_oclasses = {}

        # Character state
        self.resistances = {"cold resistance"}
        self.has_excalibur = False
        self._last_prayer_turn = -1000
        self._last_eat_turn = -100
        self._peaceful_positions = set()  # (y, x) of known peacefuls
        self._peaceful_monster_ids = set()  # monster IDs confirmed peaceful
        self._last_move_dir = (0, 0)      # last direction sent via _move_dir

        # Debug
        self._ugs_count = 0
        self._prompt_steps = 0
        self._yn_count = 0
        self._xwait_count = 0
        self._getlin_count = 0
        self._more_count = 0

    # ================================================================
    # Core: step / update
    # ================================================================

    def _env_step(self, idx):
        """Raw env.step with observation copy."""
        obs, reward, done, truncated, info = self.env.step(idx)
        self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
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
        elif type(action) is int and action < len(self.actions):
            # Plain int: treat as action index
            idx = action
        else:
            # NLE action enum or other: look up by value
            idx = self._val2idx.get(int(action))

        if idx is None:
            return

        if self._env_step(idx):
            self._parse_blstats()
            raise AgentFinished()

        # Save initial message before prompt handling
        raw_msg = self.obs.get('message', b'')
        self.initial_message = bytes(raw_msg).decode('latin-1', errors='replace').replace('\x00', '').strip()

        # Handle prompts iteratively
        prompt_count = 0
        for _ in range(200):
            msg_raw = self.obs.get('message', b'')
            self.message = bytes(msg_raw).decode('latin-1', errors='replace').replace('\x00', '').strip()
            misc = self.obs.get('misc', [0, 0, 0])

            # Generator for multi-step actions (eat letter, engrave text, etc.)
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

            # NLE misc mapping (verified empirically):
            # misc[0] = in_yn_function (yn prompt - game wants a single char response)
            # misc[1] = in_getlin (text entry - game wants a string)
            # misc[2] = xwaitforspace (but env handles this, should never reach us)
            #
            # The env's _perform_known_steps handles xwait and getlin internally.
            # Only yn prompts (misc[0]) reach the agent with allow_all_yn_questions=True.

            # yn prompt (misc[0] = in_yn_function)
            if misc[0]:
                self._yn_count += 1
                resp = self._handle_yn_prompt()
                if resp is not None:
                    if self._env_step(resp):
                        self._parse_blstats()
                        raise AgentFinished()
                continue

            # Text entry (misc[1] = in_getlin)
            if misc[1]:
                self._getlin_count += 1
                # Handle eat menu: type food letter
                if 'What do you want to eat' in self.message:
                    food_letter = None
                    for letter, item in self.inventory.items():
                        oc = self.inv_oclasses.get(letter, -1)
                        if oc == FOOD_CLASS and 'cursed' not in item.lower() and 'corpse' not in item.lower():
                            food_letter = letter
                            break
                    if food_letter:
                        resp_idx = self._val2idx.get(ord(food_letter))
                        if resp_idx is not None:
                            if self._env_step(resp_idx):
                                self._parse_blstats()
                                raise AgentFinished()
                            continue
                # Default: ESC out of text entry
                if self._env_step(self._val2idx.get(27, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # xwait (misc[2]) - shouldn't reach us, but handle if it does
            if misc[2]:
                self._xwait_count += 1
                if self._env_step(self._val2idx.get(32, 0)):  # SPACE
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            # --More-- in message text (backup check)
            if '--More--' in self.message:
                self._more_count += 1
                if self._env_step(self._val2idx.get(32, 0)):
                    self._parse_blstats()
                    raise AgentFinished()
                continue

            break

        if prompt_count > 3 and self.verbose:
            print(f"  PROMPT_LOOP: {prompt_count} iterations, msg={self.message[:60]}")
        self._prompt_steps += prompt_count
        self._update_game_state()

    def _handle_yn_prompt(self):
        """Decide response to yn prompt. Returns action index."""
        msg = self.message
        # Don't attack peacefuls - record their position AND monster ID
        if 'Really attack' in msg:
            if self.blstats and self._last_move_dir != (0, 0):
                dy, dx = self._last_move_dir
                py, px = self.blstats.y, self.blstats.x
                ty, tx = py + dy, px + dx
                if 0 <= ty < MAP_H and 0 <= tx < MAP_W:
                    self._peaceful_positions.add((ty, tx))
                    g = int(self.glyphs[ty, tx]) if self.glyphs is not None else 0
                    if GLYPH_MON_OFF <= g < GLYPH_PET_OFF:
                        self._peaceful_monster_ids.add(g - GLYPH_MON_OFF)
            return self._val2idx.get(ord('n'))
        # Don't force locks
        if 'force the lock' in msg:
            return self._val2idx.get(ord('n'))
        # Eat/dip/apply menus: ESC out (handled by two-step approach in eat()/dip())
        if 'What do you want to eat' in msg or 'What do you want to dip' in msg:
            return self._val2idx.get(27)  # ESC
        # Default: yes
        return self._val2idx.get(ord('y'))

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
            self._stairs_down = set()
            self._fountains = set()
            self._level_turns = 0
            self._peaceful_positions = set()
            self.food.on_level_change()

        # Track turns on this level
        if self.blstats.time != self._last_turn:
            self._level_turns += 1
            self._last_turn = self.blstats.time

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
                    # Only store terrain glyphs in objects (not monster/player glyphs)
                    self.objects[r, c] = v
                    # Accumulate stairs and fountains (persist until level change)
                    if cm in _STAIRS_DOWN:
                        self._stairs_down.add((r, c))
                    if cm in _FOUNTAIN:
                        self._fountains.add((r, c))
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
                    # Monster on tile: mark seen/walkable but DON'T overwrite objects
                    # (preserves terrain info like stairs underneath)
                    self.seen[r, c] = True
                    if self.objects[r, c] == -1:
                        self.walkable[r, c] = True
                    # Don't overwrite objects with monster glyph
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
        self._last_kill_name = None
        for prefix in ["you kill the ", "you kill ", "you destroy the ", "you destroy "]:
            if prefix in msg:
                name = msg.split(prefix, 1)[1].split("!")[0].split(".")[0].strip()
                py, px = self.blstats.y, self.blstats.x
                self.food.on_kill(name, py, px, self.blstats.time, self.resistances)
                self._last_kill_name = name
                break

    # ================================================================
    # Navigation
    # ================================================================

    def _is_peaceful_glyph(self, g):
        """Check if a monster glyph represents a peaceful monster."""
        if not (GLYPH_MON_OFF <= g < GLYPH_PET_OFF):
            return False
        mid = g - GLYPH_MON_OFF
        if mid in PEACEFUL_IDS:
            return True
        if mid < NUMMONS:
            name = nh.permonst(mid).mname
            if name in PEACEFUL_NAMES:
                return True
        return False

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
                    # Closed doors: only include if not exhausted (locked/stuck)
                    is_closed_door = cm in _CLOSED_DOOR
                    if is_closed_door and self.door_attempts[ny, nx] >= 5:
                        continue
                    # Pets: can swap places, include them. All other monsters: exclude.
                    is_pet = GLYPH_PET_OFF <= g < GLYPH_BODY_OFF
                    is_wild_monster = GLYPH_MON_OFF <= g < GLYPH_PET_OFF
                    if is_wild_monster:
                        continue  # Don't path through ANY wild monster
                    ok = self.walkable[ny, nx] or is_closed_door or is_pet
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
        # Trace back from target to player
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

        ny, nx = path[-1]
        dy, dx = ny - py, nx - px

        # Validate target
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            if g == GLYPH_OBJ_OFF + 447:  # boulder
                return False
            cm = _cmap(g)
            if cm in _WALL or cm == 0:
                return False

        # Diagonal through door: use cardinal instead
        if abs(dy) + abs(dx) > 1:
            src_cm = _cmap(int(self.glyphs[py, px]))
            dst_cm = _cmap(int(self.glyphs[ny, nx]))
            src_obj = _cmap(int(self.objects[py, px])) if self.objects[py, px] != -1 else -1
            dst_obj = _cmap(int(self.objects[ny, nx])) if self.objects[ny, nx] != -1 else -1
            if src_cm in _DOOR or dst_cm in _DOOR or src_obj in _DOOR or dst_obj in _DOOR:
                for cdy, cdx in [(dy, 0), (0, dx)]:
                    if cdy == 0 and cdx == 0:
                        continue
                    cr, cc = py + cdy, px + cdx
                    if 0 <= cr < MAP_H and 0 <= cc < MAP_W and (self.walkable[cr, cc] or _cmap(int(self.glyphs[cr, cc])) in _CLOSED_DOOR):
                        dy, dx = cdy, cdx
                        ny, nx = cr, cc
                        break

        # Final validation
        if 0 <= ny < MAP_H and 0 <= nx < MAP_W:
            g = int(self.glyphs[ny, nx])
            cm = _cmap(g)
            if cm in _WALL or cm == 0 or g == GLYPH_OBJ_OFF + 447:
                return False

        old_pos = (self.blstats.y, self.blstats.x)
        self._move_dir(dy, dx)
        # Retry if diagonal failed
        if 'diagonally' in self.message.lower() and abs(dy) + abs(dx) > 1:
            for cdy, cdx in [(dy, 0), (0, dx)]:
                if cdy == 0 and cdx == 0:
                    continue
                self._move_dir(cdy, cdx)
                break
        # Check if we actually moved (or at least attempted valid action)
        new_pos = (self.blstats.y, self.blstats.x) if self.blstats else old_pos
        return new_pos != old_pos

    def _move_dir(self, dy, dx):
        """Send a compass direction."""
        self._last_move_dir = (dy, dx)
        dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W',
                (-1,1):'NE',(1,1):'SE',(1,-1):'SW',(-1,-1):'NW'}
        name = dmap.get((dy, dx))
        if name and name in self._name2idx:
            self.step(self._name2idx[name])

    def _kick_dir(self, dy, dx):
        """Kick in a direction (for locked doors)."""
        dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W'}
        name = dmap.get((dy, dx))
        if name and name in self._name2idx:
            # KICK sends the action, then the direction is the next action
            self.step(A.Command.KICK)
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

    def emergency(self):
        """Handle HP critical and starvation. Matches AutoAscend thresholds."""
        bl = self.blstats
        if bl is None:
            return False

        # Prayer safety: 300 turns between prayers (AutoAscend uses 400-500)
        can_pray = (bl.time - self._last_prayer_turn) >= 300

        # HP critical: pray at HP < max/3 or HP < 8
        if can_pray and (bl.hp < max(8, bl.max_hp // 3)):
            self._last_prayer_turn = bl.time
            self.step(A.Command.PRAY)
            return True

        # Fainting from hunger: pray
        if can_pray and bl.hunger >= FAINTING:
            self._last_prayer_turn = bl.time
            self.step(A.Command.PRAY)
            return True

        return False

    def eat_ground(self):
        """Eat corpse/food from ground if standing on one."""
        bl = self.blstats
        if bl is None or bl.hunger < HUNGRY:
            return False
        if bl.time - self._last_eat_turn < 3:
            return False
        # Check if we just stepped on food (message contains "corpse" or "food")
        msg = self.initial_message.lower() if hasattr(self, 'initial_message') else ''
        if 'you see here' not in msg:
            return False
        if 'corpse' not in msg and 'food' not in msg and 'ration' not in msg:
            return False
        # Check if the corpse is safe (for corpses)
        if 'corpse' in msg:
            # Extract monster name from "You see here a X corpse"
            parts = msg.split('you see here ')
            if len(parts) > 1:
                corpse_desc = parts[1].split(' corpse')[0]
                # Remove article
                for article in ['a ', 'an ', 'the ']:
                    if corpse_desc.startswith(article):
                        corpse_desc = corpse_desc[len(article):]
                        break
                if not self.food.is_corpse_safe(corpse_desc, self.resistances):
                    return False
        self._last_eat_turn = bl.time
        # Send EAT. The yn handler will answer 'y' to "eat it?"
        self.step(A.Command.EAT)
        return True

    def eat(self):
        """Eat when hungry. Two-step: EAT command then food letter."""
        bl = self.blstats
        if bl is None:
            return False
        if bl.hunger < HUNGRY:
            return False
        if bl.time - self._last_eat_turn < 5:
            return False

        # Find food in inventory
        food_letter = None
        for letter, item in self.inventory.items():
            oc = self.inv_oclasses.get(letter, -1)
            if oc == FOOD_CLASS and 'cursed' not in item.lower() and 'corpse' not in item.lower():
                food_letter = letter
                break

        if food_letter is None:
            return False

        self._last_eat_turn = bl.time
        # Two-step eat: EAT command, then food letter
        self.step(A.Command.EAT)
        # After EAT, game shows menu. Send food letter as next action.
        food_idx = self._val2idx.get(ord(food_letter))
        if food_idx is not None:
            self._env_step(food_idx)
            self._update_game_state()
        return True

    def fight(self):
        """Fight or approach visible hostile monsters."""
        mons = [(d,r,c,n,m) for d,r,c,n,m in self.get_monsters()
                if n not in PEACEFUL_NAMES and m not in PEACEFUL_IDS
                and m not in self._peaceful_monster_ids
                and (r, c) not in self._peaceful_positions]
        if not mons:
            return False
        py, px = self.blstats.y, self.blstats.x

        # Adjacent monsters
        adj = [(d,r,c,n,m) for d,r,c,n,m in mons if d <= 1]

        # Flee from instakill monsters
        for d, r, c, n, m in adj:
            if n in INSTAKILL:
                dy, dx = py - r, px - c
                self._move_dir(dy, dx)
                return True

        # If HP critical and multiple adjacent hostiles, flee
        melee_adj = [x for x in adj if x[3] not in NEVER_MELEE]
        if len(melee_adj) >= 3 and self.blstats.hp < self.blstats.max_hp * 0.2:
            # Run away from the centroid of threats
            avg_y = sum(r for _,r,c,n,m in melee_adj) / len(melee_adj)
            avg_x = sum(c for _,r,c,n,m in melee_adj) / len(melee_adj)
            dy = -1 if avg_y > py else (1 if avg_y < py else 0)
            dx = -1 if avg_x > px else (1 if avg_x < px else 0)
            if dy != 0 or dx != 0:
                self._move_dir(dy, dx)
                return True

        # Melee the first adjacent hostile
        for d, r, c, n, m in melee_adj:
            dy, dx = r - py, c - px
            self._move_dir(dy, dx)
            return True

        # Approach nearest reachable hostile within 8 BFS tiles
        if self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self._bfs_allow_hostiles()
            best_mon = None
            best_d = 999
            for d, r, c, n, m in mons:
                fd = fight_dis[r, c]
                if fd != -1 and fd <= 12 and fd < best_d:
                    best_d = fd
                    best_mon = (r, c)
            if best_mon:
                if self.step_toward(best_mon[0], best_mon[1], fight_dis):
                    return True
        return False

    def dip_excalibur(self):
        """Dip long sword in fountain for Excalibur."""
        bl = self.blstats
        if bl is None or self.has_excalibur:
            return False
        if bl.xl < 5:
            return False

        # Check if standing on fountain (use tracked positions like stairs)
        py, px = bl.y, bl.x
        on_fountain = (py, px) in self._fountains
        if not on_fountain:
            # Also check glyphs as backup
            g_here = int(self.glyphs[py, px])
            obj_here = int(self.objects[py, px]) if self.objects[py, px] != -1 else -1
            on_fountain = _cmap(g_here) in _FOUNTAIN or _cmap(obj_here) in _FOUNTAIN

        if not on_fountain:
            return False

        # Find long sword in inventory
        sword_letter = None
        for letter, item in self.inventory.items():
            if 'long sword' in item.lower() and 'cursed' not in item.lower():
                sword_letter = letter
                break
        if sword_letter is None:
            return False

        # Two-step dip: DIP command, then sword letter
        self.step(A.Command.DIP)
        sword_idx = self._val2idx.get(ord(sword_letter))
        if sword_idx is not None:
            if self._env_step(sword_idx):
                self._parse_blstats()
                raise AgentFinished()
            # Handle follow-up prompts
            self.step(A.Command.ESC)  # clear any remaining prompts
            self._update_game_state()
        # Check if Excalibur was created
        self._parse_inventory()
        if any('Excalibur' in s for s in self.inventory.values()):
            self.has_excalibur = True
        # Remove fountain if it dried up
        if (py, px) in self._fountains:
            g = int(self.glyphs[py, px]) if self.glyphs is not None else 0
            if _cmap(g) not in _FOUNTAIN:
                self._fountains.discard((py, px))
        return True

    def explore(self):
        """Explore: open doors, go to frontier, find stairs, search, descend."""
        py, px = self.blstats.y, self.blstats.x

        # 0. On downstairs: descend if ready
        if self._on_stairs_down() and self.blstats.hp > self.blstats.max_hp * 0.5:
            if self.blstats.depth == 1:
                xl_ok = self.blstats.xl >= 2
            else:
                xl_ok = self.blstats.xl >= self.blstats.depth + 1
            time_ok = self._level_turns > 30
            if xl_ok and time_ok:
                self.step(A.MiscDirection.DOWN)
                return

        # Excalibur: only dip if ALREADY on fountain (don't navigate to it)
        if not self.has_excalibur and self.blstats.xl >= 5:
            if (py, px) in self._fountains:
                self.dip_excalibur()
                return

        dis = self.bfs()

        # 1. Adjacent closed doors: open or kick them
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = py+dy, px+dx
            if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                g = int(self.glyphs[nr, nc])
                if _cmap(g) in _CLOSED_DOOR and self.door_attempts[nr, nc] < 10:
                    self.door_attempts[nr, nc] += 1
                    dmap = {(-1,0):'N',(1,0):'S',(0,1):'E',(0,-1):'W'}
                    name = dmap.get((dy, dx))
                    if name and name in self._name2idx:
                        if self.door_attempts[nr, nc] <= 2:
                            # Try opening first
                            self.step(self._name2idx[name])
                            msg_low = self.message.lower()
                            if 'locked' not in msg_low and _cmap(int(self.glyphs[nr, nc])) not in _CLOSED_DOOR:
                                return  # Door opened
                        # Door is locked or stuck: kick it
                        self._kick_dir(dy, dx)
                        if _cmap(int(self.glyphs[nr, nc])) not in _CLOSED_DOOR:
                            return  # Door broken open
                        return  # Kick attempt (may need more kicks)

        # 2. Go to nearest closed door
        best_door, best_dd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if _cmap(int(self.glyphs[r, c])) in _CLOSED_DOOR and self.door_attempts[r, c] < 5:
                    d = dis[r, c]
                    if d != -1 and d < best_dd:
                        best_dd = d
                        best_door = (r, c)
        if best_door and best_dd > 1:
            if self.step_toward(best_door[0], best_door[1], dis):
                return

        # 3. Check if we should force descent
        # Descent XL requirements: farm before descending
        if self.blstats.depth == 1:
            xl_ready = self.blstats.xl >= 2  # Farm DL1 to XL2
        elif self.blstats.depth <= 3:
            xl_ready = self.blstats.xl >= self.blstats.depth + 1
        else:
            xl_ready = self.blstats.xl >= self.blstats.depth + 1
        force_descend = self._level_turns > 80 and xl_ready

        # 3a. Navigate to stairs if force_descend and stairs known (BEFORE frontier)
        if force_descend and self._stairs_down and self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self._bfs_allow_hostiles()
            best_s, best_sd = None, 999
            for sy, sx in self._stairs_down:
                d = fight_dis[sy, sx]
                if d != -1 and d < best_sd:
                    best_sd = d
                    best_s = (sy, sx)
            if best_s:
                if best_sd == 0:
                    self.step(A.MiscDirection.DOWN)
                    return
                moved = self.step_toward(best_s[0], best_s[1], fight_dis)
                if moved:
                    return
                # step_toward failed; try greedy as fallback
                self._greedy_move_toward(best_s[0], best_s[1])
                return

        # 3b. Go to nearest frontier (walkable tile adjacent to unseen)
        best_f, best_fd = None, 999
        for r in range(MAP_H):
            for c in range(MAP_W):
                d = dis[r, c]
                if d == -1 or d >= best_fd or not self.walkable[r, c]:
                    continue
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not self.seen[nr, nc]:
                            best_fd = d
                            best_f = (r, c)
                            break
                    if best_f and dis[best_f[0], best_f[1]] == best_fd:
                        break
        if best_f:
            if self.step_toward(best_f[0], best_f[1], dis):
                return

        # 4. No frontier: navigate to stairs
        if self._stairs_down and self.blstats.hp > self.blstats.max_hp * 0.3:
            fight_dis = self._bfs_allow_hostiles()
            best_s, best_sd = None, 999
            for sy, sx in self._stairs_down:
                d = fight_dis[sy, sx]
                if d != -1 and d < best_sd:
                    best_sd = d
                    best_s = (sy, sx)
            if best_s:
                if best_sd == 0:
                    self.step(A.MiscDirection.DOWN)
                    return
                if self.step_toward(best_s[0], best_s[1], fight_dis):
                    return

        # 5. Search near walls (find hidden doors/passages)
        best_s, best_sp = None, -999
        for r in range(MAP_H):
            for c in range(MAP_W):
                if not self.walkable[r, c] or dis[r, c] == -1:
                    continue
                adj = 0
                for dy2 in (-1, 0, 1):
                    for dx2 in (-1, 0, 1):
                        if dy2 == 0 and dx2 == 0:
                            continue
                        nr, nc = r + dy2, c + dx2
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                            cm2 = _cmap(int(self.glyphs[nr, nc]))
                            if cm2 in _WALL or cm2 == 0 or not self.seen[nr, nc]:
                                adj += 1
                if adj == 0:
                    continue
                # Dead-end detection: count walkable neighbors
                walkable_neighbors = 0
                for dy3 in (-1, 0, 1):
                    for dx3 in (-1, 0, 1):
                        if dy3 == 0 and dx3 == 0:
                            continue
                        nr2, nc2 = r + dy3, c + dx3
                        if 0 <= nr2 < MAP_H and 0 <= nc2 < MAP_W and self.walkable[nr2, nc2]:
                            walkable_neighbors += 1
                # Dead ends (1 neighbor) and corridors with many walls get huge priority
                dead_end_bonus = 200 if walkable_neighbors <= 1 else 0
                p = adj * 50 + dead_end_bonus - self.search_count[r, c] ** 2 - dis[r, c] * 2
                if p > best_sp:
                    best_sp = p
                    best_s = (r, c)

        if best_s and best_s != (py, px):
            if self.step_toward(best_s[0], best_s[1], dis):
                return

        # 6. Search at current position multiple times (secret doors need ~13 searches)
        search_rounds = min(3, max(1, 10 - self.search_count[py, px]))
        for _ in range(search_rounds):
            self.search_count[py, px] += 1
            self.step(A.Command.SEARCH)
            # Check if a new door/passage appeared
            for dy2 in (-1, 0, 1):
                for dx2 in (-1, 0, 1):
                    if dy2 == 0 and dx2 == 0:
                        continue
                    nr, nc = py + dy2, px + dx2
                    if 0 <= nr < MAP_H and 0 <= nc < MAP_W:
                        g = int(self.glyphs[nr, nc])
                        cm = _cmap(g)
                        if cm in _CLOSED_DOOR or cm in _WALKABLE:
                            if not self.seen[nr, nc] or cm in _CLOSED_DOOR:
                                return  # Found something new, explore it

    def _bfs_allow_hostiles(self):
        """BFS that treats hostile monsters as walkable (can fight through them)."""
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
                    is_closed_door = cm in _CLOSED_DOOR
                    if is_closed_door and self.door_attempts[ny, nx] >= 10:
                        continue
                    # Include hostile monsters and pets, exclude peacefuls
                    is_monster = GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF
                    if is_monster and self._is_peaceful_glyph(g):
                        continue
                    if is_monster and (ny, nx) in self._peaceful_positions:
                        continue
                    ok = self.walkable[ny, nx] or is_closed_door or is_monster
                    if not ok:
                        continue
                    if abs(dy) + abs(dx) > 1:
                        if _cmap(int(self.glyphs[y, x])) in _DOOR or cm in _DOOR:
                            continue
                    dis[ny, nx] = d + 1
                    q.append((ny, nx))
        return dis

    def _greedy_move_toward(self, ty, tx):
        """Try to move closer to (ty, tx) using cardinal directions.
        With allow_all_yn_questions=False, walking into monsters auto-attacks."""
        py, px = self.blstats.y, self.blstats.x
        best_dir = None
        best_dist = abs(ty - py) + abs(tx - px)
        # Try all 8 directions, prefer cardinal
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = py + dy, px + dx
            if not (0 <= ny < MAP_H and 0 <= nx < MAP_W):
                continue
            if not self.walkable[ny, nx]:
                # Check if it's a monster we can fight through
                g = int(self.glyphs[ny, nx])
                if not (GLYPH_MON_OFF <= g < GLYPH_CMAP_OFF):
                    continue
            # Skip diagonal through doors
            if abs(dy) + abs(dx) > 1:
                src_cm = _cmap(int(self.glyphs[py, px]))
                dst_cm = _cmap(int(self.glyphs[ny, nx]))
                if src_cm in _DOOR or dst_cm in _DOOR:
                    continue
            d = abs(ty - ny) + abs(tx - nx)
            if d < best_dist:
                best_dist = d
                best_dir = (dy, dx)
        if best_dir:
            self._move_dir(best_dir[0], best_dir[1])
        else:
            # Can't move closer, search to advance turn
            self.step(A.Command.SEARCH)

    def _engrave_elbereth(self):
        """Engrave 'Elbereth' in the dust using fingers."""
        # ENGRAVE command
        self.step(A.Command.ENGRAVE)
        # "What do you want to write with?" -> '-' (fingers)
        dash_idx = self._val2idx.get(ord('-'))
        if dash_idx is not None:
            if self._env_step(dash_idx):
                self._parse_blstats()
                raise AgentFinished()
        # Handle "Do you want to add to the current engraving?" -> 'n'
        misc = self.obs.get('misc', [0, 0, 0])
        if misc[0]:
            n_idx = self._val2idx.get(ord('n'))
            if n_idx is not None:
                if self._env_step(n_idx):
                    self._parse_blstats()
                    raise AgentFinished()
        # "What do you want to write in the dust?" -> type Elbereth + Enter
        for ch in 'Elbereth\r':
            ch_idx = self._val2idx.get(ord(ch))
            if ch_idx is not None:
                if self._env_step(ch_idx):
                    self._parse_blstats()
                    raise AgentFinished()
        # Clean up any remaining prompts
        self.step(A.Command.ESC)
        self._update_game_state()

    def _on_stairs_down(self):
        """Check if player is standing on downstairs."""
        py, px = self.blstats.y, self.blstats.x
        # Check tracked stairs positions (most reliable)
        if (py, px) in self._stairs_down:
            return True
        # Check current glyph
        g_here = int(self.glyphs[py, px]) if self.glyphs is not None else 0
        cm_here = _cmap(g_here)
        if cm_here in _STAIRS_DOWN:
            return True
        # Check stored terrain
        obj_here = int(self.objects[py, px]) if self.objects[py, px] != -1 else -1
        cm_obj = _cmap(obj_here) if obj_here != -1 else -1
        return cm_obj in _STAIRS_DOWN

    def _has_frontier(self):
        """Check if there are unexplored tiles adjacent to explored walkable tiles."""
        for r in range(MAP_H):
            for c in range(MAP_W):
                if not self.walkable[r, c]:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < MAP_H and 0 <= nc < MAP_W and not self.seen[nr, nc]:
                            return True
        return False

    # ================================================================
    # Main loop
    # ================================================================

    def main(self):
        try:
            obs, info = self.env.reset(seed=self.seed)
            self.obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
            bl = obs.get('blstats')
            if bl is not None:
                self._raw_bl = bl.copy()
            self._update_game_state()

            # Clear initial prompts
            try:
                self.step(A.Command.ESC)
                self.step(A.Command.ESC)
            except AgentFinished:
                raise

            # Disable autopickup (items cause inventory issues and encumbrance)
            try:
                self.step(A.Command.AUTOPICKUP)
                if 'Autopickup: ON' in self.message:
                    self.step(A.Command.AUTOPICKUP)
            except AgentFinished:
                raise

            stall_turn = -1
            stall_count = 0
            while True:
                try:
                    # Check descent after every action
                    if self._on_stairs_down() and self._level_turns > 50:
                        if self.blstats.xl >= self.blstats.depth + 1 and self.blstats.hp > self.blstats.max_hp * 0.5:
                            self.step(A.MiscDirection.DOWN)
                            continue

                    # Stall detection: if turn doesn't advance, force search
                    cur_turn = self.blstats.time if self.blstats else 0
                    if cur_turn == stall_turn:
                        stall_count += 1
                        if stall_count > 3:
                            self.step(A.Command.SEARCH)
                            stall_count = 0
                            continue
                    else:
                        stall_turn = cur_turn
                        stall_count = 0

                    if self.emergency():
                        continue
                    if self.fight():
                        continue
                    if self.eat_ground():
                        continue
                    if self.eat():
                        continue
                    if self.dip_excalibur():
                        continue
                    self.explore()
                except RuntimeError:
                    self.step(A.Command.SEARCH)

        except AgentFinished:
            pass
        return self.score
