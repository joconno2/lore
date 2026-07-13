import os, sys, time
import numpy as np
SOKO = os.path.join(os.path.dirname(__file__), "data/bots/autoascend/autoascend/soko_solver")
sys.path.insert(0, SOKO)
import maps as maps_mod, live_solve as L
IGNORE, EMPTY, WALL, BOULDER, TARGET, START = 0, 1, 2, 3, 4, -1
CH = {'<': EMPTY, '>': START, '.': EMPTY, '?': EMPTY, '+': EMPTY,
      '0': BOULDER, '-': WALL, '|': WALL, ' ': IGNORE, '^': TARGET}


class SM:
    def __init__(s, pos, g): s.pos = pos; s.sokomap = g
    def _reach(s):
        from collections import deque
        h, w = s.sokomap.shape; seen = np.zeros((h, w), bool); dq = deque([s.pos]); seen[s.pos] = True
        while dq:
            y, x = dq.popleft()
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                ny, nx = y+dy, x+dx
                if 0<=ny<h and 0<=nx<w and not seen[ny,nx] and s.sokomap[ny,nx]==EMPTY:
                    seen[ny,nx]=True; dq.append((ny,nx))
        return seen
    def move(s, by, bx, dy, dx):
        g = s.sokomap
        assert g[by,bx]==BOULDER and g[by-dy,bx-dx]==EMPTY and s._reach()[by-dy,bx-dx] and g[by+dy,bx+dx] in (EMPTY,TARGET)
        s.pos=(by,bx); g[by,bx]=EMPTY
        if g[by+dy,bx+dx]==EMPTY: g[by+dy,bx+dx]=BOULDER


def strip(t, n): return "\n".join(l[n:] for l in t.splitlines())
def conv(t):
    g = np.array([[CH[c] for c in l] for l in t.splitlines() if l])
    if int((g==TARGET).sum())!=1: raise ValueError("targets")
    st = list(zip(*(g==START).nonzero()))
    if len(st)!=1: raise ValueError("start")
    p = st[0]; g[g==START]=EMPTY; return SM(p, g)
def bcount(sk): return int((sk.sokomap==BOULDER).sum())


def aligned_col(smap, ans):
    """Column strip where the KNOWN solution replays to 0 boulders (the oracle)."""
    for c in range(9):
        try:
            sk = conv(strip(smap, c))
            for (y, x), (dy, dx) in ans:
                sk.move(y, x, dy, dx)
            if bcount(sk) == 0:
                return c
        except Exception:
            continue
    return None


for i, (smap, ans) in enumerate(maps_mod.maps.items()):
    col = aligned_col(smap, ans)
    if col is None:
        print(f"map {i}: NO ALIGNED COL", flush=True); continue
    sk = conv(strip(smap, col)); nb = bcount(sk)
    live = L._live_squares((sk.sokomap==EMPTY)|(sk.sokomap==BOULDER),
                           frozenset(map(tuple, zip(*(sk.sokomap==TARGET).nonzero()))))
    t0 = time.time(); mv = L.solve(conv(strip(smap, col)), max_expansions=100000); dt = time.time()-t0
    ok = False
    if mv is not None:
        rk = conv(strip(smap, col))
        try:
            for m in mv: rk.move(*m)
            ok = bcount(rk) == 0
        except Exception:
            ok = False
    print(f"map {i}: boulders={nb} col={col} live_sq={len(live)} ans_len={len(ans)} "
          f"-> mine={'sol'+str(len(mv)) if mv else 'NONE'} valid={ok} in {dt:.2f}s", flush=True)
