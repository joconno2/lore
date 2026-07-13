"""Offline validation of the live Sokoban solver (no agent deps).

Loads the vendored map data + the solver directly (both dependency-light) and
checks: (1) it solves each template board, (2) it re-solves boards perturbed by
random legal pushes (the mid-solve live-divergence case that crashes the
replay-based vendored solver). Solutions are replayed through a local copy of
the vendored move() semantics to confirm boulders_left == 0.
"""
import os
import sys
import random
from collections import deque

import numpy as np

SOKO = os.path.join(os.path.dirname(__file__),
                    "data/bots/autoascend/autoascend/soko_solver")
sys.path.insert(0, SOKO)
import maps as maps_mod          # pure data
import live_solve as L           # pure search

IGNORE, EMPTY, WALL, BOULDER, TARGET, START = 0, 1, 2, 3, 4, -1
CH = {'<': EMPTY, '>': START, '.': EMPTY, '?': EMPTY, '+': EMPTY,
      '0': BOULDER, '-': WALL, '|': WALL, ' ': IGNORE, '^': TARGET}


class SM:
    """Minimal stand-in for autoascend.soko_solver.SokoMap (same move semantics)."""
    def __init__(self, pos, grid):
        self.pos = pos
        self.sokomap = grid

    def _reach(self):
        h, w = self.sokomap.shape
        seen = np.zeros((h, w), bool)
        dq = deque([self.pos]); seen[self.pos] = True
        while dq:
            y, x = dq.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and self.sokomap[ny, nx] == EMPTY:
                    seen[ny, nx] = True; dq.append((ny, nx))
        return seen

    def move(self, by, bx, dy, dx):
        g = self.sokomap
        assert g[by, bx] == BOULDER
        assert g[by - dy, bx - dx] == EMPTY
        assert self._reach()[by - dy, bx - dx]
        assert g[by + dy, bx + dx] in (EMPTY, TARGET)
        self.pos = (by, bx)
        g[by, bx] = EMPTY
        if g[by + dy, bx + dx] == EMPTY:
            g[by + dy, bx + dx] = BOULDER
        # onto TARGET: boulder falls in and vanishes


def strip_cols(text, n):
    return "\n".join(l[n:] for l in text.splitlines())


def convert(text):
    rows = [[CH[c] for c in line] for line in text.splitlines() if line]
    g = np.array(rows)
    if int((g == TARGET).sum()) != 1:
        raise ValueError("need exactly one target")
    st = list(zip(*(g == START).nonzero()))
    if len(st) != 1:
        raise ValueError("need one start")
    pos = st[0]
    g[g == START] = EMPTY
    return SM(pos, g)


def fresh(smap):
    for c in range(0, 9):
        try:
            return convert(strip_cols(smap, c)), c
        except Exception:
            continue
    return None, None


def bcount(sk):
    return int((sk.sokomap == BOULDER).sum())


def replay(sk, moves):
    for m in moves:
        sk.move(*m)
    return bcount(sk)


def legal_pushes(sk):
    g = sk.sokomap
    reach = sk._reach()
    h, w = g.shape
    out = []
    for by, bx in zip(*(g == BOULDER).nonzero()):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sy, sx = by - dy, bx - dx
            dyy, dxx = by + dy, bx + dx
            if not (0 <= dyy < h and 0 <= dxx < w and 0 <= sy < h and 0 <= sx < w):
                continue
            if g[sy, sx] == EMPTY and reach[sy, sx] and g[dyy, dxx] in (EMPTY, TARGET):
                out.append((int(by), int(bx), dy, dx))
    return out


def main():
    random.seed(0)
    n_ok = n_tot = p_ok = p_tot = 0
    for i, (smap, _ans) in enumerate(maps_mod.maps.items()):
        sk, col = fresh(smap)
        if sk is None:
            print(f"map {i:2d}: UNPARSEABLE"); continue
        n_tot += 1
        moves = L.solve(convert(strip_cols(smap, col)))
        solved = moves is not None and replay(convert(strip_cols(smap, col)), moves) == 0
        n_ok += solved

        pk = convert(strip_cols(smap, col))
        made = 0
        for _ in range(3):
            lp = legal_pushes(pk)
            if not lp or bcount(pk) == 0:
                break
            pk.move(*random.choice(lp)); made += 1
        p_tot += 1
        if bcount(pk) == 0:
            p_ok += 1; pstat = "presolved by pushes"
        else:
            pm = L.solve(pk)
            ok = pm is not None and replay(pk, pm) == 0
            p_ok += ok
            pstat = (f"re-solved after {made} pushes" if ok else f"RE-SOLVE FAILED ({made} pushes)")
        print(f"map {i:2d} col{col}: {'SOLVED' if solved else 'FAILED':6s} "
              f"{len(moves) if moves else '-'} pushes | {pstat}")
    print(f"\nTemplate: {n_ok}/{n_tot}   Perturbed re-solve: {p_ok}/{p_tot}")


if __name__ == "__main__":
    main()
