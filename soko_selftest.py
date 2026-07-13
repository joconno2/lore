from autoascend import soko_solver as ss
def strip_cols(text, n):
    return "\n".join(l[n:] for l in text.splitlines())
for i, (smap, ans) in enumerate(ss.maps.items()):
    found = None
    for c in range(0, 9):
        try:
            sk = ss.convert_map(strip_cols(smap, c))
            ok = True
            for (y, x), (dy, dx) in ans:
                sk.move(y, x, dy, dx)
            rem = int((sk.sokomap == ss.BOULDER).sum())
            if rem == 0:
                found = c; break
        except Exception:
            continue
    print("map %d  ->  col_strip=%s" % (i, found))
