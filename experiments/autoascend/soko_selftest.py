from autoascend import soko_solver as ss
for i, (smap, ans) in enumerate(ss.maps.items()):
    sk = ss.convert_map(smap)
    ok, err, done = True, None, 0
    for (y, x), (dy, dx) in ans:
        try:
            sk.move(y, x, dy, dx); done += 1
        except Exception as e:
            ok, err = False, (done, str(e)[:50]); break
    rem = int((sk.sokomap == ss.BOULDER).sum())
    print("map %d  anslen %d  %s  boulders_left %d" % (
        i, len(ans), "OK" if ok else "FAIL@%d %r" % err, rem))
