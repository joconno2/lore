import json, glob, statistics as st, collections
def num(x):
    try: return int(x)
    except: return None
def descriptors(pat, label):
    R = [json.load(open(f)) for f in glob.glob(pat)]
    if not R: return None
    n = len(R)
    def frac(k): return round(100 * sum(1 for r in R if r.get(k)) / n)
    dc = collections.Counter(r.get("death_cat") for r in R)
    d = [num(r.get("max_depth")) for r in R if num(r.get("max_depth")) is not None]
    xl = [num(r.get("xl")) for r in R if num(r.get("xl")) is not None]
    # behavior descriptor vector: branch coverage + death profile + depth/xl
    print("%-9s n=%3d | mines %2d%% soko %2d%% minetown %2d%% quest %2d%% | deaths: %s | dl med %s xl med %s" % (
        label, n, frac("did_mines"), frac("did_sokoban"),
        frac("did_minetown"), frac("did_quest"),
        {k: round(100*v/n) for k, v in dc.most_common(3)},
        st.median(d) if d else "-", st.median(xl) if xl else "-"))
# strategically-distinct macro variants
descriptors("prof_mcr_*.json", "mines1st")   # default (mines-first bias)
descriptors("bo_soko_*.json", "soko1st")     # soko-first branch order
descriptors("soko1_*.json", "soko1")
descriptors("mabl_*.json", "mabl")
descriptors("mablf_*.json", "mablf")
descriptors("abl_full_*.json", "full")
descriptors("be_engine_*.json", "engine")
descriptors("cfb_base_*.json", "base")       # no director -- behavior floor
