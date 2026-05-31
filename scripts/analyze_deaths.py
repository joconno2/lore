#!/usr/bin/env python3
"""Analyze death patterns from eval results."""
import json, sys, statistics

path = sys.argv[1] if len(sys.argv) > 1 else "results/expert_v14_law_200ep.json"
with open(path) as f:
    d = json.load(f)
eps = d["episodes"]

scores = [e["score"] for e in eps]
print(f"Total episodes: {len(eps)}")
print(f"Score: mean={statistics.mean(scores):.1f} med={statistics.median(scores):.1f}")

# Zero/negative episodes
zeros = [e for e in eps if e["score"] <= 0]
print(f"\nZero/negative: {len(zeros)}/{len(eps)}")
for e in sorted(zeros, key=lambda e: e["steps"])[:10]:
    s = e["score"]
    st = e["steps"]
    t = e["turn"]
    dl = e["dlevel"]
    xl = e["xlevel"]
    print(f"  score={s:.0f} steps={st} turns={t} dl={dl} xl={xl}")

# Steps distribution
print(f"\nAll episodes steps: mean={statistics.mean([e['steps'] for e in eps]):.0f}")
print(f"All episodes turns: mean={statistics.mean([e['turn'] for e in eps]):.0f}")

# Score brackets
brackets = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000)]
print("\nScore distribution:")
for lo, hi in brackets:
    n = sum(1 for s in scores if lo <= s < hi)
    print(f"  [{lo:4d}, {hi:4d}): {n:3d} ({100*n/len(eps):.0f}%)")

# Steps/turn distribution
spt = [e["steps"]/max(e["turn"],1) for e in eps]
print(f"\nSteps/turn: mean={statistics.mean(spt):.1f}")
bad_spt = sum(1 for s in spt if s > 2.0)
print(f"Episodes with SPT > 2: {bad_spt}/{len(eps)}")

# DL/XL distribution
from collections import Counter
print(f"\nDL distribution: {dict(Counter(e['dlevel'] for e in eps).most_common())}")
print(f"XL distribution: {dict(Counter(e['xlevel'] for e in eps).most_common())}")
