#!/usr/bin/env python3
"""Analyze episodes with bad steps/turn ratio."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/expert_v24_200ep.json"
with open(path) as f:
    d = json.load(f)
eps = d["episodes"]

bad = []
for e in eps:
    spt = e["steps"] / max(e["turn"], 1)
    if spt > 2.0:
        bad.append(e)

print(f"Bad SPT (>2.0) episodes: {len(bad)}/{len(eps)}")

# Sort by SPT
bad.sort(key=lambda e: e["steps"] / max(e["turn"], 1), reverse=True)
print("\nWorst 15 by SPT:")
for e in bad[:15]:
    spt = e["steps"] / max(e["turn"], 1)
    seed = e["seed"]
    score = e["score"]
    steps = e["steps"]
    turns = e["turn"]
    dl = e["dlevel"]
    xl = e["xlevel"]
    print(f"  seed={seed:3d} spt={spt:6.1f} steps={steps:5d} turns={turns:5d} score={score:4.0f} dl={dl} xl={xl}")

# SPT distribution
import statistics
all_spt = [e["steps"] / max(e["turn"], 1) for e in eps]
print(f"\nSPT distribution:")
print(f"  mean={statistics.mean(all_spt):.2f}")
print(f"  median={statistics.median(all_spt):.2f}")
print(f"  p90={sorted(all_spt)[int(len(all_spt)*0.9)]:.2f}")
print(f"  p95={sorted(all_spt)[int(len(all_spt)*0.95)]:.2f}")
print(f"  max={max(all_spt):.2f}")

# Good vs bad score comparison
good = [e for e in eps if e["steps"] / max(e["turn"], 1) <= 2.0]
good_scores = [e["score"] for e in good]
bad_scores = [e["score"] for e in bad]
print(f"\nGood SPT (<=2): n={len(good)} mean_score={statistics.mean(good_scores):.0f}")
print(f"Bad SPT (>2):   n={len(bad)} mean_score={statistics.mean(bad_scores):.0f}")
