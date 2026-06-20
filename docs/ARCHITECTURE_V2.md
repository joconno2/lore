# LORE v2: EC-tuned LLM oracle on a frozen AutoAscend base

Decision date: 2026-06-20. Supersedes the expert-system line (agent2*.py).

## The pivot

The original LORE built its own NetHack expert system (agent2.py ... agent2_v5.py)
to reach the deep game, then planned an EC/LLM knowledge layer on top. Two
problems killed that path:

1. The expert system was a clean-room reimplementation of AutoAscend. Its own
   modules cite it (`fight.py` <- fight_heur.py, `equipment.py` <- character.py).
   The `AUTOASCEND_GAP_ANALYSIS.md` doc is literally "what AutoAscend does that
   we don't." Reimplementing a 15K-line bot to maybe reach its level is pure
   engineering with no research novelty.
2. It runs in NetHackScore-v0 (23 actions) which physically cannot wield, wear,
   use items, engrave, sacrifice, or dip. Six of ten gaps to AutoAscend are
   action-space-blocked. The best expert version (v2) scored mean 700; the Jun-4
   iterations regressed to 125-185.

Meanwhile AutoAscend (NeurIPS 2021 NetHack Challenge winner) is vendored at
`data/bots/autoascend/` and, benchmarked on our infra (10 ep, NetHackChallenge-v0),
scores **mean 23,670 / median 17,161 / max 99,288**, reaching DL12.

So: freeze AutoAscend as the base. LORE becomes the EC-tuned LLM oracle that
intervenes at AutoAscend's *weak* decision points.

## Why this is a paper (not a +3% nudge)

A near-optimal base shrinks the headroom for any add-on. The escape is to target
where AutoAscend is *not* good:

- Its strategy (`global_logic.py:current_strategy`) is a fixed linear milestone
  route. Deep-dungeon descent (`GO_DOWN`) is an unimplemented `TODO`. AutoAscend
  has no real endgame past Sokoban / Mines End. That is the score plateau and
  the reason it scores high but does not ascend.
- It dies in recurring knowledge-dependent ways. 10-ep sample: 3/10 deaths
  "while fainted" (food management), 1 shopkeeper anger, 1 prayer-timing death,
  1 item-parse `AssertionError` crash.
- Its main loop kills the agent on 5 consecutive panics (`Cyclic Panic`
  RuntimeError). Those are explicit "I don't know what to do here" moments.

The contribution: replace/augment specific high-uncertainty decision points with
an LLM oracle whose retrieval interface is EC-optimized (when to query, what
context, how to parse), and show it beats the frozen heuristic at those points /
lifts ascension. Clean attribution (discrete symbolic decision points), targets
the actual open problem, and is novel (NetPlay is zero-shot LLM at 405; Motif is
reward shaping; none augment a SOTA symbolic bot).

## Frozen base setup

- Image: `aall/autoascend:frozen` (built on trx). NGC pytorch:21.08-py3 base,
  NLE 0.7.3 (seeding-patched, compiled from source), gym 0.19, numpy 1.21.
  Build gotcha fixed: `git submodule update --init --recursive` after the v0.7.3
  checkout (recursive clone + tag checkout leaves libtmt empty -> missing tmt.h).
- AutoAscend's code is NOT baked into the image. Mount the source and run:
  `docker run --rm -v <src>:/workspace -w /workspace -e PYTHONPATH=/workspace
   aall/autoascend:frozen python <harness> <N> <base_seed> <out.json>`
- Headless harnesses (no Ray, no X11):
  - `aa_bench.py` -- score/depth benchmark.
  - `aa_gap.py` -- gap analysis: per-episode panic reasons, end-reason buckets,
    milestone distribution, depth. Finds the weak decision points.
- Compute: trx only (qd-bw roundrobin co-tenant is fine). Not mega_knight
  (QD-Continual) or its Condor pool while those run. Large sweeps -> Condor later.

## Plan

1. [done] Freeze AutoAscend, confirm it scores in range on our infra.
2. [running] Gap analysis (50+ ep): quantify panic histogram, death causes,
   milestone reach. Rank decision points by headroom.
3. Pick 3-5 oracle intervention points from the ranked gaps.
4. Wire the LLM oracle (Gemma via vLLM on trx) at those points behind a clean
   interface. v2 expert system kept only as a fast mechanism testbed.
5. EC-optimize the oracle interface (CMA-ES / MAP-Elites). Fitness = score /
   ascension. Measure delta over the frozen base.

Target: CoG 2027.
