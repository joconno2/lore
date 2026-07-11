# LORE ascension capability ladder — 2026-07-11

The two-fold bar (Jim): beat the symbolic SOTA (AutoAscend) AND do it with a novel
LLM angle, not hard-coding on AA. Metric = ascension-progress. This doc is the
paper-ready summary of the ENDGAME half: milestones AutoAscend structurally cannot
reach, and what the LORE scenario layer has proven, with the current gap.

## Why the endgame is the LLM angle

The progress-engine half (macro director) beats base AA on progress but is a
hand-rule — the LLM adds no edge there (valid null: LLM ≈ mock, richer state 92%
identical). The endgame is different: AutoAscend has **zero code** for it. Its
`GO_DOWN` milestone is an unimplemented TODO; it never assembles the ascension kit,
never performs the invocation, never enters the planes, never ascends. So driving
the endgame sequence is NEW capability, not rebuilding AA — this clears the
integrity bar (fix AA's bugs structurally; only claim the LLM for what AA lacks).

## The ladder (each rung: AA's rate → LORE status)

AA rates are from the n=250-450 behavioral profile (our NetHackChallenge-v0
distribution), where AA reaches median DL2-3, deepest ~DL17, and 0% of any endgame
milestone.

| # | Rung | AA | LORE status (scenario harness, wizard placement + wished kit) |
|---|------|----|--------------------------------------------------------------|
| 1 | Hold invocation kit (Bell, Candelabrum+7 candles, Book of the Dead) | 0% | **PROVEN** — items wishable; AA's `parse_text` ring/amulet-worn crash fixed (monkeypatch, AA stays frozen) |
| 2 | Perform the invocation RITUAL (attach candles→light candelabrum→ring Bell→read Book) | 0% | **PROVEN** — executes via low-level keypresses (bypasses AA's assert-on-unknown-items); candelabrum ends "(7 candles, lit)" |
| 3 | Reach Gehennom, equipped and alive | 0% | **PROVEN** — teleport lands in Gehennom (dungeon_num=1, DL28-29); tank kit (XL30, AC-15) survives in place; food/choke handling fixed |
| 4 | Reach the vibrating square (fire the REAL invocation) | 0% | **GAP** — needs Gehennom maze descent; teleport clamps at ~DL29 so it must walk. Current descent agent reaches the downstair **0/16 seeds** (dies mid-traversal to varied Gehennom monsters before covering the maze) |
| 5 | Take the Amulet from Moloch's Sanctum | 0% | not started (blocked on #4) |
| 6 | Ascend the 4 Elemental planes | 0% | not started |
| 7 | Astral plane #offer → ASCEND | 0% | not started |

## The gap at rung 4 (root-caused, instrumented)

Reaching the vibrating square is the sole blocker to the real invocation. Evidence:
- Wizard `^V` teleport CLAMPS to the deepest generated level (~DL29); the
  invocation level (~DL45-53) is not generated, so it must be reached by descending.
- AA's exploration is Gehennom-blind: it stalls in the maze; the downstair is deep
  in an un-traversed region and is never revealed.
- Built a model-free descent (glyph-BFS stair-take, frontier explorer, dig-toward-
  unexplored). The exploration WORKS (covers a large connected area, reachable set
  expands) but is a **survival-across-full-traversal** problem: the agent dies to
  varied Gehennom monsters (vampire lord, umber hulk, giant eel drowning, barbed
  devil…) before covering the maze. Multi-seed: **downstair reached 0/16**.

So rung 4 needs a holistic **Gehennom descent agent**: sustained survival (heal
+ selective flee, not fight-everything) + systematic maze coverage (+ dig/search
for the walled-off downstair) + reliable per-step reflexes. Substantial, but every
component is de-risked and instrumented.

## Two paths (the escalated scope decision)

(a) **Build the full Gehennom descent agent** — the end-to-end path to an in-game
invocation. Substantial; the pieces exist and need integration + survival tuning,
evaluated multi-seed from the start (single-seed is RNG-confounded by kit edits).

(b) **Scenario-isolate rungs 4-7 as pure capability demos** — place the agent
directly at each rung (vibrating square, Sanctum, each plane) with the kit, and
show the LLM+wiki drives the sequence AA cannot, skipping the Gehennom-nav slog.
Faster to a paper result; a capability claim rather than fair-reach.

Rungs 1-3 are proven regardless. The paper's endgame contribution is the ladder:
"an LLM+knowledge layer drives a frozen symbolic SOTA through endgame milestones it
structurally cannot reach," with the reach clearly bounded.

Detail + instrumented arc: `Agent/daily/2026-07-11.md`. Code: `experiments/
autoascend/lore_scenario.py`, `descent_run.py`. Commits through feda313.
