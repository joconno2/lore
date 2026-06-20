# AutoAscend gap analysis (frozen base), 2026-06-20

50 episodes, NetHackChallenge-v0, seeds 100-149, `aall/autoascend:frozen` on trx.
Data: `results/bench_2026-06-20/autoascend_gap_50ep.json`. Harness: `aa_gap.py`.

## Headline

mean 14,283 / median 9,139 / max 74,792 / min 324.

## Where it ends (milestone reached at episode end)

| Milestone | Episodes |
|-----------|----------|
| BE_ON_FIRST_LEVEL | 17 |
| FIND_SOKOBAN | 14 |
| FIND_MINETOWN | 9 |
| FIND_GNOMISH_MINES | 6 |
| SOLVE_SOKOBAN | 4 |
| (anything past Sokoban) | **0** |

**0 of 50 episodes advance past Sokoban in the scripted milestone route.** Some
runs reach DL15-18 by depth, but the strategy layer has no endgame milestones
driving them (`GO_DOWN` is an unimplemented TODO). AutoAscend dives without a
plan once the scripted route runs out. This is the score ceiling and the reason
it does not ascend.

## How it dies (end reasons)

- Monster deaths, spread thin across ~30 species (white unicorn x6 is the mode).
  No single dominant killer -> mostly "fair" deaths from inherent difficulty.
- **5/50 (10%) are `AssertionError` crashes** in AutoAscend's own item/parse
  logic, not game deaths. 10% of episodes lost to bugs.
- Petrification (cockatrice/chickatrice) x3 -- knowledge-dependent instadeaths.

## Friction (panics: 11,197 total, ~224/episode)

Dominated by low-level state-desync, all recovered-from:
- "items below me changed" 1720
- "end point is no longer accessible" 1313
- "no such food is lying here" 1084
- "position changed" 406
- "Monster on a next tile when moving" / "position do not match after move" (many)

These are mechanical state-tracking issues, not decision points. Not LLM-oracle
targets (an LLM cannot help with "item below me changed"). They cost steps, not
games.

## Ranked oracle targets

1. **Post-Sokoban / deep-game strategy (highest headroom).** 0/50 get a real
   endgame plan. The oracle supplies high-level strategy for the deep dungeon
   (branch choice, when to dive vs consolidate, quest, altar/gear goals). Targets
   the score ceiling and ascension directly. This is the paper.
2. **Instadeath veto at knowledge-dependent points (targeted).** Petrification,
   corpse safety, unicorn/shopkeeper anger. Oracle as a warn/veto on provably
   fatal moves. Individually rare but high-cost.
3. **Crash robustness (cheap, non-research).** 10% AssertionError rate -> a
   try/except wrapper that ESCs and continues recovers ~free score. Worth doing
   but it is code-hardening, not oracle work.

## Intervention #1 result: Sokoban crash patch (lore_patches.py)

Runtime monkeypatch (no base edit) routing the Sokoban-solver desync into
AutoAscend's own abandon path. 50 ep, same seeds:

| | baseline | patched |
|---|---|---|
| mean | 14,283 | 15,850 (+11%) |
| median | 9,139 | 9,139 |
| max | 74,792 | 78,302 |
| AssertionError crashes | 5 | 2 |
| milestone reach | caps at SOLVE_SOKOBAN | 4 reach FIND_MINES_END |

First episodes to advance past Sokoban. Recovered 3/5 deep crashes (one seed
24k->78k). Residual: 2 different brittle asserts (turn-inactivity, chest-open) ~1
episode each -- diminishing. Median unmoved: crashes were never the median run's
problem. Architecture (wrap, don't edit) validated.

## Next oracle target (from this data)

- Tactical instadeath veto: **white unicorn kills 6/50 (12%)**, petrification 3.
  High-frequency, knowledge-dependent, doesn't require reaching the deep game.
  Best first oracle target.
- Strategic endgame: still 0 past Mines End. Bigger ceiling, but needs the agent
  to reliably get deep first.

## Implication for the design

The thesis holds with data: AutoAscend's weakness is the absent endgame (0/50
past Sokoban) plus specific knowledge-deaths, not its early/mid-game mechanics.
The LLM oracle's home is high-level deep-game strategy (#1), with instadeath veto
(#2) as a tactical second target. EC optimizes when/what to query at those points.
