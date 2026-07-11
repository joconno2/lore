# LORE status — 2026-07-11 (supersedes LORE_STATUS_2026-06-29)

Metric: **ascension-progress**, not score (Jim, Jul 10). Base = frozen AutoAscend,
NetHackChallenge-v0, NLE 0.7.3, `aall/autoascend:frozen` on trx. All numbers are
paired (same seeds) unless noted. Ladder scored by `progress_ladder.py`.

## Headline: the progress engine beats base decisively (hand-rule)

ENGINE = mock macro-director + `crash_recovery` + `sokoban_fix`. n=100 paired, seeds 800-899.

| metric | base AA | engine | 
|---|---|---|
| median depth | DL1 | DL4 |
| mean depth | 2.78 | 4.97 |
| reached DL5 | 24% | 49% |
| reached DL15 | 0% | 2% |
| deepest | DL9 | DL17 |
| starvation deaths | 45% | 21% |

Engine wins depth on 68/100 seeds (base 21, tie 11). ~2x median depth, half the
starvation. **This is a real progress win over the symbolic SOTA — but it is the
hand-rule macro, not an LLM.**

## The LLM macro-director adds no edge (VALID negative)

The oracle vLLM (Qwen2.5-14B-AWQ) runs on trx's own GPU. **Prior "LLM worse"
results were a dead-server artifact (vLLM down → every query fell back to
GO_MINES) and are void.** With a live server (32 real queries/game):

- LLM macro vs mock hand-rule, n=100 paired: **median DL5 both**, LLM slightly
  worse (over-dives to Mines, 67 vs 56 combat deaths, Minetown 17 vs 24%).
- LLM with richer state (AC, gold, kit, visible-hostiles, nearest-hostile-dist)
  vs plain LLM: **92/100 seeds identical.** Extra information does not change the
  decision.

Conclusion: the macro-objective choice (1-of-7 branch objectives) is
**rule-saturated** — a low-dimensional strategic pick a fixed expert order already
nails. No decision space for LLM judgment to add value.

## Every within-engine macro lever is null

- **DL1-leave XL cushion** (3/6/8): higher cushion = MORE starvation (20→27%) and
  LESS depth. DL1 is both food- and XP-scarce; lingering just starves. Dead end.
- **Branch order** soko_first vs mines_first: mixed. soko_first slightly deeper
  (mean 4.80 vs 4.45) but reaches Minetown 1% vs 17% (physical) — it abandons the
  Minetown strength-spike (altar/BUC-ID/shops). Not a win. Keep mines_first.
- **Descent gate** (hold `>` when under-geared): impossible via move-veto —
  blocking descent with no alternate action trips AA's cyclic-panic guard
  (inactivity_counter ≥ 5). NetHack gives no safe shallow XP to consolidate on.
- **Sokoban indent fix**: correct structural bug-fix (hard AssertionError →
  graceful abandon), but ~0 aggregate ladder impact (only ~4-7% reach Sokoban).

## Why: depth is gated by the early-mid survival gauntlet

Deep-survivor split (engine, n=100): DL≥10 games (n=9) all cleared the Mines
(100%) and 44% did Sokoban, reaching XL10. The 51% that die shallow (DL≤4) mostly
never reached a branch (Mines 11%), dying to combat (32) and starvation (15)
under-leveled. **Getting through the DL1-4 gauntlet into the branches is the
gateway to depth; it resists macro levers** (routing/pacing/gating all null) and
tactical levers (AA is already tactically SOTA; per-decision veto proven null).

## Methodology note

`progress_ladder.py`'s Minetown/SOLVED-Soko metrics were milestone-based and
**contaminated** — the macro director sets FIND_SOKOBAN as an objective, so
`milestone_num ≥ 3` fired without physically reaching Minetown (inflated
soko_first "Minetown" 87% → physically 1%). Fixed: `aa_profile.py` now records
physical `did_minetown` (`global_logic.minetown_level`); the ladder prefers it.
**Trust physical branch fields, never `milestone_num`, for director comparisons.**

## Two-fold bar and the open pivot

- Beat SOTA on progress: **YES** (engine, hand-rule).
- Via a novel LLM angle: **NOT YET** — the macro-director is exhausted.

Escalated to Jim (fix-or-pivot). LLM-angle options:
- (a) richer/harder macro decisions — **now shown null**.
- (b) the ENDGAME sequence AA has zero code for (invocation → Amulet → planes; no
  hand-rule to beat). Gated on games reaching deep enough (currently ~0% past DL15).
- (c) knowledge-gap calls (wish text, altar BUC-ID, prayer timing) — testable sooner.

Both (b) and (c) ultimately need the early-mid survival gauntlet cracked first.

Arc: `Agent/daily/2026-07-11.md`. Commits: 51d1949, 35ddd3d.
