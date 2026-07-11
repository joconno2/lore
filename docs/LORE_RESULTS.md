# LORE — the complete result (two-sided, one mechanism)

Where LLMs help a symbolic SOTA (AutoAscend) in NetHack, and where they don't.
One mechanism explains both. All numbers measured; artifacts committed.

## The unifying mechanism (the spine)

**LLM-value = f(grounding); the floor is hallucination in exact-token spaces.**
The LLM adds value exactly when the exact tokens are PROVIDED (grounded), and
fails when it must GENERATE them. NetHack item names and code identifiers are both
exact-token spaces. This single principle explains the negative and the positive.

## Negative: LLM null at in-game DECISIONS (rule-governed)

Every in-loop decision AutoAscend makes has a precise expert rule (deep wikis +
15K lines of near-optimal heuristics), so the LLM's ceiling is parity and its floor
is hallucination:
- macro strategy (LLM ≈ mock, 92/100 identical), tactical veto (causally inert),
  endgame sequence (LLM = mock) — all **null**.
- wish selection — bare LLM agrees 1/14 with the expert rule and **hallucinates
  non-existent items 71%** ("wand of magic resistance"); retrieval-grounding →
  parity (no edge). **worse, then null**.
So "an EC-tuned LLM oracle improves a frozen symbolic SOTA at NetHack" is false:
no decision where the LLM beats the rule.

## Positive: LLM valuable at code-grounded ANALYSIS (not rule-governed)

Failure-diagnosis has no pre-existing expert rule, so the parity-ceiling doesn't
apply. Built and measured an **autonomous LLM+grounded-retrieval debugger** for the
15K-line agent — from failure symptoms alone: extract terms → validate against the
codebase vocabulary → rarity-rank → grep to the function → neutral BUG/FUNDAMENTAL
gate → diagnose. Scorecard (symptoms only, Qwen-14B, `autonomous_debug5.py`):

| case | result |
|---|---|
| DL1-stick (milestone `experience_level>=8` gate) | localized+diagnosed ✓ |
| floating-eye melee (`melee_monster_priority`) | localized+diagnosed ✓ |
| prayer-while-adjacent (`is_safe_to_pray`) | localized+diagnosed ✓ |
| Elbereth-immune loop | right root cause, approx function ◐ |
| container empty-sack (cross-file int-vs-str id) | plausible-but-wrong ✗ |
| DL8+ diverse combat (fundamental, no bug) | correctly DECLINED ✓ |

**Envelope (corrected by an unbiased real-bug benchmark).** The scorecard above is
hand-picked and OVERESTIMATES. An unbiased benchmark — every qualifying bug-fix commit
from the upstream AutoAscend git history (254 commits → 15 cases, the debugger run on
each parent tree from the author's commit-subject symptom alone; `realbug_benchmark/`)
— gives the real ladder:

| eval set | diagnosis strict | lenient |
|---|---|---|
| synthetic isolated Python bugs | 8/8 = 100% | — |
| hand-picked real (the scorecard) | 4/5 = 80% | — |
| unbiased real v2 (clean-9) | 2/9 = 22% | 4/9 = 44% |
| **unbiased real v3 (all-19, match-any)** | **1/19 = 5% [CI 1-25%]** | **6/19 = 32%** |

So the reliable strict-correct rate is **~5-22%** (single-to-low-double digits), not 80%
and not the synthetic 8/8 — synthetic benchmarks overestimate ~4-20x, and even a
hand-picked real scorecard overestimates the unbiased rate ~4-16x. And it is NON-
DETERMINISTIC: the cyclic-panic case scored correct in v2 and partial in v3 (same input,
temp 0.2). The sole reliably-correct case in both runs is `go_to_item` (empty-mask guard)
— the maximally grounded + local bug. Pipeline decomposition (`gate_stability.py`, 3
runs): retrieval is deterministic (~32% localization) and the BUG/FUNDAMENTAL gate is
STABLE (0/19 flips, 95% BUG-recall) — the variance and the low accuracy live entirely in
the generative DIAGNOSIS stage. The reliable parts are retrieval + triage; generative
diagnosis of complex real logic is the weak link (the exact-token-generation floor). **The two-factor mechanism (from the benchmark):** every
CORRECT diagnosis was also a localization hit (localization NECESSARY), but localization
is NOT sufficient — 5/9 localized, only 2 diagnosed (the 3 localized-but-wrong are
complex multi-line logic rewrites the LLM reached but could not diagnose). So
**correct diagnosis = grounded-symptom-localization × bug-locality**, both required.
Failure modes are all consistent with it: ungroundable symptom (0 search terms:
"Fixes", "Fix RL") → boilerplate (the exact-token floor again); localized-but-complex →
vague or FABRICATED (invented a syntax error that wasn't the bug); one gate false-decline.
**Which factor binds? Bug-complexity, not localization (`copilot.py`).** Handed ONLY the
exact buggy function + symptom (perfect localization — the realistic co-pilot), strict-
correct is still 2/15 = 13% (≈ autonomous), lenient rises only 32%→47%. So surfacing the
code does NOT rescue diagnosis — it refutes "better retrieval would fix it" and corrects
the earlier "co-pilot works because a human surfaces the code" claim: the co-pilot points
at the right AREA ~half the time but names the specific fix ~13%. Localization gates
attempting; bug-simplicity gates succeeding, and most real bugs are not simple.
**Not an AA artifact — it generalizes (`realbug_benchmark/generalization_rich/`).** The
same co-pilot protocol on `rich` (a pure-Python terminal-rendering library, a totally
different domain) gives 1/20 = 5% strict / 9/20 = 45% lenient — matching AA's 13%/47%.
And rich's misses are overwhelmingly EXACT-TOKEN edits (add `~` to a URL regex, `==`→`is`,
off-by-one `len(text)-1`, `assert`→default): the model sees WHERE but not WHAT-exactly,
the same exact-token floor as hallucinated NetHack item names and the Sokoban `'^'`
constant. So the diagnosis ceiling and the exact-token mechanism are domain-general.
**And it has ZERO standalone bug-DETECTION (`discrim.py`).** The 5-22% is recall under
prompts that ASSERT a bug ("what bug, what fix?"). With a neutral prompt that offers
"NO BUG", run on both buggy and fixed versions: P(BUG|buggy)=P(BUG|fixed)=1/15 —
discrimination 0.00. Offered the option it says "NO BUG" ~93% regardless of ground truth.
So the debugger cannot tell buggy from fixed code; its diagnoses appear only when a
failure is ASSERTED (the symptom is itself part of the grounding). It is strictly a
POST-FAILURE diagnoser (legitimate — you invoke it because the bot died), never an
autonomous bug finder. This forecloses the "autonomous bug-finder" overclaim; the real
positive is: given a real failure, a 5-22% grounded fix-suggester.
The controlled synthetic ceiling still holds (8/8 isolated, value-tracing 5/5, un-annotated
found — "value-tracing fails" and "annotation inflates" both DISPROVEN), but isolated
bugs are the easy tail: the debugger is REAL but NARROW.
Retrieval had to be GROUNDED: LLM-generated search terms (keyword / name-select /
agentic) all hallucinated non-existent identifiers and failed — the same floor as
the wish decision. Grounding the search (real symptom-terms) unlocks it. The LLM
also **co-found 2 novel bugs** this way (prayer, floating-eye), both fixed. Generic,
not AA-specific: it diagnosed a real Python bug in the LORE tooling too.

## Engineering (non-LLM, banked)

- **DL1-stick** is AutoAscend's own dominant flaw: the `BE_ON_FIRST_LEVEL` milestone
  gates leaving DL1 on `experience_level>=8`, so ~52% of games (baseline validated
  fair vs native config) farm DL1 to starvation and never descend. `apply_unstick_dl1`
  fixes it — the ONE lever that moves depth (base median DL2 → DL5).
- The DL5 wall past it is fundamental combat survival — invariant across 13 macro
  variants (n=1660), forced Valkyrie, disengage, dive-pacing, AA-parameter-tuning,
  and 4 targeted death-mode bug-fixes (Elbereth/prayer/eye correct but depth-neutral:
  p=0.16-0.93; they re-route deaths to combat). Ascension lives only in the rare tail.
- Endgame via wizard `^V` is bounded: confined ~54-cell no-dig pockets, planes
  unreachable (dnum locked). Not legitimate ascension progress.

## The paper

"LLMs can't out-DECIDE a symbolic SOTA in a rule-governed game; a grounded LLM
pipeline CAN autonomously debug it, but only within a sharp envelope — correct
diagnosis = grounded-symptom-localization × bug-locality, ~22% on unbiased real bugs
(not the 80% a hand-picked scorecard implies)." One mechanism (grounding unlocks;
exact-token hallucination floors; locality bounds) ties all three together: the null
decisions, the bounded debugger, and its measured failure modes. The contribution is
the ENVELOPE, mapped and mechanistically explained on a real 15K-line agent — negative
+ bounded-positive + the two-factor law that predicts which bugs it gets.

Detail: `Agent/daily/2026-07-11.md`. Code: `experiments/autoascend/` (oracle.py,
wish_grounded.py, llm_debug_*.py, autonomous_debug{4,5}.py, the *_fix/*_safety patches).
