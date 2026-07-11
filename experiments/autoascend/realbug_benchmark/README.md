# Real-bug diagnosis benchmark (unbiased, upstream AutoAscend history)

The hand-picked debugger scorecard (4/5) and the synthetic benchmark (8/8) both
OVERESTIMATE. This measures the autonomous debugger on **unbiased real bugs**: every
qualifying bug-fix commit from the upstream AutoAscend git history, not cherry-picked.

## Method

- `aa_extract.py` — mine `github.com/maciej-sypetkowski/autoascend` (254 commits). Take
  every commit whose subject matches fix/bug, find the PRIMARY changed function (most
  changed parent-side lines) in an agent-logic file, keep it if that one function holds
  >=55% of the commit's changed lines and is 4-95 lines. Emit (author-symptom = commit
  subject, buggy parent function, ground-truth fix diff). 15 cases.
- `bench_run.py` (on trx, Qwen-14B vLLM) — for each case, checkout the PARENT (buggy)
  tree and run the EXACT `autonomous_debug5` pipeline from the author symptom alone:
  grounded+rarity retrieval -> BUG/FUNDAMENTAL gate -> diagnosis. Records retrieved
  functions, gate, diagnosis.
- `aa_score.py` — hand-scored verdicts (judge: a stronger model, ground-truth diffs in
  hand). CLEAN = symptom genuinely describes the extracted function (fair test); BUNDLED
  = commit bundled changes, subject describes a different change than the extracted
  function (symptom/function mismatch, reported but excluded from the fair rate).

## Result

Two unbiased runs. **v2** (`aa_extract.py`/`bench_run.py`/`aa_score.py`): 15 single-
primary-function cases, scored vs the one extracted function. **v3** (`*3.py`, bigger +
cleaner): 19 cases capped to bug-sized diffs, localization/diagnosis scored as match
against ANY changed function in the commit — this dissolves the bundled-commit confound
that forced v2's clean/bundled split.

The overestimate ladder (diagnosis correct):

| eval set | strict | lenient (+right-theme) |
|---|---|---|
| synthetic isolated Python bugs | 8/8 = 100% | — |
| hand-picked real (AA scorecard) | 4/5 = 80% | — |
| unbiased real v2, clean-9 | 2/9 = 22% | 4/9 = 44% |
| **unbiased real v3, all-19 (match-any)** | **1/19 = 5% [CI 1-25%]** | **6/19 = 32% [CI 15-54%]** |

So the reliable strict-correct rate on unbiased real bugs is **~5-22%** (single-to-low-
double digits), not the 80% the hand-picked scorecard implied; lenient (right theme,
wrong specifics) ~32-44%. Localization (any changed function in top-6): v3 6/19 = 32%,
v2 clean 5/9 = 56%. The sole reliably-correct case in BOTH runs is `go_to_item` (empty-
mask guard) — the maximally grounded + local bug.

**Non-determinism is in DIAGNOSIS, not triage (`gate_stability.py`, 3 runs).** Retrieval
is deterministic (term-rarity + grep, no sampling). The BUG/FUNDAMENTAL gate is also
stable: 0/19 flips across 3 runs, 18/19 stably BUG (95% recall on real bugs), the one
decline (`326741c`, a race-conditional corpse filter) stably declined. So the pipeline
decomposes cleanly: retrieval (deterministic, ~32% localization) and gate (stable, 95%
BUG-recall) are the RELIABLE parts; the generative DIAGNOSIS stage is where both the
low accuracy (~5-22%) and the run-to-run variance live (the cyclic-panic case flipped
CORRECT→PARTIAL v2→v3). The weak link is generative diagnosis of complex real logic —
exactly the exact-token-generation floor the whole result turns on.

**The diagnosis floor is a CEILING, not sampling variance (`pass_k.py`, 5 samples on
the 6 localized cases).** pass@5 ≈ pass@1: sampling does NOT rescue the failures. Only
`go_to_item` is correct (5/5, reliably). The other 5 are CONSISTENTLY WRONG the same way
every sample — same fixation on "message formatting" (cyclic panic), same wrong ranged-
priority theme, same symptom-echo, same hallucinated syntax error. So best-of-k / self-
consistency won't help: when not grounded+local the model is reliably WRONG, not merely
unreliable. The v2→v3 flip was a rare borderline case, not the pattern. This is the
strongest form of the bounded claim — the floor is a genuine ceiling.

**Method artifact found:** all 5 samples on the Sokoban case hallucinated an "incomplete
line / `offse` syntax error" that isn't the bug — an artifact of the debugger truncating
each retrieved function body to 1200 chars, so the cut-off looks like broken code.
Truncating code display manufactures false positives; a real fix would feed whole
functions (bounded by the 4k context, so retrieve fewer + show them fully).

**Truncation-fix re-test forecloses the "your low rate is just truncation" objection
(`notrunc.py`).** Re-ran the 6 localized cases showing top-3 functions WHOLE. The
Sokoban `offse` hallucination vanished (artifact confirmed + fixable), and one case
improved (cyclic panic now gets the infinite-loop + terminate concept, not "message
formatting"). But 4/6 stay wrong — whole functions lift strict-correct only ~1/6→2/6.
So the 5-22% is a real capability CEILING, not a display artifact. And a sharper failure
mode surfaced: on whole code the model invented a DIFFERENT fake syntax error ("missing
closing paren") on another case — it CONFIDENTLY FABRICATES a plausible bug when it can't
find the real one. Fabrication-when-stumped is general, not a truncation side-effect.

**Localization is NOT the bottleneck — diagnosis is (`copilot.py`, perfect-localization
test).** Hand the model ONLY the exact buggy function + symptom (the realistic co-pilot:
a dev points at suspect code). Strict-correct stays 2/15 = 13% — essentially the same as
fully-autonomous (5-22%); lenient rises modestly 32%→47%. So perfect localization does
NOT rescue diagnosis: the "if only retrieval were better, the debugger would work" story
is refuted. The co-pilot points at the right AREA ~half the time but names the specific
fix only ~13%. (Bright spot: the Sokoban truncation hallucination vanished with the full
function — it found the real `'^'` char-check bug.) This CORRECTS the earlier "the
co-pilot works because a human surfaces the code" claim: surfacing the code helps the
area, not the fix. The binding constraint is BUG COMPLEXITY, not retrieval.

**Mechanism (two-factor, confirmed in both runs).** Every CORRECT diagnosis was also a
localization hit (localization is NECESSARY), but localization is NOT sufficient: in v3,
6 localized but only 1 diagnosed strictly; the localized-but-wrong cases are complex
multi-line logic rewrites (Sokoban trap-detection, --More-- parser) the LLM could reach
but not diagnose. So **correct diagnosis = grounded-symptom-localization x bug-locality**
— both required.

Failure modes, all consistent with grounding-x-locality:
1. Ungroundable symptom (0 search terms: "Fixes", "Fix RL", "monk armor habits",
   "forking") -> generic boilerplate. The exact-token hallucination floor.
2. Localized but complex logic -> vague ("the logic is incomplete") or FABRICATED
   (invented a syntax error that wasn't the bug). The complex-real-code limit, quantified.
3. Gate false-declined one real bug ("eating habits") as FUNDAMENTAL. Calibration costs recall.

## Reading

The autonomous debugger is REAL but NARROW: it reliably diagnoses only bugs that are
(a) grounded by a symptom that names/implies the locus and (b) local/simple enough to
diagnose in complex real code. On unbiased real bugs that is ~22% (strict), not the 80%
the hand-picked scorecard implied. The mechanism (grounding x locality) predicts exactly
which bugs it gets.
