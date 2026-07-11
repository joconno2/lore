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

**Non-determinism (v2 vs v3):** the cyclic-panic case scored CORRECT in v2 and PARTIAL
in v3 — same input, temp 0.2, flipped. The debugger is not deterministic; the true rate
has run-to-run spread on top of the sampling CI.

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
