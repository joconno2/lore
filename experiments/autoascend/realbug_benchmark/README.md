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

The overestimate ladder (diagnosis correct):

| eval set | rate |
|---|---|
| synthetic isolated Python bugs | 8/8 = 100% |
| hand-picked real (AA scorecard) | 4/5 = 80% |
| **unbiased real, clean-9 (fair)** | **2/9 strict = 22%, 4/9 lenient = 44%** |
| unbiased real, all-15 | 2/15 = 13% / 5/15 = 33% |

Localization (target function in top-6 retrieved): 5/15, 5/9 clean.

**Mechanism (two-factor).** Every CORRECT diagnosis was also a localization hit
(localization is NECESSARY), but localization is NOT sufficient: 5 localized, only 2
diagnosed correctly. The 3 localized-but-wrong are complex multi-line logic rewrites
(2 Sokoban, 1 --More-- parser) the LLM could reach but not diagnose. So
**correct diagnosis = grounded-symptom-localization x bug-locality** — both required.

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
