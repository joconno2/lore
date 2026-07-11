# LORE — the complete result (two-sided, one mechanism)

Where LLMs help a symbolic SOTA (AutoAscend) in NetHack, and where they don't.
One mechanism explains both. Every number measured on trx (Qwen-14B); artifacts committed.

## The mechanism (the spine)

**LLM-value = f(grounding); the floor is failure to GENERATE exact tokens.**
The LLM adds value when the exact tokens are PROVIDED (grounded) and fails when it must
GENERATE them. NetHack item names and code identifiers are both exact-token spaces. This
one principle explains the negative, the positive, and the positive's ceiling.

## Negative — LLM null at in-game DECISIONS (rule-governed)

Every in-loop decision AutoAscend makes has a precise expert rule (deep wikis + 15K lines
of near-optimal heuristics), so the LLM's ceiling is parity and its floor is hallucination.
Measured (`LLM_VALUE_ABLATIONS.md`):

| decision | result | verdict |
|---|---|---|
| macro strategy | LLM ≈ mock, 92/100 seeds identical | null |
| tactical instadeath veto | causally inert (0/200 outcomes changed) | null |
| endgame descent action | LLM = mock = hardcoded | null |
| wish — bare LLM | 1/14 agree with rule, hallucinates items 71% | worse |
| wish — LLM + retrieval | hallucination fixed → parity, no edge | null |

"An EC-tuned LLM oracle improves a frozen symbolic SOTA at NetHack" is false: no decision
where the LLM beats the rule.

## Positive — LLM as a POST-FAILURE code diagnoser (not rule-governed)

Failure-diagnosis has no pre-existing expert rule, so the parity ceiling does not apply.
Built an autonomous LLM+grounded-retrieval debugger for the 15K-line agent: from a failure
symptom, extract terms → validate against the codebase vocabulary → rarity-rank → grep to
the function → BUG/FUNDAMENTAL gate → diagnose (`autonomous_debug5.py`). The existence proof:
from real in-game death symptoms it surfaced 2 code-verified gaps in the frozen bot, both
fixed — (1) prayer-while-adjacent: `is_safe_to_pray` (agent.py:739) checks ONLY the prayer
timeout, no adjacency, so AA prays next to a hostile and dies mid-prayer (fix cut prayer
deaths 5→1/100); (2) floating-eye-melee: `melee_monster_priority` (fight_heur.py:28-34)
applies only a SOFT −110 penalty, no hard block when other hostiles are near, so it still
melees the eye in multi-monster fights → paralysis → death. Both are real gaps in the frozen
SOTA, though depth-neutral (the DL5 wall is fundamental combat mortality, not these).

Then bounded it with an unbiased real-bug benchmark (`realbug_benchmark/`) — every
qualifying bug-fix commit from the upstream AutoAscend git history, the debugger run from
the author's commit-subject symptom. The result is REAL but sharply bounded:

**1. Diagnosis rate — the hand-picked scorecard overestimates ~4-16x.**

| eval set | strict | lenient |
|---|---|---|
| synthetic isolated Python bugs | 8/8 = 100% | — |
| hand-picked real scorecard | 4/5 = 80% | — |
| unbiased real v2 (n=9) | 2/9 = 22% | 4/9 = 44% |
| unbiased real v3 (n=19, match-any) | 1/19 = 5% [CI 1-25%] | 6/19 = 32% |

Reliable strict-correct is ~5-22%. Methodological result: synthetic benchmarks overestimate
LLM debuggers; use unbiased git-history bugs.

**2. The bottleneck is generative DIAGNOSIS, not retrieval or triage.**
Pipeline decomposition (`gate_stability.py`): retrieval is deterministic (~32% localization);
the gate is stable (0/19 flips, 95% BUG-recall). The variance and the low accuracy live
entirely in the generative diagnosis stage. Co-pilot test (`copilot.py`) — hand the model
the EXACT buggy function (perfect localization): still 2/15 = 13% strict / 47% lenient ≈
autonomous. So surfacing the code does not rescue diagnosis; better retrieval would not fix
it. The tool points at the right AREA ~45% of the time but names the exact FIX ~5-13%.

**3. The ceiling is real, not sampling or truncation.**
pass@5 ≈ pass@1 (`pass_k.py`): failures are consistent, not random — best-of-k won't help.
Whole functions (no truncation) still leave 4/6 wrong. When stumped the model CONFIDENTLY
FABRICATES a plausible bug (invents a syntax error that isn't there).

**4. Zero standalone bug-DETECTION, and it cannot SELF-FILTER.**
Neutral prompt offering "NO BUG", run on buggy AND fixed versions (`discrim.py`):
P(BUG|buggy) = P(BUG|fixed) = 1/15, discrimination 0.00. Offered the option it says "NO BUG"
~93% regardless of ground truth. The 5-22% is recall only when a failure is ASSERTED — the
symptom is itself part of the grounding. So it is a POST-FAILURE diagnoser (you invoke it
because the bot died), never an autonomous bug finder. And its confidence is UNCALIBRATED
(`copilot_conf.py`): mean self-confidence WRONG 61 ≥ CORRECT 55 (values cluster ~37-75
regardless of correctness) — it is confidently wrong and sometimes underconfident when
right, so it cannot flag its own correct diagnoses. Deployment consequence: a candidate
generator requiring EXTERNAL verification (run the fix / human review), not a trustworthy
autonomous diagnoser; ~13% of its confident outputs are the real fix and it can't tell which.

**5. Domain-general, not an AA artifact.**
Same co-pilot protocol on `rich` (pure-Python terminal rendering, different domain;
`generalization_rich/`): 1/20 = 5% strict / 45% lenient, matching AA's 13%/47%. rich's misses
are overwhelmingly EXACT-TOKEN edits (add `~` to a URL regex, `==`→`is`, off-by-one
`len(text)-1`, `assert`→default) — the model sees WHERE but not WHAT-exactly, the same floor
as hallucinated NetHack item names and the Sokoban `'^'` constant.

**6. The ceiling is GENERATION, not knowledge — recognition 67% >> generation 13%
(`recognize2.py`).** Forced-choice (real fix + 3 distractors from other cases, chance 25%):
the model picks the correct fix 10/15 = 67%, versus 2/15 = 13% when asked to GENERATE it.
So the diagnostic knowledge is largely present; the floor is producing the exact tokens.
(Yes/no fix-verification collapses to the same NO-bias as detection — 0/15 accept real AND
wrong — so the usable form is forced-choice, not binary judgment.) Two consequences: (a) a
bigger model with better exact-token generation plausibly closes the 13%→67% gap — the
GPU-blocked bigger-model test is worth running, not a long shot; (b) deployed as a candidate
RANKER (tool proposes fixes, LLM picks) it hits ~67%, a genuinely usable tool where candidates
can be sourced — versus 13% open generation. But the ranker needs EXTERNAL candidates: a
self-contained generate-then-rank pipeline is bounded at ~13% because neither temperature
diversity (pass@5 ≈ pass@1) NOR prompt diversity (5 distinct framings, recall@5 ≈ 13%;
`diverse.py`) escapes the generation floor — different framings still fail to produce the
exact token (they trade cases: one framing recovers a bug single-prompt missed but loses
another). So the 67% ranker is human-in-the-loop (or tool-fed candidates), not autonomous.

**Net:** the positive is a *general, post-failure, grounded fix-suggester* — ~5-22% strict open
generation but ~67% as a candidate ranker, ~45% right-area, cross-domain, ceiling in exact-token
GENERATION (knowledge largely present). Real and usable as a ranking co-pilot; not the
"autonomous debugger beats the symbolic SOTA's weakness" overclaim.

## Two-factor law (predicts which bugs it gets)

**Correct diagnosis = grounded-symptom-localization × bug-locality.** Localization is
NECESSARY (every correct diagnosis localized) but NOT sufficient (perfect localization still
13%); the binding constraint is bug-simplicity / exact-token — the model handles 1-token/local
bugs (empty-mask guard, inverted condition, wrong constant) and fails complex multi-line
rewrites. The failure assertion (symptom) is part of the grounding; without it, detection
collapses to chance. Same mechanism as the negative: grounding unlocks, exact-token generation
floors.

## Engineering / the ascension wall (non-LLM, banked)

- **DL1-stick** is AutoAscend's own dominant flaw: the `BE_ON_FIRST_LEVEL` milestone gates
  leaving DL1 on `experience_level>=8`, so ~52% of games (baseline validated fair) farm DL1 to
  starvation. `apply_unstick_dl1` is the ONE lever that moves depth (base median DL2 → DL5).
- The **DL5 wall** past it is fundamental mid-game combat survival — invariant across 13 macro
  variants (n=1660), forced Valkyrie, disengage, AA-parameter-tuning, and 4 targeted death-mode
  bug-fixes (depth-neutral, p=0.16-0.93). The one lever that "crashed" rather than cleanly
  failing (dive-pacing) is now root-caused: the gate freezes the agent on the downstairs →
  AA's inactivity/cyclic-panic guard trips; AA descends because the level's XP is exhausted, so
  pacing cannot grind absent XP (a faithful fix = AA level-selection redesign). Wall airtight.
- Endgame via wizard `^V` is bounded (confined ~54-cell no-dig pockets, planes unreachable).
  Not legitimate ascension progress.

## The paper

"LLMs can't out-DECIDE a rule-saturated symbolic SOTA; a grounded LLM pipeline is a useful
POST-FAILURE fix-suggester for it, within a sharp envelope (5-22% strict, ~45% area,
cross-domain, zero standalone detection), and correct diagnosis = grounded-localization ×
bug-locality." One mechanism — grounding unlocks, exact-token generation floors — ties the
null decisions, the bounded diagnoser, and the ascension wall together. The contribution is
the ENVELOPE, mapped, adversarially verified, and mechanistically explained on a real 15K-line
agent and confirmed cross-domain: negative + bounded-positive + the predictive two-factor law.

## Open (Jim-gated)

- Does a **bigger local model** raise the diagnosis ceiling, or is exact-token generation a
  hard floor? Currently GPU-blocked on trx (Jim's `rejection_sampling` owns the card).
- **Framing/pivot decision** — rests on this bounded, cross-domain result.

Detail: `Agent/daily/2026-07-11.md`. Code: `experiments/autoascend/` (oracle.py, wish_grounded.py,
autonomous_debug{4,5}.py, `realbug_benchmark/` + `generalization_rich/`, the *_fix/*_safety patches).
