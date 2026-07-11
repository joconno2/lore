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

**Envelope:** clear/local bugs reliable (4/4 localized), fundamental correctly
declined (calibration gate), subtle cross-file value-tracing fails (an LLM reasoning
limit — fails even when both code sites are shown; may be model-size-dependent).
Retrieval had to be GROUNDED: LLM-generated search terms (keyword / name-select /
agentic) all hallucinated non-existent identifiers and failed — the same floor as
the wish decision. Grounding the search (real symptom-terms) unlocks it. The LLM
also **co-found 2 novel bugs** this way (prayer, floating-eye), both fixed.

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

"LLMs can't out-DECIDE a symbolic SOTA in a rule-governed game, but a grounded
LLM pipeline can autonomously DEBUG it (find real bugs, decline fundamental ones)."
One mechanism (grounding unlocks; exact-token hallucination floors) ties the null
decisions and the working debugger together. Negative + positive + a working,
characterized artifact.

Detail: `Agent/daily/2026-07-11.md`. Code: `experiments/autoascend/` (oracle.py,
wish_grounded.py, llm_debug_*.py, autonomous_debug{4,5}.py, the *_fix/*_safety patches).
