# LORE pivot brief — both axes of the two-fold bar fail; decision needed

One page for the pivot call. The bar was: beat SOTA (AutoAscend) AND via a novel
LLM angle, measured on ascension-progress. Tonight's measurements close both axes
on the current framing. Options below; the evidence for each is in
`docs/LLM_VALUE_ABLATIONS.md` and `Agent/daily/2026-07-11.md`.

## What is established (measured, not argued)

**Axis 1 — LLM adds no decision value.** Null-to-worse at every in-loop decision:
macro (LLM=mock, 92/100 identical), tactics veto (causally inert), endgame-seq
(LLM=mock), wish selection (bare LLM 1/14 agree with the expert rule, 71%
hallucinates non-existent items; +retrieval → parity, no edge). Mechanism: NetHack
is rule-governed (deep wikis + a locally-optimal SOTA), so any knowledge-rich
decision already has a precise rule; the LLM ceiling is parity, floor is
hallucination. Not a model-size problem.

**Axis 2 — no engineering path to ascension either.**
- The benchmark baseline was broken: base AA sticks at DL1 on 52% of seeds (its
  own `BE_ON_FIRST_LEVEL` XL≥8 milestone gate → farms to starvation, never
  descends). The "progress engine beats AA p=1.9e-9" win decomposes to ONE bug
  fix (the DL1 unstick); crash_recovery/sokoban_fix add 0 depth.
- Legit play walls at median DL5 across every lever: macro/soko/branch-order all
  median 5; descent_gate crashes AA (50/60, AgentPanic-for-flow-control);
  disengage null (Wilcoxon p=0.16). Depth is bimodal — 31% die by DL3 (XL4), only
  5% reach DL10+. The barrier is early-mid combat survival, which AA is already
  SOTA at. AA has no "grind for XP" behavior, so pacing the dive needs new
  capability AA deliberately lacks — not a lever tweak.
- "Closer to ascending" exists only in wizard-`^V` scenario tooling (place a
  kitted agent at DL27-49) — not legitimate play. (CORRECTION 2026-07-11: the
  earlier "invocation levels are sealed no-dig pockets" claim was an instrumentation
  artifact — AA's level model is empty right after a wizard teleport, so the BFS
  saw no frontier. Deep levels ARE navigable: the working explorer traverses 167
  cells. The invocation-demo blocker is SURVIVAL at DL47-50, not topology; retesting
  with genocide now. This may reopen the real-invocation capability rung.)

## The pivot options

- **(a) Drop the LLM; frame as a symbolic endgame-extension of a frozen SOTA.**
  WALLED. Legit play can't pass DL5; the deep-endgame capability is scenario-only
  (wizard teleport), not real ascension progress. No defensible "much closer to
  ascending" claim from legit play. Not recommended without a major new capability
  build (a real Gehennom nav + early-game survival — the 15K-line problem).
- **(b) A fundamentally different, non-decision LLM role.** In-loop decisions are
  closed. Candidate: LLM as a generator in an outer QD/MAP-Elites loop (propose a
  diverse population, evaluate by rollout, archive) — connects to AALL methods
  (MAP-Elites, QD-Continual). The LLM proposes, rollouts judge; it never makes an
  in-game decision, so the rule-parity ceiling doesn't apply. This is the most
  AALL-leveraged direction — BUT with a measured constraint: the generation target
  can NOT be AA-macro-strategy. That space is already exhausted — 13 macro variants
  across n=1660 games (prof_mcr, be_engine, mabl/mablf, bo_soko, soko1, abl_*,
  surv, prof_msf/msp) are ALL median DL5. QD/LLM search over macro-policy inherits
  the DL5 wall (early-game combat survival, not policy, is the ceiling). Option (b)
  has legs only if the LLM generates something OUTSIDE the walled spaces — not
  in-loop decisions (rule-parity), not macro-policy (DL5-walled). What that target
  is (curriculum for a learned agent? behavior-space exploration where "beat AA"
  isn't the metric? level/scenario generation?) is the real (b) design question,
  and it likely means redefining the goal away from "beat AA on ascension."
- **(c) A different, non-rule-governed domain.** NetHack is the wrong domain for
  the oracle thesis (exact-name, wiki-documented, SOTA-saturated). A domain where
  no precise rule exists is where LLM knowledge could beat the absence of a rule.
  Larger scope change.

## Recommendation posture

Not picking for you. If the goal stays ascension, (a) needs a real capability
build and (b)/(c) leave NetHack-ascension behind. If the goal is a publishable
result about LLMs-on-SOTA, the measured negative (Axis 1) + the broken-baseline
finding are already a clean paper; (b) is the natural positive-result direction
and the one that uses AALL's own methods. Your call.
