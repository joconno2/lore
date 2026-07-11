# LORE — does an LLM oracle beat symbolic rules in NetHack? (measured ablations)

Complete, measured evidence on the project's core premise: an EC-tuned LLM oracle
layered on frozen AutoAscend (symbolic SOTA) improves play. Every LLM-vs-rule
ablation, one table. Verdict up front: **no** — the LLM is null-to-worse at every
decision type, because NetHack is rule-governed (deep wikis + a locally-optimal
SOTA), so an expert rule captures the optimum and the LLM's ceiling is parity.

## The decision ablations (LLM vs symbolic rule/expert)

| Decision | Setup | Result | Verdict |
|---|---|---|---|
| **Macro strategy** (which branch objective) | n=100 paired, same seeds, real oracle | LLM ≈ mock (median DL5 both; LLM slightly worse, more combat deaths). Richer state (AC/kit/threat) → 92/100 seeds identical | **null** |
| **Tactical instadeath veto** (petrify/paralysis) | counterfactual replay, perfect-knowledge mock | veto causally inert (fired 127/200, changed outcome 0/200); perfect-knowledge veto 3W/4L | **null** |
| **Endgame descent action** (dig/stairs/explore) | scenario, LLM vs mock | LLM == mock == hardcoded (descending is knowledge-trivial) | **null** |
| **Wish selection — bare LLM** | Qwen-14B vs adaptive expert rule, n=14 states | agreement 1/14 (7%); **hallucinates non-existent items 10/14 (71%)** ("wand of magic resistance", "wand of speed boost") | **worse** |
| **Wish selection — LLM + retrieval** (the actual premise) | grounded prompt (valid-items list), n=14 | hallucination 0/14 (fixed); agreement 9/14; remaining diffs mostly equivalent (cloak-MR vs dragon-MR) + 2 mis-prioritizations | **parity, no edge** |

## Why (the mechanism)

NetHack is **rule-governed**: expert strategy is documented exhaustively (wikis),
item names/effects are exact, and AutoAscend already encodes 15K lines of
near-optimal heuristics. On any decision with a known optimal rule:
- the LLM's **ceiling is parity** (it can at best reproduce the rule), and
- its **floor is hallucination** (inventing invalid items/actions in an exact-name domain).

Retrieval grounding removes the floor (fixes hallucination → parity) but cannot
create an edge above the rule, because the knowledge is already in the rule/wiki.
This is not a model-size problem — no LLM beats a rule that already captures the
optimum on a rule-determined decision.

## What is NOT the LLM (the real, engineering contributions)

- **Progress engine** (hand-rule macro director + `crash_recovery` + `sokoban_fix`):
  beats base AA on ascension-progress — median DL1→DL4, mean 2.78→4.97, deepest
  9→17, starvation 45%→21%, wins 68/100, Wilcoxon p=1.9e-9. A **hand rule**, not the LLM.
- **Endgame capability extension** (structural crash-fixes + iterative-teleport +
  genocide survival + fixed invocation ritual): the layer reaches DL49, survives
  1479 turns in Gehennom, and executes the invocation ritual — all milestones AA
  hits 0% of (AA can't even hold the endgame kit: `parse_text` asserts). All
  **engineering/scenario tooling**, not LLM-driven.

## Implication (the pivot)

The premise "an EC-tuned LLM oracle improves a frozen symbolic SOTA at NetHack" is
empirically unsupported end to end: no decision-value (measured), and the capability
extensions don't use the LLM. Options:
- (a) Drop the LLM; frame the contribution as a **symbolic endgame-extension** of a
  frozen SOTA (progress engine + capability ladder) — real engineering.
- (b) A fundamentally **different, non-decision LLM role** outside AA's loop (the
  in-loop decision role is closed).
- (c) A **different domain** that is NOT rule-governed, where LLM knowledge beats
  the absence of a rule (NetHack is the wrong domain for the oracle thesis).

Arc + numbers: `Agent/daily/2026-07-11.md`. Ablation code: `experiments/autoascend/`
(`aa_profile.py`, `oracle.py`, `wish_cmp2.py`, `wish_grounded.py`). Commits → 531863e.
