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

### The unifying principle (ties the negative and the positive together)

The whole two-sided investigation reduces to one mechanism: **the LLM's floor is
hallucination in EXACT-TOKEN spaces (NetHack item names AND code identifiers), and
GROUNDING — providing the exact tokens — is what unlocks value.** The LLM adds value
exactly when the exact tokens are PROVIDED, and fails when it must GENERATE them:
- **Decisions (null result):** bare LLM must generate item names → hallucinates
  ("wand of magic resistance") → worse than the rule; retrieval-grounding → parity.
- **Debugging (positive result):** GIVEN the exact code (grounded) the LLM diagnoses
  real bugs correctly (validated co-pilot: DL1-gate, Elbereth, prayer, floating-eye,
  distractor-localization); asked to GENERATE the exact identifiers to FIND the code
  (autonomous retrieval — keyword, name-selection, and agentic search all tried) it
  hallucinates non-existent identifiers and fails. Same floor.

So the debugging co-pilot works because a human/tool surfaces the code; full autonomy
fails on the same hallucination floor as the wish decision. LLM-value ≈ f(grounding),
floored by exact-token hallucination. (Model-dependent caveat: Qwen-14B; a larger
model may hallucinate less in agentic search — future work.)

## What is NOT the LLM (the real, engineering contributions)

- **Progress engine** (hand-rule macro director + `crash_recovery` + `sokoban_fix`):
  beats base AA on ascension-progress — median DL1→DL4, mean 2.78→4.97, deepest
  9→17, starvation 45%→21%, wins 68/100, Wilcoxon p=1.9e-9. A **hand rule**, not the LLM.
  **Decomposition (abl_* batches, n~100/arm):** base med DL2 → +macro-director med
  DL5 (DL1-stuck 46%→5%) → +crash_recovery DL5 → +sokoban_fix DL5. The ENTIRE depth
  win is the macro director unsticking DL1; `crash_recovery` adds 0 depth (only
  removes the 7% crash-death tail), `sokoban_fix` adds 0 (median game never reaches
  Sokoban). So "progress engine" = one lever.
  **Integrity caveat (measured 2026-07-11):** this margin is dominated by ONE AA
  bug fix. Base AA sticks at DL1 on ~52% of seeds (n=300) — AA's `BE_ON_FIRST_LEVEL`
  milestone gates leaving DL1 on `experience_level>=8`, so seeds whose DL1 can't
  grant XL8 farm to starvation and never descend (`descend_cmd`=0, stairs visible
  from turn 1). This is AA's own known flaw (authors left a commented-out escape;
  ~42% in their notes). `apply_unstick_dl1` re-enables the escape → the 3 verified
  stuck seeds descend (DL1→DL4/DL3/DL8). So "beats AA" is largely "fixed the DL1
  starvation deadlock," a legitimate structural fix but not a novel method beating
  SOTA. After the unstick, the next wall is early-mid **combat** (68% of engine
  deaths, median XL6/DL5, underleveled diving into the Mines). That DL5 wall was
  then probed across every intervention class — all null except the DL1 unstick:
  macro/strategy (13 variants, n=1660, all median DL5), Elbereth-loop bug-fix
  (0/0/100 no-op; p=0.39), disengage/survival (p=0.16), dive-pacing (crashes AA),
  and character role (forced Valkyrie, the best ascension role, still median DL5).
  So the DL5 median is AA's fundamental mid-game combat-death rate, invariant to
  strategy/role/survival-tuning; ascension lives only in the rare tail (p90 DL8,
  p95 DL10). Baseline validated as fair (native `no_progress_timeout`: DL1-stuck
  45%, same as ours — the stick is real AA behavior, not a harness artifact).
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
