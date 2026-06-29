# LORE status & decision package — 2026-06-29 (autonomous session)

## TL;DR
Exhaustive, rigorous testing (paired, crash-matched, perfect-knowledge mocks as
controls) shows **an LLM oracle cannot beat AutoAscend on score via better
decisions** — AA is near-decision-optimal and the LLM *agrees* with it (97% on
tactics; identical to the perfect rule on petrifier-avoidance). The **one clean
SOTA-beat is structural: `crash_recovery`, +2% mean / +8% p90, 14W/0L (n=250)**,
fixing AA's own crashes that kill its high-value games. A knowledge instadeath-
veto lifts the high tail (+24% p95) but is a trivial hardcodable rule (the LLM
just reproduces it) and nets ~0 on the mean (rarity + path-divergence collateral).

## What was tested (all null/loss for LLM decision-improvement)
| lever | result |
|---|---|
| tactical threat-veto (cockatrice etc.) | NULL even at perfect knowledge |
| survival / "pro who wouldn't die" disengage | NULL; LLM picks FIGHT 97% (agrees w/ AA) |
| food-economy oracle | LOSS (over-eats, wastes game) |
| descent-timing / unstick DL1 | LOSS even at perfect-knowledge ("precise") |
| LLM-driven endgame descent | NULL (descending is knowledge-trivial) |
| petrification veto | marginal; trivial rule (LLM == mock); lifts p95 but collateral |
| **crash_recovery (structural)** | **WIN: +2% mean, +8% p90, 14W/0L** |

## Why the LLM can't win on decisions (the core finding)
1. **AA is near-optimal where it acts.** Across every danger/decision point, the
   LLM (and perfect knowledge) either agrees with AA or does worse. NetHack
   decisions AA makes are already well-tuned in 15K lines of heuristics.
2. **AA's "macro flaws" are score-efficient.** It farms DL1 to XL8 and often
   starves (42% of games, median DL2-3) — but that farming *banks XP-score*, and
   the high-scoring games come from farm-then-descend-strong. Forcing earlier
   descent (even perfect "descend only the doomed") LOSES score: the doomed,
   food-poor seeds just die to combat at similar-low score, and any disruption
   caps the strong tail. The doomed/productive split isn't observable from state.
3. **Score lives in a fragile high-value tail** (mean 9.4k, median 5.5k, max
   107k). The tail dies to AA's own crashes (18/250, mean 22k) and knowledge-
   gated instadeaths (petrif/paralysis, 11/250, mean 17k incl one 79k). Preserve
   the tail → crash_recovery (structural) + instadeath-veto (knowledge).

## Assets produced
- **`crash_recovery`**: clean +2%/+8%-p90 SOTA-beat (structural).
- **Behavioral dataset**: `prof_*.json` n=250+ on trx; `aa_profile.py`,
  `aggregate_profile.py`, `compare_arm.py`, `final_table.py`, `tails.py`.
- **Macro characterization**: `docs/AA_MACRO_GAPS.md`.
- **Oracle stack**: threat/survival/food/unstick/descent queries (oracle.py),
  validated against live Qwen2.5-14B-AWQ (vLLM on trx).
- The rigorous **negative result** (LLM can't improve a strong symbolic agent's
  decisions) — itself a finding, if reframable for venue.

## Decision needed (Jim) — the two-fold bar isn't met by decision-improvement
Options:
1. **Reframe the contribution** to what the data supports: a large-scale
   behavioral characterization of the symbolic SOTA + structural robustness
   (`crash_recovery`) + the rigorous "LLM-can't-out-decide-a-tuned-symbolic-SOTA"
   result. (Honest; not a "beat SOTA with LLM" headline.)
2. **Change the LLM's role** off live decision-control (where it's null) — e.g.
   offline heuristic/parameter synthesis, curriculum, or a weaker base/variant
   where the symbolic SOTA leaves real room.
3. **Change base/target** — a setting where an LLM's world-knowledge is decisive
   and no 15K-line expert system already encodes it.
