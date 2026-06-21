# AutoAscend gap analysis (frozen base), 2026-06-20

50 episodes, NetHackChallenge-v0, seeds 100-149, `aall/autoascend:frozen` on trx.
Data: `results/bench_2026-06-20/autoascend_gap_50ep.json`. Harness: `aa_gap.py`.

## Headline

mean 14,283 / median 9,139 / max 74,792 / min 324.

## Where it ends (milestone reached at episode end)

| Milestone | Episodes |
|-----------|----------|
| BE_ON_FIRST_LEVEL | 17 |
| FIND_SOKOBAN | 14 |
| FIND_MINETOWN | 9 |
| FIND_GNOMISH_MINES | 6 |
| SOLVE_SOKOBAN | 4 |
| (anything past Sokoban) | **0** |

**0 of 50 episodes advance past Sokoban in the scripted milestone route.** Some
runs reach DL15-18 by depth, but the strategy layer has no endgame milestones
driving them (`GO_DOWN` is an unimplemented TODO). AutoAscend dives without a
plan once the scripted route runs out. This is the score ceiling and the reason
it does not ascend.

## How it dies (end reasons)

- Monster deaths, spread thin across ~30 species (white unicorn x6 is the mode).
  No single dominant killer -> mostly "fair" deaths from inherent difficulty.
- **5/50 (10%) are `AssertionError` crashes** in AutoAscend's own item/parse
  logic, not game deaths. 10% of episodes lost to bugs.
- Petrification (cockatrice/chickatrice) x3 -- knowledge-dependent instadeaths.

## Friction (panics: 11,197 total, ~224/episode)

Dominated by low-level state-desync, all recovered-from:
- "items below me changed" 1720
- "end point is no longer accessible" 1313
- "no such food is lying here" 1084
- "position changed" 406
- "Monster on a next tile when moving" / "position do not match after move" (many)

These are mechanical state-tracking issues, not decision points. Not LLM-oracle
targets (an LLM cannot help with "item below me changed"). They cost steps, not
games.

## Ranked oracle targets

1. **Post-Sokoban / deep-game strategy (highest headroom).** 0/50 get a real
   endgame plan. The oracle supplies high-level strategy for the deep dungeon
   (branch choice, when to dive vs consolidate, quest, altar/gear goals). Targets
   the score ceiling and ascension directly. This is the paper.
2. **Instadeath veto at knowledge-dependent points (targeted).** Petrification,
   corpse safety, unicorn/shopkeeper anger. Oracle as a warn/veto on provably
   fatal moves. Individually rare but high-cost.
3. **Crash robustness (cheap, non-research).** 10% AssertionError rate -> a
   try/except wrapper that ESCs and continues recovers ~free score. Worth doing
   but it is code-hardening, not oracle work.

## Intervention #1 result: Sokoban crash patch (lore_patches.py)

Runtime monkeypatch (no base edit) routing the Sokoban-solver desync into
AutoAscend's own abandon path. 50 ep, same seeds:

| | baseline | patched |
|---|---|---|
| mean | 14,283 | 15,850 (+11%) |
| median | 9,139 | 9,139 |
| max | 74,792 | 78,302 |
| AssertionError crashes | 5 | 2 |
| milestone reach | caps at SOLVE_SOKOBAN | 4 reach FIND_MINES_END |

First episodes to advance past Sokoban. Recovered 3/5 deep crashes (one seed
24k->78k). Residual: 2 different brittle asserts (turn-inactivity, chest-open) ~1
episode each -- diminishing. Median unmoved: crashes were never the median run's
problem. Architecture (wrap, don't edit) validated.

## Next oracle target (from this data)

- Tactical instadeath veto: **white unicorn kills 6/50 (12%)**, petrification 3.
  High-frequency, knowledge-dependent, doesn't require reaching the deep game.
  Best first oracle target.
- Strategic endgame: still 0 past Mines End. Bigger ceiling, but needs the agent
  to reliably get deep first.

## Intervention results (Jun 20-21) and the action-injection wall

Two interventions beyond the sokoban patch were built and measured. Both taught
the same architectural lesson.

- **Descent gate (oracle-gated descent timing): DEAD LEVER.** 50-ep, conservative
  policy: mean 15,850 -> 6,325 (-60%), depth 5.7 -> 2.6. AutoAscend's descent
  pacing is already near-optimal; forcing it to linger halves depth and starves
  it (11 faint deaths). Also the hold-via-AgentPanic mechanism is unsafe: 29/50
  died of Cyclic Panic (5 consecutive panics). Abandoned. Takeaway: the oracle
  must not re-tune decisions the base already makes well.

- **Petrification melee veto: FIRES but INSUFFICIENT.** Telemetry confirms it
  fires every time (veto_query=veto_fired=103 over 3 eps), so the earlier "wrong
  layer" call was the telemetry bug, not the hook. But 2/3 still petrified:
  vetoing the agent's melee doesn't stop the cockatrice's OWN adjacency attack.
  Stopping a bad action != taking the good one.

**Architectural wall:** veto/priority hooks are easy (return a low priority) but
insufficient. Effective tactical interventions require ACTIVE action injection --
make the agent flee / keep distance / engrave Elbereth / fire ranged -- into
AutoAscend's action loop. That is the real next build for tactical instadeaths.

**Telemetry lesson:** AutoAscend's StatsLogger.log_event raises KeyError on
unknown names; a swallowing try/except hid it and produced false "didn't fire"
nulls twice. Always verify telemetry before trusting a null. Fixed via
lore_patches.COUNTERS.

## ROOT CAUSE of deep-run deaths: starvation death-spiral (Jun 21, via tracing)

Diagnosed seed 107 (74k, DL14, the best run) by tracing actions+messages to death.
The "Petrified by a cockatrice" end_reason is proximate; the real killer is
**starvation**:

```
turn 65632  "You faint from lack of food"
turn 65656  "You are too disoriented for this"   (can't act)
turn 65657  kills a naga hatchling but can't eat it (fainting)
turn 65684  "You faint from lack of food"
turn 65705  petrified while incapacitated
```

This is why all 3 petrification hooks (melee veto, heatmap repulsion) did nothing
-- the agent isn't choosing to engage anything; it's incapacitated by hunger and
can't avoid anything. Identical scores under every petrification intervention
confirmed zero behavioral effect.

**Scale:** 15/50 deaths explicitly mention faint/starvation (undercounts -- "faint
then killed by X" logs as a monster death). Among deep deaths (DL>=8, the high
scores): 107/143/106/103 all die "while fainted"/"while praying". Starvation is
the single biggest non-crash killer, and it kills the runs that matter most.

**Mechanism:** AutoAscend emergency-eats/prays only at FAINTING (agent.py:1436,
1445). By then it's disoriented and can't act. With no inventory food and prayer
on cooldown (~400-1000 turn limit), it enters a faint loop and dies. It acts one
hunger-stage too late. Fix direction: act at WEAK (before disorientation) +
better food sourcing/conservation on long runs. Highest-value target.

## Implication for the design

The thesis holds with data: AutoAscend's weakness is the absent endgame (0/50
past Sokoban) plus specific knowledge-deaths, not its early/mid-game mechanics.
The LLM oracle's home is high-level deep-game strategy (#1), with instadeath veto
(#2) as a tactical second target. EC optimizes when/what to query at those points.
