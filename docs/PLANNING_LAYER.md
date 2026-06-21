# LORE planning layer: wiki-grounded long-horizon plans over frozen AutoAscend

Decision (Jun 21): single-action patches are a dead end. Five reflex-level
interventions (descent gate, melee veto, heatmap repulsion, broad override,
pray-at-WEAK) all came back neutral-to-negative -- AutoAscend's reflexes are
already good, so poking single actions disturbs more than it fixes. The only win
(sokoban crash patch, +11%) worked because it fixed a *structural* gap, not a
reflex.

The contribution is the structural gap AutoAscend actually has: **no persistent,
knowledge-grounded, long-horizon planning.** Its `current_strategy` is a fixed
linear milestone route; `GO_DOWN` (endgame) is an unimplemented TODO; it has no
notion of "I am food-poor and deep, so for the next 500 turns my objective is to
stockpile food and get to a shop." LORE supplies that layer.

## Thesis

An EC-optimized, wiki-grounded LLM **planner** that maintains a persistent goal
stack over a frozen SOTA symbolic base, producing long-horizon behavior the base
structurally cannot. The LLM does open-ended strategy (where heuristics fail);
the wiki KB makes the goals NetHack-correct; EC tunes the planning/retrieval
interface (MAP-Elites over what-to-retrieve / when-to-replan / goal-priority).

## Components

1. **Plan store (persistent).** A prioritized stack of Goals living on the agent
   across turns. Each Goal: id, params, priority, entry condition, exit/success
   condition, abandon condition, status. Survives until satisfied/abandoned.

2. **Goal library (executable, grounded).** Each goal maps to multi-turn behavior
   via AutoAscend's existing strategies + parsed-JSON facts. Initial set:
   - `stockpile_food(min_nutrition)` -- proactively eat safe corpses (corpse_effects.json),
     keep rations; the real starvation fix (not prayer).
   - `acquire(item_class)` -- gloves / ranged / MR source / unicorn-horn: seek shops/altars, pick up, equip.
   - `gear_to(ac_threshold)` -- wear_best_stuff until AC below target.
   - `reach(branch, level)` -- go_to_level (Mines/Minetown/Sokoban/Quest).
   - `consolidate(xl_for_depth)` -- farm current level until XL >= f(depth).
   - `avoid_class(monster_set)` -- standing distance policy (petrifiers, floating eye).
   - `descend_when(ready_predicate)` -- gate the dive on a readiness checklist.
   Goals are the EC genome's vocabulary; priorities/conditions are tunable.

3. **Knowledge retrieval.** Two tiers:
   - Structured: direct lookups in `data/parsed/*.json` (monster speed/AC/MR,
     corpse safety, item stats, prayer mechanics) for exact preconditions.
   - Prose: an index over `data/wiki/pages` + `data/guides` (BM25 to start;
     embeddings later) so the planner pulls situational strategy passages.

4. **Planner (LLM oracle).** Input: compact game state + retrieved knowledge +
   current plan. Output: revised goal stack (add/reprioritize/abandon goals).
   Runs on re-plan events, not every turn (bounded; the cadence is an EC knob).

5. **Executor.** A high-priority preempt on `global_strategy`. Each turn it runs
   the highest-priority active goal whose entry condition holds, dispatching to
   that goal's AutoAscend behavior. Falls through to the base when no goal is
   active. One reactive step per turn, like the override, but goal-driven and
   persistent rather than stateless.

6. **Monitor / re-planner.** Marks goals satisfied/abandoned from state; triggers
   a re-plan on goal completion, new threat class, milestone change, or timeout.

7. **EC layer (later).** MAP-Elites / CMA-ES over the interface: retrieval depth,
   re-plan cadence, goal-priority weights, override-aggressiveness. Fitness =
   score / depth / ascension. Frozen base + frozen LLM; EC evolves the interface.

## Why this beats the patches

A goal persists and shapes hundreds of turns toward a coherent objective, so it
can't be undone by the next reflex, and it targets the structural gap (no
long-horizon plan) instead of fighting the base's good reflexes. Starvation
becomes "stockpile food for the next stretch," not "pray now"; petrification
becomes "acquire gloves, then exploit cockatrice corpses," not "dodge."

## Build sequence

1. Plan/Goal data model + executor seam (preempt), with 1-2 hardcoded goals
   (stockpile_food, gear_to) to prove a persistent goal drives multi-turn
   behavior and beats the base on the starvation seeds. No LLM yet.
2. Add structured + prose retrieval; wire the LLM planner to emit the goal stack.
3. Broaden the goal library; measure on the 50-ep cohort vs +11% baseline.
4. EC over the interface.

Target: CoG 2027.
