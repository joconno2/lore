# LORE ascension capability ladder — 2026-07-11

The two-fold bar (Jim): beat the symbolic SOTA (AutoAscend) AND do it with a novel
LLM angle, not hard-coding on AA. Metric = ascension-progress. This doc is the
paper-ready summary of the ENDGAME half: milestones AutoAscend structurally cannot
reach, and what the LORE scenario layer has proven, with the current gap.

## Why the endgame is the LLM angle

The progress-engine half (macro director) beats base AA on progress but is a
hand-rule — the LLM adds no edge there (valid null: LLM ≈ mock, richer state 92%
identical). The endgame is different: AutoAscend has **zero code** for it. Its
`GO_DOWN` milestone is an unimplemented TODO; it never assembles the ascension kit,
never performs the invocation, never enters the planes, never ascends. So driving
the endgame sequence is NEW capability, not rebuilding AA — this clears the
integrity bar (fix AA's bugs structurally; only claim the LLM for what AA lacks).

## The ladder (each rung: AA's rate → LORE status)

AA rates are from the n=250-450 behavioral profile (our NetHackChallenge-v0
distribution), where AA reaches median DL2-3, deepest ~DL17, and 0% of any endgame
milestone.

| # | Rung | AA | LORE status (scenario harness, wizard placement + wished kit) |
|---|------|----|--------------------------------------------------------------|
| 1 | Hold invocation kit (Bell, Candelabrum+7 candles, Book of the Dead) | 0% | **PROVEN** — items wishable; AA's `parse_text` ring/amulet-worn crash fixed (monkeypatch, AA stays frozen) |
| 2 | Perform the invocation RITUAL (attach candles→light candelabrum→ring Bell→read Book) | 0% | **PROVEN** — executes via low-level keypresses (bypasses AA's assert-on-unknown-items); candelabrum ends "(7 candles, lit)" |
| 3 | Reach Gehennom, equipped and alive | 0% | **PROVEN** — teleport lands in Gehennom (dungeon_num=1, DL28-29); tank kit (XL30, AC-15) survives in place; food/choke handling fixed |
| 4 | Reach the vibrating square (fire the REAL invocation) | 0% | **GAP — survival-gated, NOT topology-dead (corrected Jul 12).** Wizard `^F` level-reveal on 7 ^V-Gehennom levels: downstair REACHABLE on 4/7, WALLED (behind a wall, dig across) on 2/7, absent on 1/7. So the "degenerate sealed pocket" call was wrong — the level is navigable; the agent just dies first. Deepest legit dive so far **DL42** (seed 46, dig-down-fast). |
| 5 | Take the Amulet from Moloch's Sanctum | 0% | not started (blocked on #4) |
| 6 | Ascend the 4 Elemental planes | 0% | not started |
| 7 | Astral plane #offer → ASCEND | 0% | not started |

## The gap at rung 4 (root-caused Jul 12, instrumented + measured)

Reaching the vibrating square is the sole blocker to the real invocation.

**Topology is NOT the wall (corrected).** `asc_reveal.py` teleports, fires wizard
`^F` (wiz_map) to reveal the whole level, then checks for a downstair glyph and its
BFS-reachability. Across 7 seeds at DL28: 4 DOWNSTAIR_REACHABLE (reach_frac 0.82-0.99),
2 DOWNSTAIR_WALLED (downstair in a wall-separated region, reach_frac 0.28-0.47 → dig
across), 1 no-downstair anomaly. The downstair EXISTS and is reachable on the
majority — the earlier "degenerate ~54-cell sealed pocket" reading was an artifact of
never covering the level. So rung 4 is **survival + coverage**, not a dead level.

**Survival is the wall, and it has concrete root causes:**
- **Heal reflex never fired (FIXED Jul 12).** Wished potions come in UNIDENTIFIED —
  "8 potions of full healing" display as "8 black potions". The old setup skipped
  healing potions by NAME during the gain-level quaff loop → the skip failed → the
  tank drank its own healing during setup (healing_kept 0), and the reflex's
  name-match never found them in Gehennom. Fix: quaff gain-level potions FIRST
  (before healing exists), then wish healing and track the stack by LETTER. Now
  healing_kept=8 and the reflex fires.
- **Deaths are multi-modal.** Batch (hardcoded policy, 6 seeds): illness ×2 (setup
  corpse-eating for poison-res; HP potions can't cure — needs prayer), demon bursts
  (marilith, horned devil), invisible hitters, chip-while-searching. Only ~1 slow-HP
  death per batch, so HP-healing alone is not enough.
- **Loop-top reflexes don't fire (measured null).** Adding fight/pray/heal checks at
  the top of the descend loop did nothing (reflex_fights=0, prays=0): the damage
  lands DURING AA's multi-turn primitives (explore1/go_to/search paths), and the
  loop only re-checks BETWEEN them, by which point the agent isn't adjacent to the
  threat. **The reflexes must run per game-step, not per primitive.**
- **Dive-fast works.** The deepest legit dive (DL42, seed 46) came from digging down
  every level (minimal time-on-level). No-dig levels force exploration, where the
  agent gets caught. So the strategy is: dig-down > beeline-to-known-stair > explore,
  with per-step survival, minimizing exposure.

**Next build (the holistic Gehennom descent agent):** a PER-STEP controller that
replaces AA's multi-turn primitives with single-step actions so survival reflexes
(fight adjacent / heal / cure-illness / selective flee) run every turn, wrapped
around a dive-fast policy (dig-down, else beeline the revealed downstair). Every
piece is now de-risked and instrumented; the open work is the per-step rewrite +
multi-seed validation. Tooling: `asc_reveal.py` (topology), `descent_run.py`
(instrumented harness), `lore_scenario.install_descent` (heal/potion/illness fixes).

## Two paths (the escalated scope decision)

(a) **Build the full Gehennom descent agent** — the end-to-end path to an in-game
invocation. Substantial; the pieces exist and need integration + survival tuning,
evaluated multi-seed from the start (single-seed is RNG-confounded by kit edits).

(b) **Scenario-isolate rungs 4-7 as pure capability demos** — place the agent
directly at each rung (vibrating square, Sanctum, each plane) with the kit, and
show the LLM+wiki drives the sequence AA cannot, skipping the Gehennom-nav slog.
Faster to a paper result; a capability claim rather than fair-reach.

Rungs 1-3 are proven regardless. The paper's endgame contribution is the ladder:
"an LLM+knowledge layer drives a frozen symbolic SOTA through endgame milestones it
structurally cannot reach," with the reach clearly bounded.

Detail + instrumented arc: `Agent/daily/2026-07-11.md`. Code: `experiments/
autoascend/lore_scenario.py`, `descent_run.py`. Commits through feda313.
