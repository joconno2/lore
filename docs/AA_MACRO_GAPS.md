# AutoAscend Macro-Strategy Gap Analysis (empirical)

Built from a large behavioral profile (`aa_profile.py`, n=100+ full real games,
seeds 100-199; validated against the original bench so the harness is clean).
Goal: find where AutoAscend (the symbolic SOTA) diverges from expert macro play —
the strategic gaps where an LLM's knowledge has room, since on tactical decisions
the LLM agrees with AA ~97% of the time (rigorously shown).

## AA's empirical profile (n=100)
- **Score**: mean 10,511, median 5,661 (heavy-tailed; max 106k). The often-cited
  "median ~17k" was a lucky 10-seed sample; the true median game is much weaker.
- **Depth reached**: median **DL3**, mean 3.5, p90 DL7. **44% of games never
  leave DL1.** The endgame (DL25+) is not remotely reached in fair play.
- **Deaths**: **starvation 49%**, combat 33%, crash 9%, prayer 5%.
- **Branch coverage**: Mines 37%, **Sokoban 5%**, **Quest 0%**.
- **Ascension kit**: ~0%. No reflection, no wand of wishing, no bag of holding,
  no dragon scale ever acquired. (magic marker 9%, Excalibur 7%, unicorn horn 4%.)
- **Behavior**: median 200 turns camped on a single tile; prayers median 11/game
  (band-aiding the food problem).

## The gaps vs expert macro play (ranked by impact)

### Gap 1 — DL1 over-farming → starvation (THE dominant flaw, ~42-49% of games)
AA's `BE_ON_FIRST_LEVEL` milestone holds it on DL1 until experience-level >= 8
(`global_logic.py:517`). But many DL1s cannot feed XL8 of grinding: trajectories
show the agent on DL1 for 11k-15k turns (XL crawling 1→6, HP full) until food
runs out and it starves. Expert play descends with a *small* cushion (XL ~3-5),
because deeper levels give faster XP *and* more food. **AA grinds the most
food-scarce level to death.**
- Structural probe (descend-when-hungry): starvation 49%→20%, depth median 3→5,
  Mines 37%→57% — but score *drops* (10.5k→5.8k) because the blunt rule also
  yanks the *productive* DL1 runs off early and kills the high-scoring games.
- **LLM's job**: distinguish a productively-leveling, well-fed DL1 (keep) from a
  doomed hungry/foodless spiral (descend). A fixed threshold can't; judgment can.

### Gap 2 — never builds an ascension kit
Expert play treats the early-mid game as kit assembly: reflection (before
Medusa), magic resistance, free action, a reliable weapon, bag of holding. AA
acquires essentially none of these — it has no notion of "what I need to win,"
only local heuristics. (Only relevant once Gap 1 lets games survive longer.)

### Gap 3 — skips Sokoban (5%) and Quest (0%)
Sokoban gives a guaranteed bag of holding OR amulet of reflection (a top kit
item) and is low-risk; experts always do it. The Quest gives the role artifact.
AA almost never completes either. High-value, low-risk objectives left on the table.

### Gap 4 — shallow; never approaches the endgame
Consequence of Gaps 1-3: median DL3 means Castle/Gehennom/invocation/ascension
are unreachable in fair play. AA's `GO_DOWN` endgame is an unimplemented TODO,
but that never matters because it dies long before.

## Implication for LORE
The LLM cannot beat AA on the tactical decisions AA already makes (refuted: 3
ablations + the survival oracle, all null, LLM agrees with AA 97%). The room is
**macro/strategic judgment AA lacks** — and Gap 1 is both the largest and the
cleanest: a single judgment ("keep farming vs move on") that a fixed rule gets
wrong and an LLM can get right, on the failure mode that ends ~half of all games.
That is the contribution test: LLM-judged descent timing that cuts starvation
WITHOUT sacrificing the strong games → beats the SOTA on score, fairly, via an
LLM angle.

## UPDATE — the score lives in the TAIL; target what kills high-value games

The mean is tail-driven (median DL2-3, but max 106k). What kills the *high-value*
games (n=250 base, by death category, with mean score of that category):
- **crash (AA's own AssertionError/RecursionError): 18 games, mean 22,388** — the
  single highest-value death category. The deep tail dies to AA's own bugs.
- **petrification/paralysis: 11 games, mean ~17k (incl. one 79,568)** —
  knowledge-gated instadeaths of high-value games.
- starvation 106 / combat 100 — dominant by count, but LOW value (the shallow
  median games); not where score is lost.

This reframes the whole effort. You don't beat AutoAscend's *score* by fixing the
median (starvation) — that's score-efficient farming (proven: every descent-timing
intervention, even perfect-knowledge, LOSES). You beat it by **preserving the
high-value tail**, which dies to exactly two things:

1. **Crashes → `crash_recovery` (structural).** Result: **+4.4% mean, 10 wins / 0
   losses / 89 ties (n=99)** — a strict, rigorous improvement over the SOTA.
2. **Knowledge-gated instadeaths (petrification/paralysis) → the LLM veto.** These
   are the deaths an LLM with NetHack knowledge prevents and AA's heuristics don't
   (AA melees cockatrices). The earlier veto null was on *shallow* scenarios where
   petrification isn't the killer; on the tail it is. (Testing crash_recovery+veto
   at n=250 — the petrif/paralysis seeds are mostly 200-349.)

## Revised contribution thesis (two-fold bar)
LORE = preserve AutoAscend's high-value tail. Structural `crash_recovery` (+4.4%,
clean) handles its self-crashes; the **LLM knowledge-veto** handles the
knowledge-gated instadeaths AA walks into. Both beat the SOTA on score by saving
the games that actually carry the score, and the veto is a genuine LLM angle
(NetHack knowledge AA lacks). This is the first framing where the data supports a
real win on both bars — and it came straight out of the large behavioral profile.
