# Bigger-model test: does scale break the exact-token DIAGNOSIS ceiling?

The 14B diagnosis floor (generation ~5-22%) with high recognition (67%) predicted:
knowledge is present, generation floors on exact tokens. Test the prediction by scaling
the model. Ran the SAME co-pilot / recognition / concise-generation benchmarks on
Qwen2.5-32B-Instruct-AWQ (swapped into the lore_vllm container on trx, then restored the 14B).

## Result — scale improves RECOGNITION, not GENERATION

| metric | 14B | 32B |
|---|---|---|
| RECOGNITION (forced-choice, real fix + 3 distractors; chance 25%) | 10/15 = 67% | **14/15 = 93%** |
| GENERATION, co-pilot (diagnose given the function) | 13% strict | ~7% (verbose; truncation-confounded) |
| GENERATION, concise (state the one-line fix) | — | **0/15 = 0%** |

Scale takes recognition to near-perfect (93%) but does NOT break the generation ceiling —
open diagnosis stays ~0-13% (flat-to-worse). The 32B produces confident, plausible, WRONG
one-liners: bumps `assert < 5` to `< 5000` (not the graceful-quit fix), keeps the buggy
`glyphs[...] in MON.ALL_MONS` line, invents non-existent methods, and puts the right "traps"
idea on the wrong case. So the model increasingly KNOWS the fix (93% recognition) but still
cannot GENERATE it — the recognition-generation gap WIDENS with scale (54pt at 14B -> 86pt
at 32B). This is a strong confirmation of the exact-token-generation floor: it does not
scale away. (Caveat: 32B-AWQ 4-bit; but recognition rose, so the model is not broadly
degraded — generation specifically stays floored.)

## Deployment implication (reinforced)

A bigger model is a much better RANKER (93%) but not a better generator. So the usable form
— candidate ranking with EXTERNALLY-sourced candidates — gets STRONGER with scale, while
autonomous open diagnosis stays floored. The debugger's value scales as a triage/ranking
co-pilot, not as an autonomous fixer.
