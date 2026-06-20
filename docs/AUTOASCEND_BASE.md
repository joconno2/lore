# Frozen AutoAscend Base

The LORE pivot (Jun 20, 2026): freeze AutoAscend (NeurIPS 2021 NetHack Challenge
winner) as the base agent; LORE becomes an EC-tuned LLM oracle that intervenes at
AutoAscend's weak decision points. We do not reimplement AutoAscend. Our prior
expert system (agent2.py v2-v5) is retired.

## Build

AutoAscend is a 2021 codebase pinned to old deps (gym 0.19, numpy 1.21, NLE
0.7.3, numba 0.52) on an NGC pytorch:21.08 base. It cannot share the modern
nle-1.2.0 / gymnasium venv, so it lives in its own Docker image.

- Source: `data/bots/autoascend/` (vendored)
- Image: `aall/autoascend:frozen` (built on trx)
- Dockerfile patch: added `git submodule update --init --recursive` after the
  `git checkout v0.7.3`. The original recursive clone happens before the tag
  checkout, which leaves `third_party/libtmt` empty and the NLE compile fails on
  a missing `tmt.h`. Re-syncing submodules after the checkout fixes it.

## Headless benchmark

`data/bots/autoascend/aa_bench.py` runs N seeded episodes with no Ray and no X11
(bin/main.py's `simulate` needs Ray, `run` needs X11). It builds `EnvWrapper`
directly, runs `env.main()`, and dumps `get_summary()` per episode (score, depth,
XP, milestone, panic count, end_reason). This is also the integration surface for
the oracle.

```bash
docker run --rm --ipc=host \
  -v ~/nethack-aall/lore/data/bots/autoascend:/workspace -w /workspace \
  -e PYTHONPATH=/workspace aall/autoascend:frozen \
  python aa_bench.py <N> <base_seed> /workspace/out.json
```

## First result (Jun 20, 10 episodes, seeds 42-51, NetHackChallenge-v0)

mean 23,671, median 17,161, max 99,288, min 1,929. Reaches DL10-12, Minetown,
Sokoban. The base runs correctly on our infra. (Higher than the often-cited
8,556 because that is a NetHackScore-v0 number; this is NetHackChallenge-v0 and
the score is heavy-tailed. n=10 is small; trust the median.)

Raw: `results/bench_2026-06-20/autoascend_frozen_10.json`.

## Weak points (empirical, n=10)

The deaths concentrate, and not where the TODOs suggested:

| Failure mode | Count | Notes |
|--------------|-------|-------|
| Starvation (fainted from lack of food, then killed) | 4-5 | Dominant killer. Food strategy failing mid-game. |
| Killed while praying | 1 | Prayer-timing failure. |
| Code exception (item parsing) | 1 | `AssertionError('2 dwarven roots')`. Engineering, not research. |
| Combat/hazard (poison, lightning, centaur, ant) | 3 | Mid-game Mines survival. |

Despite AutoAscend's sophisticated food system, ~40-50% of deaths are starvation.
This is the first oracle target: when/what to eat, food conservation, when to
pray for food. The unimplemented `GO_DOWN` endgame strategy (global_logic.py:150)
does not matter yet — the agent dies in the mid-game before reaching it.

## Next

1. Larger gap-analysis sweep (50-100 episodes) on Condor/mega_knight (NOT trx
   during the qd-bw roundrobin). Confirm the starvation signal and rank weak
   points by frequency and recoverable score.
2. Instrument AutoAscend's decision dispatch (global_logic.py `current_strategy`,
   agent.py atomic actions, panic events) to log decision context at the weak
   points, so the oracle has a defined hook surface.
3. Wire the LLM oracle at the top-ranked decision point; measure delta vs frozen
   base.
