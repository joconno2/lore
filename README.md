# LORE: Learned Optimization with Retrieved Expertise

> **Direction change (2026-06-20):** LORE now freezes AutoAscend (the NeurIPS
> 2021 NetHack Challenge winner, mean ~23,670 on our infra) as the base, and
> adds an EC-tuned LLM oracle that intervenes at AutoAscend's weak decision
> points. The expert-system line below (agent2*.py) is retired. See
> [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md) for the current plan. The
> rest of this README documents the superseded expert-system work.

LORE is a NetHack agent. The long-term goal is to push past the best symbolic
bot, AutoAscend, on the decisions it gets wrong, toward the unsolved problem of
ascension.

No AI has ascended NetHack 3.6+. The game is unsolved.

## Where the score comes from

NetHackScore-v0 score is dominated by dungeon depth reached and experience,
plus gold and items. Learned agents plateau around dungeon level 3: they fight
but cannot discover the multi-step, knowledge-dependent strategies deeper play
requires (corpse safety, trap identification, prayer timing, Sokoban, resource
management). That knowledge lives in spoiler databases and wikis but is not
learnable from reward alone within any practical training budget.

LORE's bet is that a symbolic expert system gets the agent reliably into the
deep game, and an EC-optimized knowledge layer (and, later, an LLM oracle) wins
the long-tail contextual decisions that AutoAscend spends most of its code on.
The expert base is the precondition: an agent stuck on DL1 never exercises a
single decision the knowledge layer is meant to improve.

## Architecture

The agent is a modular expert system. The main loop (`nhc/agent2.py`, AgentV2,
the current base) explores, descends, fights, eats, and prays, dispatching to
specialized subsystems:

```
nhc/
  obs_parser.py   # Parse raw NLE glyphs/blstats/message into semantic game state
  navigation.py   # BFS pathfinding on the 21x79 grid, per-level memory
  strategy.py     # Milestone system and level objectives (AutoAscend global_logic style)
  fight.py        # Fight/flee/Elbereth decisions, priority-based (AutoAscend fight_heur)
  combat.py       # Monster threat evaluation from parsed monster + corpse data
  food.py         # Corpse tracking, freshness, nutrition budget
  prayer.py       # Prayer safety: when prayer will help vs anger the god
  item_id.py      # Constraint-based item identification (BotHack core.logic style)
  equipment.py    # Weapon DPS, armor AC, auto-equip
  sokoban.py      # Sokoban solver via precomputed solutions matched to layout
  soko_maps.py    # Known Sokoban map signatures + solutions
  elbereth_env.py # NLE env wrapper passing eat/engrave getlin prompts through

  agent2.py       # AgentV2: current base, best performer (mean 700)
  agent2_v3.py    # Jun-4 iterations (v3/v4/v5). Regressed vs v2, see Results.
  agent2_v4.py
  agent2_v5.py

  # Knowledge layer (LORE contribution, in progress)
  kb.py           # Entity property table (381 monsters x 14 props), 27 situational rules
  rules.py        # Corpse effects, monster properties, Elbereth, prayer mechanics
  models.py       # KBConditioner + actor-critic (from the PPO line, see History)
```

The knowledge base is built from a 447MB corpus: 14K wiki pages, 384 monsters
and 455 items extracted from 3.6.7 source, 33 artifacts, full prayer and corpse
mechanics, and 5 reference bot repos.

## Roadmap

**Stage 1 (current): expert base to depth.** Get the symbolic agent reliably
past the early dungeon. Score compounds fastest on two things: Sokoban
completion (guaranteed depth plus luckstone/items) and surviving the Mines.
Target the depth distribution first, score follows.

**Stage 2: EC-optimized knowledge layer.** Once the base reaches DL5-10
reliably, evolve how the agent uses its knowledge base (per-rule trust weights,
entity weights, query thresholds) with CMA-ES / MAP-Elites. Fitness is raw game
score. The measured delta over the expert base is the contribution.

**Stage 3: LLM knowledge oracle.** Replace the static KB with retrieval-augmented
queries to a local LLM (Gemma on threadripper via vLLM) for the contextual
decisions the rule tables can't cover. EC optimizes the interface (when to query,
what context to include, how to parse the response); the LLM stays frozen.

Target venue: CoG 2027. Near-term milestone: ~5,000 mean (AutoAscend-class is
8,556).

## Results

NetHackScore-v0 (engrave-passthrough env), Valkyrie (val-hum-law-fem),
30 episodes, seeds 42-71.

| Agent | Mean | Median | Max | DL reached (death) | Notes |
|-------|------|--------|-----|--------------------|-------|
| Expert v2 | **700** | 308 | 4,559 | 27% DL1, reaches DL8 | Jun-1 base. Best agent. |
| Expert v5 | 185 | 104 | 654 | 60% DL1, none past DL4 | Jun-4 iteration. Regressed. |
| Expert v4 | 125 | 90 | 380 | -- | Jun-4 iteration. Regressed. |
| Expert v3 | -- | -- | -- | -- | Incompatible with current env. |

v2 is the canonical base. The Jun-4 v3/v4/v5 iterations (milestone gating that
farms DL1 until XL3, stricter stuck detection) regressed descent badly: v5 dies
on DL1 in 60% of episodes vs v2's 27%, and never gets past DL4 where v2 reaches
DL8. The Jun-4 line should be reverted or its changes cherry-picked against v2.

The DL1 death rate is the headline problem even for v2: a quarter of episodes
never escape the first level (secret-door search and early survival), so they
never reach the depth where score lives. v2's best run already hits 4,559, so
Stage 1 is making v2's good runs typical, not finding new ceiling.

For context, published results on NetHackScore-v0:

| Agent | Type | Mean Score |
|-------|------|------------|
| AutoAscend | Symbolic (15K LOC) | 8,556 |
| Sample Factory APPO | Neural (2B steps) | 3,245 |
| HiHack (IL from AutoAscend) | Neural + IL | 1,551 |
| NetPlay (GPT-4) | LLM zero-shot | 405 |

## History (PPO line)

LORE started as a learned-policy project: KB-conditioned PPO (a ~2.6M-param
recurrent actor-critic with a 21K-param KBConditioner) plus CMA-ES meta-control.
That line confirmed the RL ceiling and was set aside:

| Experiment | Steps | Mean | Notes |
|------------|-------|------|-------|
| B2 v2 (PPO, no KB) | 500M | 65 | DL1-3 combat only. Matches the known RL ceiling. |
| B4 (KB-conditioned PPO) | 500M | 63 | KB did not help raw PPO. |
| Phase 2 (CMA-ES meta-controller) | -- | 180 | Best over 200 gens. |

The expert system already exceeds all of these, which is why the project pivoted
to the symbolic base described above. `models.py`, `kb.py`, and the
`experiments/lore/` PPO entry points remain for the Stage 2/3 knowledge layer.

## Usage

```bash
# Benchmark the current expert agent (AgentV2)
python eval_v2_baseline.py

# PPO line (history / Stage 2-3 reuse)
python -m experiments.lore.train_ppo --use-kb --total-steps 500000000
python -m experiments.lore.evolve_meta --checkpoint runs/lore_kb_s0/latest.pt
```

## Requirements

- Python 3.10+
- NLE 1.2.0
- PyTorch 2.6+ with CUDA (only for the PPO line; the expert agent is CPU-only)
- HTCondor cluster for parallel evaluation (optional)

## Installation

```bash
pip install -e .
```

## Related Work

- **NLE and NetHack agents:** Kuttler et al. (NeurIPS 2020), Hambro et al. (NeurIPS 2022), Piterbarg et al. (NeurIPS 2023)
- **Symbolic bots:** AutoAscend (NeurIPS 2021 NetHack Challenge winner), BotHack
- **LLM + RL:** Motif (Klissarov et al., ICLR 2024), SPRING (Wu et al., NeurIPS 2023), Eureka (Ma et al., ICLR 2024)
- **EC + LLM:** EvoPrompt (Guo et al., ICLR 2024), QDAIF (Bradley et al., ICLR 2024)
- **LLM on NetHack:** NetPlay (Jeurissen et al., CoG 2024), BALROG (Paglieri et al., ICLR 2025)

## License

MIT
