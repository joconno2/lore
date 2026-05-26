# LORE: Learned Optimization with Retrieved Expertise

LORE is a NetHack agent that uses evolutionary computation to optimize how a neural policy retrieves and applies domain knowledge. The goal is to close the gap between learned agents (~3,000 mean score) and the best symbolic bot, AutoAscend (~8,500 mean score).

No AI has ascended NetHack 3.6+. The game is unsolved.

## Approach

Learned agents plateau at dungeon level 3. They can fight but cannot discover the multi-step, knowledge-dependent strategies required for deeper play (corpse safety, trap identification, prayer timing, resource management). This knowledge exists in spoiler databases and wikis but is not learnable from reward alone within any practical training budget.

LORE attacks this in three phases:

**Phase 1: KB-conditioned PPO.** A recurrent actor-critic (LSTM, ~2.6M params) is trained via PPO on NetHackScore-v0. A KBConditioner module (21K params) injects structured wiki knowledge into the policy's feature stream through two pathways: entity attention over a 14-dimensional property table covering all 381 monsters, and soft rule activations over 27 situational rules derived from the NetHack wiki.

**Phase 2: Evolutionary meta-optimization.** CMA-ES evolves a 172-dimensional meta-controller vector that tunes how the agent uses the KB: per-rule trust weights, entity type weights, rule priority biases, query threshold, and attention temperature. Fitness is raw game score from full episodes. The neural policy is frozen during evolution.

**Phase 3: LLM knowledge oracle.** Replace the static KB with retrieval-augmented queries to a local LLM, and evolve the retrieval strategy (when to query, what context to include, how to parse responses) via MAP-Elites. The LLM is frozen; EC optimizes the interface.

## Project Structure

```
nhc/
  models.py      # ObsEncoder, Agent (actor-critic), KBConditioner, ConsensusHMoE
  kb.py          # Entity property table, rule table, glyph classification
  rules.py       # NetHack domain knowledge: corpse effects, monster properties, Elbereth, prayer
  env.py         # NLE environment wrapper
  store.py       # SQLite metrics store, atomic run directories
  rnd.py         # Random Network Distillation (exploration bonus)
  training/
    trainer.py   # PPO+GAE training loop
    losses.py    # Policy/value/entropy losses
    rollout.py   # Rollout buffer
    env_pool.py  # Batched async vector environment

experiments/
  lore/
    train_ppo.py    # Phase 1 entry point
    evolve_meta.py  # Phase 2 entry point (CMA-ES)
    eval_all.py     # Evaluation across baselines
  shared/
    cmaes.py        # CMA-ES implementation
    openes.py       # OpenAI-ES implementation
    eval_agent.py   # Episode evaluation (single + vectorized)

condor/
  pack/             # Docker packaging for cluster jobs
  workers/          # Condor worker scripts
```

## Usage

```bash
# Phase 1: PPO baseline (no KB)
python -m experiments.lore.train_ppo --total-steps 500000000

# Phase 1: KB-conditioned PPO
python -m experiments.lore.train_ppo --use-kb --total-steps 500000000

# Phase 2: CMA-ES meta-controller evolution
python -m experiments.lore.evolve_meta --checkpoint runs/lore_kb_s0/latest.pt

# Evaluation
python -m experiments.lore.eval_all \
  --b2-ckpt runs/lore_baseline_s0/latest.pt \
  --b4-ckpt runs/lore_kb_s0/latest.pt
```

## Results So Far

| Experiment | Steps | Mean Score | Notes |
|------------|-------|------------|-------|
| B1 (HO-MoE v5) | -- | 82 | Prior baseline from nethack-aall |
| B2 v2 (PPO, no KB) | 500M | 65 | DL1-3 combat only. Matches known RL ceiling. |
| B4 (KB-conditioned PPO) | -- | -- | Next up |
| LORE (KB + PPO + CMA-ES) | -- | -- | After B4 |

For context, the best published results on NetHackScore-v0:

| Agent | Type | Mean Score |
|-------|------|------------|
| AutoAscend | Symbolic (15K LOC) | 8,556 |
| Sample Factory APPO | Neural (2B steps) | 3,245 |
| HiHack (IL from AutoAscend) | Neural + IL | 1,551 |
| NetPlay (GPT-4) | LLM zero-shot | 405 |

## Requirements

- Python 3.10+
- NLE 1.2.0+
- PyTorch 2.6+ with CUDA
- 32GB GPU recommended (RTX 5090 or equivalent) for training
- HTCondor cluster for CMA-ES evaluation (optional, can run locally)

## Installation

```bash
pip install -e .
```

## Related Work

LORE builds on several lines of research:

- **NLE and NetHack agents:** Kuttler et al. (NeurIPS 2020), Hambro et al. (NeurIPS 2022), Piterbarg et al. (NeurIPS 2023)
- **LLM + RL:** Motif (Klissarov et al., ICLR 2024), SPRING (Wu et al., NeurIPS 2023), Eureka (Ma et al., ICLR 2024)
- **EC + LLM:** EvoPrompt (Guo et al., ICLR 2024), ECS (Sun et al., 2026), QDAIF (Bradley et al., ICLR 2024)
- **LLM on NetHack:** NetPlay (Jeurissen et al., CoG 2024), BALROG (Paglieri et al., ICLR 2025)

## License

MIT
