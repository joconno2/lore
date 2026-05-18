# LORE: Learned Optimization with Retrieved Expertise

Knowledge-augmented evolutionary agent for NetHack. Combines structured wiki knowledge with neural RL and evolutionary meta-optimization.

## Architecture

1. **Structured KB** (`nhc/kb.py`): Entity properties and situational rules extracted from the NetHack wiki. Static lookup, no LLM at runtime.
2. **KBConditioner** (`nhc/models.py`): 21K-param module injecting KB context into the Agent's LSTM via entity attention and rule activation pathways.
3. **EC Meta-Controller**: 172-dim vector (rule trust weights, entity type weights, attention temperature) evolved by CMA-ES on raw game score.

## Usage

```bash
# Phase 1: PPO training (B2 baseline, no KB)
python -m experiments.lore.train_ppo --total-steps 500000000

# Phase 1: PPO training (KB-conditioned)
python -m experiments.lore.train_ppo --use-kb --total-steps 500000000

# Phase 2: CMA-ES on meta-controller
python -m experiments.lore.evolve_meta --checkpoint runs/lore_kb_s0/latest.pt

# Evaluation
python -m experiments.lore.eval_all --b2-ckpt runs/lore_baseline_s0/latest.pt --b4-ckpt runs/lore_kb_s0/latest.pt
```

## Requirements

- NLE 1.2.0
- PyTorch 2.x with CUDA
- RTX 5090 (32GB) recommended
