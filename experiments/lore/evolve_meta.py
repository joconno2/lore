#!/usr/bin/env python3
"""LORE Phase 2: CMA-ES evolution of the KB meta-controller.

Loads a PPO-trained KB-conditioned Agent checkpoint, freezes all neural
params, and evolves only the 172-dim meta-controller vector (trust
weights, entity type weights, rule priority bias, query threshold,
attention temperature) using CMA-ES.

Fitness = mean episode score over N episodes of NetHackScore-v0.

Can run locally (GPU vectorized eval) or on Condor (CPU parallel).

Usage:
    # Local GPU (threadripper/vertex)
    python -m experiments.lore.evolve_meta \
        --checkpoint runs/lore_kb_s0/latest.pt \
        --generations 200 --pop-size 32 --episodes 20

    # Quick test
    python -m experiments.lore.evolve_meta \
        --checkpoint runs/lore_kb_s0/latest.pt \
        --generations 10 --pop-size 8 --episodes 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.cmaes import CMAES
from experiments.shared.eval_agent import evaluate_episodes, evaluate_episodes_vec
from nhc.env import NUM_ACTIONS
from nhc.models import Agent, KBConditioner


def load_kb_agent(ckpt_path: Path, device: str = "cpu") -> Agent:
    """Load a KB-conditioned Agent from a training checkpoint."""
    kb = KBConditioner(num_rules=80, kb_dim=64)
    model = Agent(num_actions=NUM_ACTIONS, kb_conditioner=kb)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd, strict=False)

    # Freeze all neural params (only meta-controller evolves)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model


def evaluate_with_meta(model: Agent, meta_params: np.ndarray,
                       num_episodes: int = 20, max_steps: int = 5000,
                       device: str = "cpu", num_envs: int = 8) -> float:
    """Set meta-controller params and evaluate."""
    model.kb.set_meta_params(meta_params)
    if device == "cuda" or device.startswith("cuda:"):
        result = evaluate_episodes_vec(
            model, num_episodes=num_episodes, num_envs=num_envs,
            max_steps=max_steps, device=device)
    else:
        result = evaluate_episodes(
            model, num_episodes=num_episodes, max_steps=max_steps,
            device=device)
    return result["mean_score"]


def main():
    p = argparse.ArgumentParser(description="LORE Phase 2: CMA-ES meta-controller")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to PPO-trained KB-conditioned Agent checkpoint")
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--pop-size", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--episodes", type=int, default=20,
                   help="Episodes per fitness evaluation")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--num-envs", type=int, default=8,
                   help="Parallel envs for vectorized eval (GPU only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--results-dir", default="results/lore_meta")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading KB-conditioned Agent from %s" % args.checkpoint)
    model = load_kb_agent(args.checkpoint, device=args.device)
    n_meta = model.kb.meta_param_count()
    print("Meta-controller params: %d" % n_meta)

    # Initialize CMA-ES from zeros (neutral meta-controller)
    x0 = np.zeros(n_meta, dtype=np.float64)
    cma = CMAES(x0, sigma0=args.sigma, pop_size=args.pop_size)

    print("CMA-ES: pop=%d, n=%d, sigma=%.3f" % (cma.lam, cma.n, args.sigma))
    print("Fitness: mean score over %d episodes" % args.episodes)
    print("=" * 60)

    best_score = -float("inf")
    best_params = x0.copy()
    log = []

    for gen in range(args.generations):
        t0 = time.time()
        solutions = cma.ask()
        fitnesses = np.zeros(len(solutions))

        for i, sol in enumerate(solutions):
            fitnesses[i] = evaluate_with_meta(
                model, sol.astype(np.float32),
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                device=args.device,
                num_envs=args.num_envs,
            )

        cma.tell(solutions, fitnesses)
        elapsed = time.time() - t0

        gen_best = fitnesses.max()
        gen_mean = fitnesses.mean()
        if gen_best > best_score:
            best_score = gen_best
            best_params = solutions[fitnesses.argmax()].copy()
            np.save(results_dir / "best_meta_params.npy", best_params)

        entry = {
            "gen": gen,
            "pop_best": float(gen_best),
            "pop_mean": float(gen_mean),
            "best_ever": float(best_score),
            "sigma": float(cma.sigma),
            "elapsed": elapsed,
        }
        log.append(entry)

        print("Gen %3d | best=%.1f mean=%.1f best_ever=%.1f sigma=%.4f [%.1fs]"
              % (gen, gen_best, gen_mean, best_score, cma.sigma, elapsed))

        # Save log periodically
        if (gen + 1) % 5 == 0:
            with open(results_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)

    # Final save
    with open(results_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)
    np.save(results_dir / "best_meta_params.npy", best_params)

    print("\nDone. Best score: %.1f" % best_score)
    print("Best params saved to %s/best_meta_params.npy" % results_dir)


if __name__ == "__main__":
    main()
