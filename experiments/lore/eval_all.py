#!/usr/bin/env python3
"""LORE evaluation: run all baselines and the full system.

Evaluates each system on 100 episodes of NetHackScore-v0 and outputs
a comparison table.

Usage:
    python -m experiments.lore.eval_all \
        --b2-ckpt runs/lore_baseline_s0/latest.pt \
        --b4-ckpt runs/lore_kb_s0/latest.pt \
        --lore-ckpt runs/lore_kb_s0/latest.pt \
        --lore-meta results/lore_meta/best_meta_params.npy \
        --episodes 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.shared.eval_agent import evaluate_episodes_vec
from nhc.env import NUM_ACTIONS
from nhc.models import Agent, KBConditioner


def load_agent(ckpt_path: Path, use_kb: bool = False,
               device: str = "cpu") -> Agent:
    """Load an Agent checkpoint."""
    kb = KBConditioner() if use_kb else None
    model = Agent(num_actions=NUM_ACTIONS, kb_conditioner=kb)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def eval_system(name: str, model: Agent, num_episodes: int,
                device: str, num_envs: int = 8) -> dict:
    """Evaluate a system and return stats."""
    result = evaluate_episodes_vec(
        model, num_episodes=num_episodes, num_envs=num_envs,
        max_steps=5000, device=device)
    scores = result["scores"]
    return {
        "name": name,
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
        "mean_depth": result["mean_depth"],
        "episodes": len(scores),
    }


def main():
    p = argparse.ArgumentParser(description="LORE evaluation suite")
    p.add_argument("--b2-ckpt", type=Path, default=None,
                   help="B2: Monolithic PPO (no KB)")
    p.add_argument("--b4-ckpt", type=Path, default=None,
                   help="B4: KB-conditioned PPO (no EC)")
    p.add_argument("--lore-ckpt", type=Path, default=None,
                   help="LORE: KB-conditioned PPO checkpoint")
    p.add_argument("--lore-meta", type=Path, default=None,
                   help="LORE: evolved meta-controller params (.npy)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("results/lore_eval"))
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    # B1: HO-MoE v5 baseline (known value)
    results.append({
        "name": "B1: HO-MoE v5 (Jay)",
        "mean": 82.0, "median": 65.0, "std": 45.0,
        "max": 280.0, "min": 5.0, "mean_depth": 2.1, "episodes": 100,
        "source": "prior_result",
    })

    # B2: Monolithic PPO
    if args.b2_ckpt and args.b2_ckpt.exists():
        print("Evaluating B2: Monolithic PPO...")
        model = load_agent(args.b2_ckpt, use_kb=False, device=args.device)
        results.append(eval_system("B2: Monolithic PPO", model,
                                   args.episodes, args.device, args.num_envs))

    # B4: KB-conditioned PPO (no EC meta-controller)
    if args.b4_ckpt and args.b4_ckpt.exists():
        print("Evaluating B4: KB-conditioned PPO...")
        model = load_agent(args.b4_ckpt, use_kb=True, device=args.device)
        results.append(eval_system("B4: KB + PPO (no EC)", model,
                                   args.episodes, args.device, args.num_envs))

    # LORE: full system (KB + PPO + evolved meta-controller)
    if args.lore_ckpt and args.lore_ckpt.exists():
        print("Evaluating LORE: KB + PPO + EC...")
        model = load_agent(args.lore_ckpt, use_kb=True, device=args.device)
        if args.lore_meta and args.lore_meta.exists():
            meta = np.load(args.lore_meta).astype(np.float32)
            model.kb.set_meta_params(meta)
        results.append(eval_system("LORE: KB + PPO + EC", model,
                                   args.episodes, args.device, args.num_envs))

    # Print comparison table
    print("\n" + "=" * 80)
    print("%-30s %8s %8s %8s %8s %8s" %
          ("System", "Mean", "Median", "Std", "Max", "Depth"))
    print("-" * 80)
    for r in results:
        print("%-30s %8.1f %8.1f %8.1f %8.1f %8.1f" %
              (r["name"], r["mean"], r["median"], r["std"],
               r["max"], r["mean_depth"]))
    print("=" * 80)

    # Also include SOTA reference
    print("\nSOTA reference:")
    print("  AutoAscend (symbolic):        5,336 median")
    print("  Sample Factory APPO (neural): 3,245 mean")
    print("  E2 heads-only ES (AALL):        285 best")

    # Save results
    with open(args.output / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to %s/results.json" % args.output)


if __name__ == "__main__":
    main()
