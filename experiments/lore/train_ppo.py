#!/usr/bin/env python3
"""LORE Phase 1: Train KB-conditioned Agent with PPO.

This is a thin wrapper around train_specialist that enables the KB.

Usage:
    # B4: KB-conditioned PPO (the main LORE PPO phase)
    python -m experiments.lore.train_ppo --use-kb --total-steps 500000000

    # B2: Monolithic PPO baseline (no KB)
    python -m experiments.lore.train_ppo --total-steps 500000000

    # Quick test (10M steps, ~20 min)
    python -m experiments.lore.train_ppo --use-kb --total-steps 10000000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nhc.training.trainer import TrainConfig, train_specialist


def main():
    p = argparse.ArgumentParser(description="LORE Phase 1: PPO training")
    p.add_argument("--use-kb", action="store_true",
                   help="Attach KBConditioner to the Agent")
    p.add_argument("--kb-dim", type=int, default=64)
    p.add_argument("--kb-num-rules", type=int, default=80)
    p.add_argument("--total-steps", type=int, default=500_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-env-workers", type=int, default=8)
    p.add_argument("--rollout-len", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--reward-clip", type=float, default=10.0)
    p.add_argument("--normalize-returns", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-id", default=None,
                   help="Run directory name (default: auto-generated)")
    args = p.parse_args()

    tag = "lore_kb" if args.use_kb else "lore_baseline"
    run_id = args.run_id or ("%s_s%d" % (tag, args.seed))

    cfg = TrainConfig(
        sid="lore",
        run_id=run_id,
        use_kb=args.use_kb,
        kb_dim=args.kb_dim,
        kb_num_rules=args.kb_num_rules,
        num_envs=args.num_envs,
        num_env_workers=args.num_env_workers,
        rollout_len=args.rollout_len,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        reward_clip=args.reward_clip,
        normalize_returns=args.normalize_returns,
        env_id_override="NetHackScore-v0",
        reset_step_counter=True,
        seed=args.seed,
        device=args.device,
    )
    print("=" * 60)
    print("LORE Phase 1: PPO on NetHackScore-v0")
    print("  KB enabled: %s" % args.use_kb)
    if args.use_kb:
        print("  KB dim: %d, rules: %d" % (args.kb_dim, args.kb_num_rules))
    print("  Total steps: %d" % args.total_steps)
    print("  Num envs: %d" % args.num_envs)
    print("  Run ID: %s" % run_id)
    print("=" * 60)

    summary = train_specialist(cfg)
    print("\nTraining complete.")
    print("  Final mean reward: %.1f" % summary.get("mean_reward", 0))
    print("  Best mean reward: %.1f" % summary.get("best_mean_reward", 0))


if __name__ == "__main__":
    main()
