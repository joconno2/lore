#!/usr/bin/env python3
"""Evaluate the expert system agent on NetHackScore-v0.

Usage:
    python scripts/eval_expert.py --episodes 100
    python scripts/eval_expert.py --episodes 10 --verbose
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import nle.nethack as nethack
from nle.env.tasks import NetHackScore

from nhc.expert_agent import ExpertAgent


def make_env(seed=None):
    env = gym.make(
        "NetHackScore-v0",
        observation_keys=(
            "glyphs", "blstats", "message",
            "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
        ),
        actions=nethack.ACTIONS,
        max_episode_steps=5000,
    )
    return env


def run_episode(env, agent, seed=None, verbose=False):
    obs, info = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if verbose and steps % 200 == 0:
            gs = agent.state
            print(f"  step {steps}: score={total_reward:.0f} hp={gs.hp}/{gs.max_hp} "
                  f"dl={gs.dlevel} xl={gs.xlevel} turn={gs.turn}")

    gs = agent.state
    return {
        "score": total_reward,
        "steps": steps,
        "dlevel": gs.dlevel,
        "xlevel": gs.xlevel,
        "turn": gs.turn,
        "hp": gs.hp,
        "max_hp": gs.max_hp,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    env = make_env()
    agent = ExpertAgent(verbose=args.verbose)

    results = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed + ep
        result = run_episode(env, agent, seed=seed, verbose=args.verbose)
        results.append(result)

        scores = [r["score"] for r in results]
        elapsed = time.time() - t0
        print(f"Ep {ep+1}/{args.episodes}: score={result['score']:.0f} "
              f"dl={result['dlevel']} xl={result['xlevel']} steps={result['steps']} | "
              f"avg={np.mean(scores):.1f} med={np.median(scores):.1f} "
              f"max={np.max(scores):.0f} [{elapsed:.0f}s]")

    env.close()

    scores = [r["score"] for r in results]
    dlevels = [r["dlevel"] for r in results]
    print("\n" + "=" * 60)
    print(f"Results over {len(results)} episodes:")
    print(f"  Score:  mean={np.mean(scores):.1f}  median={np.median(scores):.1f}  "
          f"max={np.max(scores):.0f}  min={np.min(scores):.0f}  std={np.std(scores):.1f}")
    print(f"  DLevel: mean={np.mean(dlevels):.1f}  max={np.max(dlevels)}")
    print(f"  Time:   {time.time() - t0:.0f}s total, {(time.time() - t0) / len(results):.1f}s/ep")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"episodes": results, "summary": {
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "max_score": float(np.max(scores)),
                "mean_dlevel": float(np.mean(dlevels)),
            }}, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
