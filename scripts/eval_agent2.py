#!/usr/bin/env python3
"""Evaluate AgentV2 on NetHackChallenge-v0."""
import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import nle.nethack as nethack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nhc.agent2 import AgentV2, AgentFinished


def make_env(character="val-hum-fem-law"):
    return gym.make(
        "NetHackScore-v0",
        observation_keys=(
            "glyphs", "blstats", "message", "misc", "specials",
            "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
        ),
        actions=nethack.ACTIONS,
        character=character,
        max_episode_steps=100000,
        allow_all_yn_questions=False,
        penalty_step=0.0,
    )


def run_episode(env, seed=0, verbose=False):
    agent = AgentV2(env, verbose=verbose, seed=seed)
    t0 = time.time()
    agent.main()

    bl = agent.blstats
    elapsed = time.time() - t0
    print(f"  DEBUG: ugs_count={agent._ugs_count} blstats={bl.depth if bl else 'None'}/{bl.xl if bl else 'None'}/{bl.time if bl else 'None'}")
    return {
        "score": agent.score,
        "steps": agent.step_count,
        "dlevel": bl.depth if bl else 0,
        "xlevel": bl.xl if bl else 0,
        "turn": bl.time if bl else 0,
        "hp": bl.hp if bl else 0,
        "max_hp": bl.max_hp if bl else 0,
        "seed": seed,
        "elapsed": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    results = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed + ep
        env = make_env()
        result = run_episode(env, seed=seed, verbose=args.verbose)
        env.close()
        results.append(result)

        scores = [r["score"] for r in results]
        elapsed = time.time() - t0
        print(f"Ep {ep+1}/{args.episodes}: score={result['score']:.0f} "
              f"dl={result['dlevel']} xl={result['xlevel']} "
              f"steps={result['steps']} turns={result['turn']} "
              f"spt={result['steps']/max(1,result['turn']):.1f} "
              f"[{elapsed:.0f}s]")

    scores = [r["score"] for r in results]
    dlevels = [r["dlevel"] for r in results]
    print(f"\n{'='*60}")
    print(f"Results over {len(results)} episodes:")
    print(f"  Score:  mean={np.mean(scores):.1f}  median={np.median(scores):.1f}  "
          f"max={np.max(scores):.0f}  min={np.min(scores):.0f}")
    print(f"  DLevel: mean={np.mean(dlevels):.1f}  max={np.max(dlevels)}")
    print(f"  Time:   {time.time() - t0:.0f}s total")

    if args.output:
        # Convert numpy types for JSON
        clean = []
        for r in results:
            clean.append({k: int(v) if hasattr(v, 'item') else float(v) if isinstance(v, float) else v
                          for k, v in r.items()})
        with open(args.output, "w") as f:
            json.dump({"episodes": clean}, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
