#!/usr/bin/env python3
"""Evaluate the expert system agent on NetHackScore-v0.

Usage:
    python scripts/eval_expert.py --episodes 100
    python scripts/eval_expert.py --episodes 20 --verbose
    python scripts/eval_expert.py --episodes 50 --analyze
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import nle.nethack as nethack

from nhc.expert_agent import ExpertAgent


def make_env(seed=None, character="val-hum-fem-neu"):
    env = gym.make(
        "NetHackChallenge-v0",
        observation_keys=(
            "glyphs", "blstats", "message", "misc",
            "inv_glyphs", "inv_strs", "inv_letters", "inv_oclasses",
        ),
        character=character,
        max_episode_steps=5000,
        no_progress_timeout=10000,
    )
    return env


class EpisodeTracker:
    """Track per-step diagnostics for a single episode."""
    def __init__(self):
        self.action_counts = Counter()
        self.priority_counts = Counter()
        self.stuck_steps = 0
        self.prompt_steps = 0
        self.combat_steps = 0
        self.nav_steps = 0
        self.search_steps = 0
        self.turns_advanced = 0
        self.last_turn = 0
        self.messages = []
        self.stuck_messages = Counter()
        self.max_stuck_run = 0
        self._cur_stuck_run = 0
        self._last_pos = None
        self._repeated_msg = Counter()

    def record_step(self, obs, action, agent):
        s = agent.state
        action_name = agent._action_name(action)
        self.action_counts[action_name] += 1

        # Track turn advancement
        if s.turn > self.last_turn:
            self.turns_advanced += s.turn - self.last_turn
            self.last_turn = s.turn

        # Track position stuck
        pos = s.position
        if pos == self._last_pos:
            self._cur_stuck_run += 1
            self.stuck_steps += 1
        else:
            self.max_stuck_run = max(self.max_stuck_run, self._cur_stuck_run)
            self._cur_stuck_run = 0
        self._last_pos = pos

        # Track message patterns
        msg = ""
        msg_raw = obs.get("message")
        if msg_raw is not None:
            msg = bytes(msg_raw).rstrip(b'\x00').decode("latin-1", errors="replace").strip()

        if msg and self._cur_stuck_run > 5:
            self.stuck_messages[msg[:60]] += 1

        # Categorize action
        if action_name == "SEARCH":
            self.search_steps += 1
        elif action_name in ("MORE", "YN") or "[yn]" in msg or "--More--" in msg:
            self.prompt_steps += 1
        elif action_name in ("N", "E", "S", "W", "NE", "SE", "SW", "NW"):
            if s.has_adjacent_monsters:
                self.combat_steps += 1
            else:
                self.nav_steps += 1

    def summary(self):
        self.max_stuck_run = max(self.max_stuck_run, self._cur_stuck_run)
        return {
            "action_counts": dict(self.action_counts.most_common(15)),
            "stuck_steps": self.stuck_steps,
            "prompt_steps": self.prompt_steps,
            "combat_steps": self.combat_steps,
            "nav_steps": self.nav_steps,
            "search_steps": self.search_steps,
            "max_stuck_run": self.max_stuck_run,
            "turns_advanced": self.turns_advanced,
            "top_stuck_messages": dict(self.stuck_messages.most_common(5)),
        }


def run_episode(env, agent, seed=None, verbose=False, analyze=False):
    obs, info = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    steps = 0
    done = False
    tracker = EpisodeTracker() if analyze else None
    death_msg = ""

    while not done:
        action = agent.act(obs)
        if tracker:
            tracker.record_step(obs, action, agent)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if terminated:
            msg_raw = obs.get("message")
            if msg_raw is not None:
                death_msg = bytes(msg_raw).rstrip(b'\x00').decode("latin-1", errors="replace").strip()

        if verbose and steps % 500 == 0:
            gs = agent.state
            print(f"  step {steps}: score={total_reward:.0f} hp={gs.hp}/{gs.max_hp} "
                  f"dl={gs.dlevel} xl={gs.xlevel} turn={gs.turn}")

    gs = agent.state
    result = {
        "score": total_reward,
        "steps": steps,
        "dlevel": gs.dlevel,
        "xlevel": gs.xlevel,
        "turn": gs.turn,
        "hp": gs.hp,
        "max_hp": gs.max_hp,
        "died": terminated and not truncated,
        "death_msg": death_msg if terminated else "",
        "seed": seed,
    }
    if tracker:
        result["diagnostics"] = tracker.summary()
    return result


def print_analysis(results):
    """Print comprehensive QA analysis."""
    scores = [r["score"] for r in results]
    steps = [r["steps"] for r in results]
    dlevels = [r["dlevel"] for r in results]
    xlevels = [r["xlevel"] for r in results]
    turns = [r["turn"] for r in results]
    died = [r for r in results if r.get("died")]
    timed_out = [r for r in results if r["steps"] >= 5000]

    print("\n" + "=" * 70)
    print("EXPERT AGENT QA REPORT")
    print("=" * 70)

    print("\n--- Score Distribution ---")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{pct:2d}: {np.percentile(scores, pct):>7.0f}")
    print(f"  mean: {np.mean(scores):>7.1f}  std: {np.std(scores):.1f}")
    positive = sum(1 for s in scores if s > 0)
    print(f"  positive: {positive}/{len(results)} ({100*positive/len(results):.0f}%)")

    print("\n--- Progression ---")
    print(f"  DLevel: mean={np.mean(dlevels):.1f}  max={max(dlevels)}  "
          f"dist={dict(Counter(dlevels).most_common(8))}")
    print(f"  XLevel: mean={np.mean(xlevels):.1f}  max={max(xlevels)}  "
          f"dist={dict(Counter(xlevels).most_common(8))}")
    print(f"  Turns:  mean={np.mean(turns):.0f}  median={np.median(turns):.0f}  max={max(turns)}")

    print("\n--- Survival ---")
    print(f"  Died: {len(died)}/{len(results)} ({100*len(died)/len(results):.0f}%)")
    print(f"  Timed out: {len(timed_out)}/{len(results)} ({100*len(timed_out)/len(results):.0f}%)")
    steps_per_turn = [s/max(t, 1) for s, t in zip(steps, turns)]
    print(f"  Steps/turn: mean={np.mean(steps_per_turn):.1f}  "
          f"(1.0 = perfect, >3 = wasting steps)")

    if died:
        print("\n--- Death Causes ---")
        death_causes = Counter()
        for r in died:
            msg = r.get("death_msg", "unknown")
            death_causes[msg[:60]] += 1
        for cause, count in death_causes.most_common(10):
            print(f"  {count:3d}x: {cause}")

    # Diagnostics analysis (if available)
    diags = [r.get("diagnostics") for r in results if r.get("diagnostics")]
    if diags:
        print("\n--- Step Budget Allocation (averaged) ---")
        avg_combat = np.mean([d["combat_steps"] for d in diags])
        avg_nav = np.mean([d["nav_steps"] for d in diags])
        avg_search = np.mean([d["search_steps"] for d in diags])
        avg_prompt = np.mean([d["prompt_steps"] for d in diags])
        avg_stuck = np.mean([d["stuck_steps"] for d in diags])
        total_steps = avg_combat + avg_nav + avg_search + avg_prompt
        print(f"  Combat:  {avg_combat:>6.0f} ({100*avg_combat/max(total_steps,1):>4.1f}%)")
        print(f"  Nav:     {avg_nav:>6.0f} ({100*avg_nav/max(total_steps,1):>4.1f}%)")
        print(f"  Search:  {avg_search:>6.0f} ({100*avg_search/max(total_steps,1):>4.1f}%)")
        print(f"  Prompts: {avg_prompt:>6.0f} ({100*avg_prompt/max(total_steps,1):>4.1f}%)")
        print(f"  Stuck:   {avg_stuck:>6.0f} steps (position unchanged)")

        avg_max_stuck = np.mean([d["max_stuck_run"] for d in diags])
        print(f"  Max stuck run: {avg_max_stuck:.0f} steps avg, "
              f"{max(d['max_stuck_run'] for d in diags)} worst")

        print("\n--- Top Actions ---")
        total_actions = Counter()
        for d in diags:
            for k, v in d["action_counts"].items():
                total_actions[k] += v
        for action, count in total_actions.most_common(15):
            pct = 100 * count / sum(total_actions.values())
            print(f"  {action:>12s}: {count:>6d} ({pct:>4.1f}%)")

        print("\n--- Stuck Message Patterns ---")
        all_stuck = Counter()
        for d in diags:
            for msg, count in d["top_stuck_messages"].items():
                all_stuck[msg] += count
        if all_stuck:
            for msg, count in all_stuck.most_common(10):
                print(f"  {count:>4d}x: {msg}")
        else:
            print("  (none detected)")

    # Find worst episodes for debugging
    print("\n--- Worst Episodes (lowest score) ---")
    sorted_eps = sorted(results, key=lambda r: r["score"])
    for r in sorted_eps[:5]:
        print(f"  seed={r['seed']:>3d}: score={r['score']:>4.0f}  dl={r['dlevel']}  "
              f"xl={r['xlevel']}  steps={r['steps']}  turns={r['turn']}  "
              f"{'DIED' if r.get('died') else 'timeout'}")

    print("\n--- Best Episodes ---")
    for r in sorted_eps[-5:]:
        print(f"  seed={r['seed']:>3d}: score={r['score']:>4.0f}  dl={r['dlevel']}  "
              f"xl={r['xlevel']}  steps={r['steps']}  turns={r['turn']}  "
              f"{'DIED' if r.get('died') else 'timeout'}")

    print("=" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--analyze", action="store_true",
                   help="Enable per-step diagnostics and print QA report")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    env = make_env()
    agent = ExpertAgent(verbose=args.verbose)

    results = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed + ep
        result = run_episode(env, agent, seed=seed, verbose=args.verbose,
                           analyze=args.analyze)
        results.append(result)

        scores_so_far = [r["score"] for r in results]
        elapsed = time.time() - t0
        status = "DIED" if result.get("died") else ""
        print(f"Ep {ep+1}/{args.episodes}: score={result['score']:.0f} "
              f"dl={result['dlevel']} xl={result['xlevel']} "
              f"steps={result['steps']} turns={result['turn']} {status}| "
              f"avg={np.mean(scores_so_far):.1f} med={np.median(scores_so_far):.1f} "
              f"max={np.max(scores_so_far):.0f} [{elapsed:.0f}s]")

    env.close()

    if args.analyze:
        print_analysis(results)
    else:
        scores = [r["score"] for r in results]
        dlevels = [r["dlevel"] for r in results]
        print("\n" + "=" * 60)
        print(f"Results over {len(results)} episodes:")
        print(f"  Score:  mean={np.mean(scores):.1f}  median={np.median(scores):.1f}  "
              f"max={np.max(scores):.0f}  min={np.min(scores):.0f}  std={np.std(scores):.1f}")
        print(f"  DLevel: mean={np.mean(dlevels):.1f}  max={np.max(dlevels)}")
        died = sum(1 for r in results if r.get("died"))
        print(f"  Deaths: {died}/{len(results)}")
        print(f"  Time:   {time.time() - t0:.0f}s total, {(time.time() - t0) / len(results):.1f}s/ep")

    if args.output:
        # Strip diagnostics for JSON (too large)
        save_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != "diagnostics"}
            if "diagnostics" in r:
                sr["diagnostics_summary"] = {
                    "stuck_steps": r["diagnostics"]["stuck_steps"],
                    "max_stuck_run": r["diagnostics"]["max_stuck_run"],
                    "search_steps": r["diagnostics"]["search_steps"],
                    "combat_steps": r["diagnostics"]["combat_steps"],
                }
            save_results.append(sr)
        with open(args.output, "w") as f:
            json.dump({"episodes": save_results}, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
