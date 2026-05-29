"""Shared episode evaluation for EC experiments.

Loads a consensus model, runs N episodes of NetHackScore-v0, returns
per-episode scores. Used by E1 (evolved router) and E2 (monolithic ES).
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from nhc.env import make_env, NUM_ACTIONS, action_mask_for
from nhc.models import Agent, ConsensusHMoE


def load_consensus(
    specialist_dir: Path,
    consensus_ckpt: Path | None = None,
    device: str = "cpu",
) -> ConsensusHMoE:
    """Load frozen specialists + optional consensus checkpoint."""
    spec_files = sorted(specialist_dir.glob("specialist_*.pt"))
    sids = [f.stem for f in spec_files]

    specs = []
    masks = []
    for f in spec_files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        a = Agent(num_actions=NUM_ACTIONS)
        a.load_state_dict(blob["model"])
        a.eval()
        specs.append(a)
        # Extract action mask from scheduler if available
        sched_file = f.with_suffix(".scheduler.json")
        if sched_file.exists():
            import json
            sched = json.loads(sched_file.read_text())
            if "action_mask" in sched:
                masks.append(torch.tensor(sched["action_mask"], dtype=torch.bool))
            else:
                masks.append(torch.ones(NUM_ACTIONS, dtype=torch.bool))
        else:
            masks.append(torch.ones(NUM_ACTIONS, dtype=torch.bool))

    model = ConsensusHMoE(specs, num_actions=NUM_ACTIONS, specialist_masks=masks)

    if consensus_ckpt is not None and consensus_ckpt.exists():
        blob = torch.load(consensus_ckpt, map_location="cpu", weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()
    return model


def evaluate_episodes(
    model: ConsensusHMoE,
    num_episodes: int = 10,
    env_id: str = "NetHackScore-v0",
    max_steps: int = 5000,
    device: str = "cpu",
    deterministic: bool = False,
) -> dict:
    """Run episodes sequentially. Use evaluate_episodes_vec for speed."""
    env = make_env(env_id)
    amask = torch.tensor(env.action_mask, dtype=torch.bool, device=device)

    scores = []
    depths = []
    lengths = []

    for ep in range(num_episodes):
        obs_raw, info = env.reset()
        obs = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in obs_raw.items()
               if k in ("glyphs", "blstats", "message")}
        state = model.initial_state(1, device)
        ep_reward = 0.0
        max_depth = 1

        for step in range(max_steps):
            with torch.no_grad():
                out = model(obs, state, action_mask=amask.unsqueeze(0))
            action = out["logits"].argmax(dim=-1).item() if deterministic else \
                torch.distributions.Categorical(logits=out["logits"]).sample().item()
            state = out["state"]

            obs_raw, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            bl = obs_raw.get("blstats", None)
            if bl is not None and len(bl) > 12:
                max_depth = max(max_depth, int(bl[12]) if bl[12] > 0 else 1)

            obs = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in obs_raw.items()
                   if k in ("glyphs", "blstats", "message")}

            if terminated or truncated:
                break

        scores.append(ep_reward)
        depths.append(max_depth)
        lengths.append(step + 1)

    env.close()
    return {
        "scores": np.array(scores),
        "depths": np.array(depths),
        "lengths": np.array(lengths),
        "mean_score": float(np.mean(scores)),
        "mean_depth": float(np.mean(depths)),
    }


def evaluate_episodes_vec(
    model: ConsensusHMoE,
    num_episodes: int = 10,
    num_envs: int = 8,
    env_id: str = "NetHackScore-v0",
    max_steps: int = 5000,
    device: str = "cpu",
    deterministic: bool = False,
) -> dict:
    """Run episodes with vectorized envs. Much faster on GPU.

    Runs num_envs episodes in parallel, collects until num_episodes
    total episodes complete.
    """
    import gymnasium as gym
    from nhc.env import _register_envs, _Wrap

    _register_envs(env_id)
    fns = [(lambda: make_env(env_id)) for _ in range(num_envs)]
    vec = gym.vector.AsyncVectorEnv(fns)

    # Get action mask from first env
    probe = make_env(env_id)
    amask = torch.tensor(probe.action_mask, dtype=torch.bool, device=device)
    probe.close()
    amask_batch = amask.unsqueeze(0).expand(num_envs, -1)

    scores = []
    depths = []
    lengths = []

    obs_raw, infos = vec.reset()
    obs = {k: torch.tensor(obs_raw[k], device=device) for k in ("glyphs", "blstats", "message")}
    state = model.initial_state(num_envs, device)
    ep_rewards = np.zeros(num_envs)
    ep_depths = np.ones(num_envs, dtype=int)
    ep_lengths = np.zeros(num_envs, dtype=int)

    total_steps = 0
    while len(scores) < num_episodes and total_steps < max_steps * num_episodes:
        with torch.no_grad():
            out = model(obs, state, action_mask=amask_batch)

        if deterministic:
            actions = out["logits"].argmax(dim=-1).cpu().numpy()
        else:
            actions = torch.distributions.Categorical(logits=out["logits"]).sample().cpu().numpy()

        # Reset state for done envs before next step
        new_state = out["state"]

        obs_raw, rewards, terminateds, truncateds, infos = vec.step(actions)
        dones = terminateds | truncateds
        ep_rewards += rewards
        ep_lengths += 1
        total_steps += num_envs

        # Track depth from blstats
        bl = obs_raw.get("blstats", None)
        if bl is not None:
            for i in range(num_envs):
                if len(bl[i]) > 12 and bl[i][12] > 0:
                    ep_depths[i] = max(ep_depths[i], int(bl[i][12]))

        for i in range(num_envs):
            if dones[i]:
                scores.append(ep_rewards[i])
                depths.append(ep_depths[i])
                lengths.append(ep_lengths[i])
                ep_rewards[i] = 0.0
                ep_depths[i] = 1
                ep_lengths[i] = 0

        # Zero state for done envs
        if isinstance(new_state, dict):
            core_h, core_c = new_state["core"]
            for i in range(num_envs):
                if dones[i]:
                    core_h[:, i] = 0
                    core_c[:, i] = 0
                    for k in range(len(new_state["spec"])):
                        sh, sc = new_state["spec"][k]
                        sh[i] = 0
                        sc[i] = 0
                    new_state["prev_option"][i] = 0

        state = new_state
        obs = {k: torch.tensor(obs_raw[k], device=device) for k in ("glyphs", "blstats", "message")}

    vec.close()
    scores = scores[:num_episodes]
    depths = depths[:num_episodes]
    lengths = lengths[:num_episodes]
    return {
        "scores": np.array(scores),
        "depths": np.array(depths),
        "lengths": np.array(lengths),
        "mean_score": float(np.mean(scores)),
        "mean_depth": float(np.mean(depths)),
    }


def get_router_params(model: ConsensusHMoE) -> np.ndarray:
    """Extract option_head + lambda_head params as a flat vector."""
    params = []
    for name, p in model.named_parameters():
        if "option_head" in name or "lambda_head" in name:
            params.append(p.data.cpu().numpy().ravel())
    return np.concatenate(params)


def set_router_params(model: ConsensusHMoE, flat: np.ndarray) -> None:
    """Set option_head + lambda_head params from a flat vector."""
    offset = 0
    for name, p in model.named_parameters():
        if "option_head" in name or "lambda_head" in name:
            n = p.numel()
            p.data.copy_(torch.from_numpy(flat[offset:offset + n]).reshape(p.shape))
            offset += n


def get_all_head_params(model: ConsensusHMoE) -> np.ndarray:
    """Extract option_head + lambda_head + policy head params as a flat vector."""
    params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ("option_head", "lambda_head", "policy")):
            if "specialists" not in name:
                params.append(p.data.cpu().numpy().ravel())
    return np.concatenate(params)


def set_all_head_params(model: ConsensusHMoE, flat: np.ndarray) -> None:
    """Set option_head + lambda_head + policy head params from a flat vector."""
    offset = 0
    for name, p in model.named_parameters():
        if any(k in name for k in ("option_head", "lambda_head", "policy")):
            if "specialists" not in name:
                n = p.numel()
                p.data.copy_(torch.from_numpy(flat[offset:offset + n]).reshape(p.shape))
                offset += n
