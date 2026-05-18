"""Rollout buffer + numpy → torch helpers used by the local trainers.

The local trainers (``train_specialist``, ``train_consensus``) build
:class:`Rollout` directly from pre-allocated GPU buffers;
``rollout_from_numpy`` is kept for tests / utilities that construct
rollouts from the numpy bundle format.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Rollout:
    """A (T, B, ...) block of transitions with enough metadata to
    recompute logits under the target policy and do V-trace."""

    obs: dict[str, torch.Tensor]          # each key: (T, B, ...)
    actions: torch.Tensor                 # (T, B)
    behavior_log_probs: torch.Tensor      # (T, B) — logp under the policy that collected this
    rewards: torch.Tensor                 # (T, B)
    dones: torch.Tensor                   # (T, B) float in {0, 1}
    values: torch.Tensor                  # (T, B) — V(s_t) under behavior policy
    bootstrap_value: torch.Tensor         # (B,)  — V(s_T)
    action_mask: torch.Tensor             # (B, A) bool
    init_state: tuple[torch.Tensor, torch.Tensor]  # (h, c) at t=0

    @property
    def T(self) -> int:
        return int(self.actions.shape[0])

    @property
    def B(self) -> int:
        return int(self.actions.shape[1])

    def to(self, device: torch.device) -> "Rollout":
        return Rollout(
            obs={k: v.to(device, non_blocking=True) for k, v in self.obs.items()},
            actions=self.actions.to(device, non_blocking=True),
            behavior_log_probs=self.behavior_log_probs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            dones=self.dones.to(device, non_blocking=True),
            values=self.values.to(device, non_blocking=True),
            bootstrap_value=self.bootstrap_value.to(device, non_blocking=True),
            action_mask=self.action_mask.to(device, non_blocking=True),
            init_state=(self.init_state[0].to(device, non_blocking=True),
                        self.init_state[1].to(device, non_blocking=True)),
        )


def stack_obs_list(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of per-step dict obs along T=0."""
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}


def numpy_obs_to_torch(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "glyphs": torch.from_numpy(np.ascontiguousarray(obs["glyphs"], dtype=np.int16)).to(device, non_blocking=True),
        "blstats": torch.from_numpy(np.ascontiguousarray(obs["blstats"], dtype=np.float32)).to(device, non_blocking=True),
        "message": torch.from_numpy(np.ascontiguousarray(obs["message"], dtype=np.uint8)).to(device, non_blocking=True),
    }


def rollout_from_numpy(bundle: dict, device: torch.device) -> "Rollout":
    """Convert a numpy rollout bundle to a :class:`Rollout` on ``device``.

    Bundle layout:
      ``obs.{glyphs,blstats,message}``  (T, B, ...)
      ``actions``                        (T, B) int64
      ``behavior_log_probs``             (T, B) float32
      ``rewards``                        (T, B) float32
      ``dones``                          (T, B) float32
      ``values``                         (T, B) float32
      ``bootstrap_value``                (B,)   float32
      ``action_mask``                    (T, B, A) bool — per-step because
                                         curriculum slots can be on different envs
      ``init_state_h, init_state_c``     (B, D) float32
      ``features``                       (T, B, D) float32 (for RND distill)

    The 3-D ``action_mask`` shape passes straight through to
    ``Agent.forward_sequence`` which already handles it.
    """
    obs_seq = {
        "glyphs": torch.from_numpy(np.ascontiguousarray(bundle["obs"]["glyphs"])).to(device, non_blocking=True),
        "blstats": torch.from_numpy(np.ascontiguousarray(bundle["obs"]["blstats"])).to(device, non_blocking=True),
        "message": torch.from_numpy(np.ascontiguousarray(bundle["obs"]["message"])).to(device, non_blocking=True),
    }
    rollout = Rollout(
        obs=obs_seq,
        actions=torch.from_numpy(bundle["actions"]).to(device, non_blocking=True),
        behavior_log_probs=torch.from_numpy(bundle["behavior_log_probs"]).to(device, non_blocking=True),
        rewards=torch.from_numpy(bundle["rewards"]).to(device, non_blocking=True),
        dones=torch.from_numpy(bundle["dones"]).to(device, non_blocking=True),
        values=torch.from_numpy(bundle["values"]).to(device, non_blocking=True),
        bootstrap_value=torch.from_numpy(bundle["bootstrap_value"]).to(device, non_blocking=True),
        action_mask=torch.from_numpy(bundle["action_mask"]).to(device, non_blocking=True),
        init_state=(torch.from_numpy(bundle["init_state_h"]).to(device, non_blocking=True),
                    torch.from_numpy(bundle["init_state_c"]).to(device, non_blocking=True)),
    )
    rollout.behavior_features = torch.from_numpy(bundle["features"]).to(device, non_blocking=True)
    rollout.episode_outcomes = bundle.get("episode_outcomes", [])
    rollout.slot_levels = bundle.get("slot_levels", [])
    return rollout
