"""Random Network Distillation (RND) exploration bonus.

Reference:
    Burda et al. 2018, "Exploration by Random Network Distillation"
    https://arxiv.org/abs/1810.12894

RND trains a predictor network to match the outputs of a fixed, randomly
initialised target network. The per-sample MSE between predictor and target
is an exploration bonus: high on novel states (poorly predicted) and low on
states the predictor has seen often (well fitted).

Usage
-----
The input is expected to be the policy encoder's output embedding — i.e.
``Agent.encoder(obs)`` / ``out["features"]``. Reusing the shared encoder
avoids a second forward pass and keeps "novelty" defined in the same
representation space as the policy's features.

The feature dim ``F`` is kept smaller than the embedding dim ``D`` (default
``F = 64`` for ``D = 256``). A smaller ``F`` makes the distillation target
cheap to fit in aggregate while still leaving a useful per-sample residual
on novel inputs.

This module is stateful only in the predictor's weights. It is intentionally
not wired into the trainer here — call ``intrinsic_reward`` to read the bonus
and ``distill_loss`` to get the scalar loss you minimise.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _rnd_mlp(in_dim: int, feat_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(True),
        nn.Linear(hidden, hidden), nn.ReLU(True),
        nn.Linear(hidden, feat_dim),
    )


class RNDTarget(nn.Module):
    """Frozen random-init MLP producing a ``(B, F)`` embedding.

    All parameters have ``requires_grad=False`` — the target is never trained.
    """

    def __init__(self, in_dim: int = 256, feat_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.net = _rnd_mlp(in_dim, feat_dim, hidden)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class RNDPredictor(nn.Module):
    """Trainable MLP with the same output shape as :class:`RNDTarget`."""

    def __init__(self, in_dim: int = 256, feat_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.net = _rnd_mlp(in_dim, feat_dim, hidden)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class RNDModule(nn.Module):
    """Bundles the frozen target and trainable predictor.

    Parameters
    ----------
    in_dim:
        Size of the policy encoder output embedding ``D`` (default 256).
    feat_dim:
        Size of the predicted feature vector ``F`` (default 64). Kept below
        ``in_dim`` per the original paper's recommendation.
    hidden:
        Hidden width shared by both networks.
    """

    def __init__(self, in_dim: int = 256, feat_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.target = RNDTarget(in_dim, feat_dim, hidden)
        self.predictor = RNDPredictor(in_dim, feat_dim, hidden)

    def _squared_error(self, emb: torch.Tensor) -> torch.Tensor:
        """``(B, F)`` elementwise squared error between predictor and target."""
        with torch.no_grad():
            t = self.target(emb)
        p = self.predictor(emb)
        return (p - t).pow(2)

    def intrinsic_reward(self, emb: torch.Tensor) -> torch.Tensor:
        """Per-sample exploration bonus.

        Returns a ``(B,)`` tensor of squared-error distances between predictor
        and (frozen) target features for each row of ``emb``. No gradient
        should flow through this bonus into the policy — detach at the call
        site if you add it to the reward.
        """
        return self._squared_error(emb).mean(dim=-1)

    def distill_loss(self, emb: torch.Tensor) -> torch.Tensor:
        """Scalar MSE loss used to train the predictor.

        Minimising this drives the predictor towards the target, shrinking the
        intrinsic reward on states that appear often in the distillation batch.
        """
        return self._squared_error(emb).mean()
