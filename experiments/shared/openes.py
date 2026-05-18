"""OpenAI-style Evolution Strategy (Salimans et al., 2017).

Scales to any parameter count. O(n) per generation, no covariance matrix.
Uses antithetic sampling (mirrored noise) for variance reduction and
Adam optimizer for the mean update.

Used by E1 (527K), E2 (820K), E4 (24K), E13 (24K).
CMA-ES (cmaes.py) is only viable below ~5K params.
"""
from __future__ import annotations

import numpy as np


class OpenES:
    """OpenAI Evolution Strategy with Adam optimizer."""

    def __init__(self, x0: np.ndarray, sigma: float = 0.1,
                 pop_size: int = 44, lr: float = 0.01,
                 decay: float = 0.999, sigma_decay: float = 1.0,
                 sigma_min: float = 0.01, antithetic: bool = True):
        self.n = len(x0)
        self.mean = x0.copy().astype(np.float64)
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.lr = lr
        self.decay = decay
        self.antithetic = antithetic

        # Population size must be even for antithetic sampling
        self.lam = pop_size if not antithetic else pop_size + (pop_size % 2)
        self.half = self.lam // 2

        # Adam state
        self.m = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.gen = 0

        # Store noise for gradient computation
        self._noise = None

    def ask(self) -> np.ndarray:
        """Sample population around current mean."""
        if self.antithetic:
            half_noise = np.random.randn(self.half, self.n)
            self._noise = np.concatenate([half_noise, -half_noise], axis=0)
        else:
            self._noise = np.random.randn(self.lam, self.n)
        return self.mean + self.sigma * self._noise

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """Update mean using fitness-weighted gradient estimate. Maximizes fitness."""
        # Fitness shaping (rank-based)
        ranks = np.zeros(self.lam)
        order = np.argsort(-fitnesses)
        for i, idx in enumerate(order):
            ranks[idx] = i
        shaped = np.maximum(0, np.log(self.lam / 2 + 1) - np.log(1 + ranks))
        shaped = shaped / (shaped.sum() + 1e-8) - 1.0 / self.lam

        # Gradient estimate
        grad = (self._noise.T @ shaped) / (self.lam * self.sigma)

        # Adam update
        self.gen += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.gen)
        v_hat = self.v / (1 - self.beta2 ** self.gen)
        step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        self.mean += step

        # Weight decay
        self.mean *= self.decay

        # Sigma decay
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
