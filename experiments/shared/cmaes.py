"""CMA-ES implementation for EC experiments.

Used by E4 (failed expert competences) where param count is ~24K.
OpenES is used for larger param counts (E1, E2).
"""
from __future__ import annotations

import numpy as np


class CMAES:
    """Minimal CMA-ES (Hansen 2006). No dependencies."""

    def __init__(self, x0: np.ndarray, sigma0: float = 0.1,
                 pop_size: int | None = None):
        self.n = len(x0)
        self.mean = x0.copy().astype(np.float64)
        self.sigma = sigma0
        self.lam = pop_size or (4 + int(3 * np.log(self.n)))
        self.mu = self.lam // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                       ((self.n + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        self.chi_n = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.eigenvalues = np.ones(self.n)
        self.eigenvectors = np.eye(self.n)
        self.gen = 0
        self._decompose_counter = 0

    def ask(self) -> np.ndarray:
        z = np.random.randn(self.lam, self.n)
        return self.mean + self.sigma * (z @ self.eigenvectors.T * self.eigenvalues)

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """Update from evaluated solutions. Maximizes fitness."""
        idx = np.argsort(-fitnesses)
        best = solutions[idx[:self.mu]]

        old_mean = self.mean.copy()
        self.mean = (self.weights[:, None] * best).sum(axis=0)
        diff = (self.mean - old_mean) / self.sigma

        invsqrtC = self.eigenvectors @ np.diag(1.0 / self.eigenvalues) @ self.eigenvectors.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (invsqrtC @ diff)

        hs = (np.linalg.norm(self.ps) /
              np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chi_n
              < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * diff

        artmp = (best - old_mean) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (np.outer(self.pc, self.pc) +
                             (1 - hs) * self.cc * (2 - self.cc) * self.C) +
                  self.cmu * (artmp.T @ np.diag(self.weights) @ artmp))

        self.sigma *= np.exp((self.cs / self.damps) *
                             (np.linalg.norm(self.ps) / self.chi_n - 1))

        self._decompose_counter += 1
        if self._decompose_counter >= max(1, self.lam / (self.c1 + self.cmu) / self.n / 10):
            self._decompose_counter = 0
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D2, B = np.linalg.eigh(self.C)
            D2 = np.maximum(D2, 1e-20)
            self.eigenvalues = np.sqrt(D2)
            self.eigenvectors = B

        self.gen += 1
