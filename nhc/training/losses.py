"""V-trace + actor-critic losses, in one short file.

The legacy code spread these across `training/losses.py` (V-trace + helpers,
~120 lines), `training/specialist_trainer.py` (entropy + PG + value, mixed
into the loop), and `training/consensus_trainer.py` (kickstart + option
bonus, also inline). Here they are pure functions.

Convention: tensors are (T, B) for time-major batches. V-trace returns
`vs` aligned with values (length T) and `pg_advantages` for the PG term.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VTraceReturns:
    vs: torch.Tensor                # (T, B) — V-trace value targets
    pg_advantages: torch.Tensor     # (T, B) — clipped advantages for PG


def vtrace(
    behavior_log_probs: torch.Tensor,   # (T, B)
    target_log_probs: torch.Tensor,     # (T, B)
    rewards: torch.Tensor,              # (T, B)
    dones: torch.Tensor,                # (T, B) in {0, 1}
    values: torch.Tensor,               # (T, B) — V(s_t) under behavior
    bootstrap_value: torch.Tensor,      # (B,)   — V(s_T)
    discount: float = 0.99,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
) -> VTraceReturns:
    """Espeholt et al. 2018, equations 1-3. Pure-PyTorch, runs on GPU.

    Notes:
      - `dones[t]` means s_{t+1} is terminal; the discount factor at t is
        gamma * (1 - dones[t]).
      - We compute in fp32 for numerical stability — the trainer will cast
        back if needed.
    """
    with torch.no_grad():
        log_rhos = target_log_probs - behavior_log_probs
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=rho_clip)
        clipped_cs = torch.clamp(rhos, max=c_clip)

        T = rewards.shape[0]
        gamma = discount * (1.0 - dones)                       # (T, B)
        # Bootstrap series of values: v_{t+1} for t = 0..T-1
        next_values = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
        deltas = clipped_rhos * (rewards + gamma * next_values - values)

        # Backwards V-trace recursion: vs - V = delta + gamma*c*(vs_{t+1} - V_{t+1})
        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v = torch.zeros_like(values)
        for t in range(T - 1, -1, -1):
            acc = deltas[t] + gamma[t] * clipped_cs[t] * acc
            vs_minus_v[t] = acc

        vs = vs_minus_v + values
        vs_next = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        pg_advantages = clipped_rhos * (rewards + gamma * vs_next - values)

    return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def policy_gradient_loss(
    target_log_probs: torch.Tensor,     # (T, B)
    advantages: torch.Tensor,           # (T, B) — detached upstream
) -> torch.Tensor:
    return -(advantages.detach() * target_log_probs).mean()


def value_loss(
    values: torch.Tensor,               # (T, B)
    targets: torch.Tensor,              # (T, B) — detached
) -> torch.Tensor:
    return 0.5 * (targets.detach() - values).pow(2).mean()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Returns positive H; trainer uses -coef * H to maximise entropy."""
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    return -(p * log_p).sum(-1).mean()


def kickstart_loss(
    student_logits: torch.Tensor,       # (T, B, A)
    teacher_logits: torch.Tensor,       # (T, B, A) — detached upstream
    action_mask: torch.Tensor | None = None,   # (B, A) bool or (T, B, A)
) -> torch.Tensor:
    """KL(teacher || student). Used to bootstrap the consensus model from
    its frozen specialists (Matthews et al. 2022)."""
    teacher_logits = teacher_logits.detach()
    if action_mask is not None:
        if action_mask.dim() == 2:
            mask = action_mask.unsqueeze(0).expand_as(student_logits)
        else:
            mask = action_mask
        student_logits = student_logits.masked_fill(~mask, -1e9)
        teacher_logits = teacher_logits.masked_fill(~mask, -1e9)
    log_s = F.log_softmax(student_logits, dim=-1)
    log_t = F.log_softmax(teacher_logits, dim=-1)
    p_t = log_t.exp()
    return (p_t * (log_t - log_s)).sum(-1).mean()


def option_entropy_bonus(option_logits: torch.Tensor) -> torch.Tensor:
    """Encourage the consensus to actually use multiple options instead of
    collapsing to argmax 1. Returns positive H."""
    return entropy_loss(option_logits)


def load_balance_loss(
    option_probs: torch.Tensor,         # (T, B, K) softmax over experts
    option_assignments: torch.Tensor,   # (T, B) int in [0, K)
    num_experts: int,
) -> torch.Tensor:
    """Switch Transformer load-balancing auxiliary loss (Fedus et al. 2021,
    eq. 3 / eq. 4 in https://arxiv.org/abs/2101.03961).

    Computes `N * sum_i(f_i * P_i)` where `f_i` is the fraction of tokens
    dispatched to expert i and `P_i` is the mean router probability for
    expert i. The α-scaling is applied by the caller when summing into the
    total loss.
    """
    N = num_experts
    assignments_flat = option_assignments.reshape(-1).long()
    counts = torch.bincount(assignments_flat, minlength=N).to(option_probs.dtype)
    f = counts / float(assignments_flat.numel())
    P = option_probs.reshape(-1, N).mean(dim=0)
    return N * (f * P).sum()


def router_z_loss(option_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss (Switch Transformer appendix; also ST-MoE Zoph et al.
    2022, https://arxiv.org/abs/2202.08906 eq. 5). Penalises the router
    logits from growing unboundedly by squaring the per-token log-partition.

    Returns `mean(logsumexp(logits, dim=-1) ** 2)` over all (T, B) tokens.
    """
    lse = torch.logsumexp(option_logits, dim=-1)
    return (lse ** 2).mean()


def option_stickiness_loss(
    option_logits: torch.Tensor,        # (T, B, K)
    prev_option: torch.Tensor,          # (T, B) long — option chosen at t-1
    dones: torch.Tensor | None = None,  # (T, B) — exclude episode boundaries
) -> torch.Tensor:
    """Cross-entropy loss with target = previous option's index.

    Pulls the router toward staying on the previously-selected option.
    Counter-pressure to the entropy-maximizing option_entropy_bonus when
    we want temporal commitment (Sutton 1999 options framework) instead of
    sparse-MoE-style flat distribution. Mean over (T, B) tokens.

    If ``dones`` is provided, samples where the previous step ended an
    episode are masked out — there's no meaningful "previous option" across
    a reset.
    """
    T, B, K = option_logits.shape
    log_probs = F.log_softmax(option_logits, dim=-1)               # (T, B, K)
    nll = -log_probs.gather(-1, prev_option.unsqueeze(-1)).squeeze(-1)  # (T, B)
    if dones is not None:
        # `prev_option` at t comes from the option chosen at t-1; it is
        # invalid at the first step after a done. Mask those out.
        mask = torch.ones_like(nll)
        mask[1:] = 1.0 - dones[:-1].float()
        denom = mask.sum().clamp(min=1.0)
        return (nll * mask).sum() / denom
    return nll.mean()


def gradient_norm(parameters) -> float:
    """Total L2 norm of parameter grads. For logging only."""
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


# =========================================================================
# PPO + GAE (Schulman 2015, 2017)
# =========================================================================
#
# Everything below replaces the V-trace path. The trainer stays fully
# synchronous (rollouts collected with current weights), so there's no
# actor staleness for V-trace's importance correction to compensate for.
# PPO+GAE is the on-policy default and the NLE-literature choice.


def compute_gae(
    rewards: torch.Tensor,            # (T, B)
    values: torch.Tensor,             # (T, B) — behavior-time critic
    bootstrap_value: torch.Tensor,    # (B,)   — V(s_T)
    dones: torch.Tensor,              # (T, B) in {0, 1}
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman 2015).

    Mirrors the V-trace bootstrap pattern: the next-step value at t=T-1
    is ``bootstrap_value``; elsewhere it is ``values[t+1]``.

    Recursion::

        delta_t  = r_t + gamma * (1 - done_t) * V_{t+1} - V_t
        adv_t    = delta_t + gamma * lam * (1 - done_t) * adv_{t+1}
        ret_t    = adv_t + V_t

    Returns ``(advantages, returns)``, both shape ``(T, B)``, both
    detached from the graph (downstream treats them as constants).
    """
    with torch.no_grad():
        T = rewards.shape[0]
        next_values = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
        nonterminal = 1.0 - dones
        deltas = rewards + gamma * nonterminal * next_values - values

        advantages = torch.zeros_like(rewards)
        acc = torch.zeros_like(bootstrap_value)
        for t in range(T - 1, -1, -1):
            acc = deltas[t] + gamma * lam * nonterminal[t] * acc
            advantages[t] = acc
        returns = advantages + values
    return advantages, returns


def ppo_policy_loss(
    new_log_probs: torch.Tensor,      # (T, B) or any shape, requires_grad
    old_log_probs: torch.Tensor,      # same shape, detached upstream
    advantages: torch.Tensor,         # same shape, detached upstream
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate loss (Schulman 2017, eq. 7).

    ``ratio = exp(new - old)``; the loss is
    ``-min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).mean()``.

    Returns ``(loss, stats)`` with stats: ``clip_frac``, ``approx_kl``,
    ``ratio_mean``, ``ratio_max``. ``approx_kl`` is the cheap
    ``(old - new).mean()`` approximation (Schulman's blog); it's
    biased low but it's what every PPO implementation logs.
    """
    old_lp = old_log_probs.detach()
    adv = advantages.detach()
    log_ratio = new_log_probs - old_lp
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    loss = -torch.min(surr1, surr2).mean()
    with torch.no_grad():
        clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
        stats = {
            "clip_frac": float(clipped.float().mean()),
            "approx_kl": float((-log_ratio).mean()),
            "ratio_mean": float(ratio.mean()),
            "ratio_max": float(ratio.max()),
        }
    return loss, stats


def ppo_value_loss(
    new_values: torch.Tensor,         # (T, B), requires_grad
    old_values: torch.Tensor,         # (T, B), detached upstream
    returns: torch.Tensor,            # (T, B), detached upstream
    clip_eps: float | None = 0.2,
) -> tuple[torch.Tensor, float]:
    """Clipped value loss (Schulman 2017; OpenAI baselines convention).

    ``v_clipped = old_values + clamp(new_values - old_values, -eps, +eps)``
    ``loss     = 0.5 * max((new_values - returns)^2,
                            (v_clipped - returns)^2).mean()``

    When ``clip_eps is None``, falls back to plain MSE. Engstrom 2020
    ("PPO Implementation Matters") found value clipping rarely helps
    and can hurt; Sample-Factory APPO disables it by default. Our v1
    run showed ``value_clip_frac`` held at 12–22% throughout training
    with ``explained_variance`` plateauing at 0.80, consistent with the
    clip actively limiting the critic.

    Returns ``(loss, clip_frac)``; when clipping is disabled,
    ``clip_frac`` is 0.0.
    """
    ret = returns.detach()
    if clip_eps is None:
        loss = 0.5 * (new_values - ret).pow(2).mean()
        return loss, 0.0
    old = old_values.detach()
    v_clipped = old + torch.clamp(new_values - old, -clip_eps, clip_eps)
    sq_uncl = (new_values - ret).pow(2)
    sq_cl = (v_clipped - ret).pow(2)
    loss = 0.5 * torch.max(sq_uncl, sq_cl).mean()
    with torch.no_grad():
        clip_frac = float((sq_cl > sq_uncl).float().mean())
    return loss, clip_frac


def explained_variance(
    returns: torch.Tensor,            # (T, B)
    values: torch.Tensor,             # (T, B)
) -> float:
    """``1 - Var(returns - values) / Var(returns)``. Standard PPO diagnostic.

    ~1 means the critic explains the returns; ~0 means it's no better
    than predicting the mean; < 0 means it's actively misleading.
    """
    with torch.no_grad():
        var_ret = returns.var()
        if float(var_ret) < 1e-8:
            return 0.0
        return float(1.0 - (returns - values).var() / var_ret)


class RunningMeanStd:
    """Welford-style running mean/variance for tensors of arbitrary shape.

    Used by the APPO trainer to normalize returns — Sample-Factory's
    ``normalize_returns=True`` default, and one of the top implementation
    details in Huang 2022 §4. Call ``.update(x)`` once per rollout with
    the new returns batch; then divide returns/advantages by
    ``sqrt(rms.var + eps)`` before using them for the value loss.

    The estimate is scalar (we track mean/var over all elements flattened)
    because NetHack reward scale is env-global, not per-dimension.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self._mean = 0.0
        self._var = 1.0
        self._count = eps   # avoids /0 before the first update

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False))
        batch_count = int(x.numel())
        self._merge(batch_mean, batch_var, batch_count)

    def _merge(self, bm: float, bv: float, bn: int) -> None:
        # Chan-et-al parallel variance combination (Wikipedia: Welford).
        delta = bm - self._mean
        total = self._count + bn
        new_mean = self._mean + delta * (bn / total)
        m_a = self._var * self._count
        m_b = bv * bn
        m2 = m_a + m_b + (delta ** 2) * (self._count * bn / total)
        self._mean = new_mean
        self._var = m2 / total
        self._count = total

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def var(self) -> float:
        return self._var

    @property
    def std(self) -> float:
        return float((self._var + 1e-8) ** 0.5)

    def state_dict(self) -> dict:
        return {"mean": self._mean, "var": self._var, "count": self._count}

    def load_state_dict(self, sd: dict) -> None:
        self._mean = float(sd.get("mean", 0.0))
        self._var = float(sd.get("var", 1.0))
        self._count = float(sd.get("count", 1e-8))
