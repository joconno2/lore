"""Local-only V-trace trainer for specialists and the consensus.

Compute model: a single 5090 box, 14 GiB host RAM. Env stepping (CPU
subprocesses) and the learner (GPU) live in the same Python process —
no Ray, no remote actors. Specialists train **one at a time** (the
prior concurrent multi-spec trainer was removed after a 14 GiB-host
crash; aggregate-SPS gains were too small to justify the OOM risk).

The throughput recipe shared by both trainers:

  - bf16 autocast on the per-step rollout forward
  - ``torch.compile`` of ``model.forward`` (default mode; LSTM graph-breaks
    are tolerated, the encoder/conv/linear stages still fuse)
  - Pre-allocated GPU rollout buffers (no per-step list-append + np.stack)
  - Pinned-host action buffer + non-blocking D2H so the env-step CPU side
    starts working before the GPU launches the next forward
  - :class:`nhc.env.BatchedAsyncVectorEnv` for pinned-env training
    (consensus, finetune) — W subprocesses each owning ``num_envs / W``
    real NLE envs, ~5 GB at W=8/N=128.
  - :class:`nhc.training.env_pool.LocalCurriculumVecEnv` for specialist
    pretraining — one subprocess per slot, each owning its own
    :class:`nhc.curriculum.CurriculumScheduler` (frontier + review).

Two public entry points:

  - :func:`train_specialist` — one specialist, all CPU env workers
    dedicated to its pool. Used for both pretraining (curriculum mode)
    and finetune (``env_id_override`` mode).
  - :func:`train_consensus` — the HO-MoE consensus, single env_id
    (NetHackScore-v0). Was previously ``train_consensus_local``.

V-trace (:func:`nhc.training.losses.vtrace`) and the per-rollout learn
step bodies (:func:`_learn_specialist`, :func:`_learn_consensus`) are
unchanged from the prior code — verified end-to-end against Espeholt
et al. 2018 equations 1-3.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Expandable-segments allocator: reduces CUDA-memory fragmentation when
# the consensus's big encoder + N=128 envs push GPU usage near the 32 GB
# ceiling. Setdefault so callers can override if they know better. Must
# be set before torch imports cuda state.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

try:
    from omegaconf import MISSING
except ImportError:
    MISSING: Any = "???"

from nhc.env import (
    NUM_ACTIONS, OBS_KEYS, action_mask_for, make_vec_env, obs_to_tensor,
)
from nhc.models import Agent, ConsensusHMoE, KBConditioner
from nhc.rnd import RNDModule
from nhc.store import RunDir
from nhc.training import losses
from nhc.training.env_pool import LocalCurriculumVecEnv, LocalPinnedVecEnv
from nhc.training.rollout import Rollout

logger = logging.getLogger(__name__)


# =========================================================================
# Config
# =========================================================================


@dataclass
class TrainConfig:
    """Structured config for the learner.

    Doubles as a Hydra structured-config schema — registered with the
    ``ConfigStore`` below under group=``train``/name=``base`` so YAMLs
    can inherit it.
    """

    sid: str = MISSING
    run_id: str = MISSING

    # Per-spec env pool shape. ``num_envs`` is the total batch B fed into
    # the GPU forward each step. ``num_env_workers`` is the W subprocess
    # count of the pinned-env BatchedAsyncVectorEnv backend; the curriculum
    # backend ignores it (it always uses one subprocess per slot, since
    # each slot has its own scheduler).
    num_envs: int = 64
    num_env_workers: int = 8
    rollout_len: int = 64

    # Budget.
    total_steps: int = 3_000_000

    # Optimisation.
    learning_rate: float = 3e-4
    discount: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 40.0

    # PPO+GAE — replaces the V-trace path (see HANDOFF_PPO.md §4).
    # Per rollout: compute advantages once with GAE, then do ppo_epochs
    # passes over the rollout split into ppo_minibatches minibatches
    # along the B axis. The clips use the standard Schulman 2017
    # formulation.
    #
    # v2 defaults reflect the v1 post-mortem (PPO_V2_PLAN.md §3a/b):
    # - value_clip defaults to None (disabled); v1 held value_clip_frac
    #   at 12-22% throughout training while explained_variance plateaued
    #   at 0.80 — the clip was holding the critic back.
    # - ppo_clip 0.2 (Schulman 2017 original); v1 at 0.1 never fired
    #   anyway (clip_frac ~ 0.005).
    ppo_clip: float = 0.2
    value_clip: float | None = None
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_minibatches: int = 4

    # Learning-rate schedule — linear warmup to ``learning_rate`` over
    # ``lr_warmup_steps`` env-steps, then cosine decay to ``lr_min``
    # over the remainder of ``total_steps``. Set ``lr_warmup_steps=0``
    # to skip warmup; set ``lr_min = learning_rate`` to disable decay.
    lr_warmup_steps: int = 0
    lr_min: float | None = None

    # Entropy-coefficient anneal. Linear from ``entropy_coef`` to
    # ``entropy_coef_final`` over ``entropy_coef_anneal_steps``
    # env-steps. ``entropy_coef_final=None`` disables the anneal.
    entropy_coef_final: float | None = None
    entropy_coef_anneal_steps: int = 0

    # Adam epsilon. Sample-Factory NLE APPO uses 1e-7; PyTorch default
    # is 1e-8. Huang 2022 lists adam_eps as one of the reproducibility
    # levers for PPO.
    adam_eps: float = 1e-8

    # Returns normalization — Sample-Factory APPO default. Maintains a
    # running mean/std of the GAE returns; scales returns by
    # sqrt(var + 1e-8) before the value loss so the critic sees
    # approximately unit-variance targets. When ``True``, trainer
    # instantiates a ``RunningMeanStd`` and updates it per rollout.
    normalize_returns: bool = False

    # Anti-forgetting curriculum review probability. 0.25 = each env reset
    # on a slot draws from previously-solved levels with prob 0.25.
    p_review: float = 0.25

    # NLE-style tanh reward clipping: ``r_clipped = tanh(r / scale)``.
    # 0.0 disables. Smooth saturation — retained for backwards-compat
    # with older configs but not used by the SF NLE recipe.
    reward_clip_tanh_scale: float = 0.0

    # Sample-Factory-style hard reward clipping: ``r = clamp(r, -c, +c)``.
    # ``None`` disables. SF NLE APPO (Dungeons & Data) uses 10.0, which
    # is also our v4 default — per-step reward is rarely above 10 anyway
    # (most bursts come from rare monster-kill / level-descent events),
    # but capping protects the value head from single-step outliers that
    # would otherwise spike loss/value under normalize_returns.
    reward_clip: float | None = None

    # KB-conditioned agent. When True, attach a KBConditioner to the Agent.
    use_kb: bool = False
    kb_dim: int = 64
    kb_num_rules: int = 80

    # Consensus-only (ignored by specialist trainer).
    kickstart_coef: float = 0.0
    option_entropy_coef: float = 0.0
    load_balance_coef: float = 0.01
    router_z_coef: float = 0.001
    # Cross-entropy of option_logits_t with target = option chosen at t-1.
    # Pushes the router to commit to its previous selection — counter to
    # the entropy/load-balance pressures which push toward uniform. v5
    # default is 0.001; set to 0.0 to disable. Set option_entropy_coef
    # and load_balance_coef to 0 alongside this for the cleanest options
    # framework setup.
    option_stickiness_coef: float = 0.0
    # Anneal option_entropy and load_balance coefficients from their
    # starting value down to 1/3 of that over the first N steps. Strong
    # diversity pressure early seeds router spread; gentler pressure
    # later lets the router specialise. 0 disables the anneal.
    router_coef_anneal_steps: int = 100_000_000

    # RND intrinsic exploration (Burda 2018). Set rnd_coef > 0 to train
    # the predictor; rnd_intrinsic_scale controls the bonus added to
    # extrinsic reward before V-trace.
    rnd_coef: float = 0.0
    rnd_feat_dim: int = 64
    rnd_intrinsic_scale: float = 1.0

    # Pin every slot to a single env_id, bypassing CURRICULA[sid] and
    # disabling review/advance/regress. Used by the finetune phase to
    # adapt pretrained specialists to a target env (e.g.
    # NetHackScore-v0) without changing their ``sid``.
    env_id_override: str | None = None

    # Reset the resumed step/update counters to 0 after loading weights.
    # Used by the finetune phase: we want the pretrained weights but a
    # fresh ``total_steps`` budget (the baseline ckpt's step==3M would
    # short-circuit ``step < total_steps`` immediately).
    reset_step_counter: bool = False

    # Override NLE's default max_episode_steps (5000). Shorter caps force
    # the agent to score quickly instead of learning to survive passively.
    max_episode_steps: int | None = None

    # Bookkeeping.
    log_every: int = 5
    ckpt_every: int = 200
    seed: int = 0
    device: str = "cuda"
    root: str = "runs"

    @classmethod
    def from_omegaconf(cls, dc: Any) -> "TrainConfig":
        from omegaconf import DictConfig, OmegaConf
        node = dc.train if isinstance(dc, DictConfig) and "train" in dc else dc
        data = OmegaConf.to_container(node, resolve=True)
        if not isinstance(data, dict):
            raise TypeError(f"expected dict-like config, got {type(data)!r}")
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in field_names})


def _register_structured_configs() -> None:
    try:
        from hydra.core.config_store import ConfigStore
    except ImportError:
        return
    cs = ConfigStore.instance()
    cs.store(group="train", name="base", node=TrainConfig)


_register_structured_configs()


# =========================================================================
# Build helpers
# =========================================================================


def _maybe_compile(model: torch.nn.Module, *, enabled: bool):
    """Compile both ``model.forward`` (rollout-time) and
    ``model.forward_sequence`` (learn-time PPO replay) with
    ``torch.compile(default mode)`` unless disabled by NHC_NO_COMPILE=1.
    Falls back to eager on compile failure. Returns the rollout-time
    forward callable; ``forward_sequence`` is swapped in-place on the
    module so existing call sites pick it up automatically.

    Default mode (not reduce-overhead) because Dynamo can't trace
    ``nn.LSTM`` and reduce-overhead's CUDA-graph capture aliases LSTM
    hidden state across rollout steps, crashing on the second step.
    Default mode graph-breaks at the LSTM but still fuses the
    encoder/conv/embedding/linear stages — empirically a 30-40% win on
    the per-step rollout forward, and 10-25% on the forward_sequence
    replay (T×B encoder pass fuses cleanly; consensus-core t-loop
    graph-breaks per iteration but the per-step heads fuse).
    """
    if not enabled or os.environ.get("NHC_NO_COMPILE", "0") == "1":
        return model.forward
    fwd = model.forward
    try:
        compiled = torch.compile(model, dynamic=False, fullgraph=False)
        fwd = compiled.forward
    except Exception as e:  # noqa: BLE001
        logger.warning("torch.compile(forward) failed, falling back to eager: %r", e)
    # forward_sequence is the PPO learn-step replay path; monkey-patching
    # the bound method makes ``model.forward_sequence(...)`` call sites
    # in ``_learn_specialist`` / ``_learn_consensus`` pick up the
    # compiled version without a signature change.
    if hasattr(model, "forward_sequence"):
        try:
            model.forward_sequence = torch.compile(
                model.forward_sequence, dynamic=False, fullgraph=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "torch.compile(forward_sequence) failed, falling back to eager: %r", e)
    return fwd


def _make_env_pool(cfg: TrainConfig) -> tuple[Any, np.ndarray | None,
                                                dict[str, np.ndarray]]:
    """Build the per-spec env pool and return its initial obs.

    Returns ``(pool, fixed_action_mask, obs_np)``. For pinned mode
    (``env_id_override`` set) the fixed action mask is shape ``(A,)`` and
    constant for the whole run; the trainer broadcasts it to ``(B, A)``.
    For curriculum mode the mask is ``None`` and the trainer fetches a
    per-step ``(B, A)`` mask via ``pool.action_mask()``.

    The two backends have different seeding semantics (pinned uses gym's
    seed-list reset; curriculum seeds each slot's RNG at __init__), so the
    initial-obs handshake differs — handled here so the trainer doesn't
    care about the backend.
    """
    if cfg.env_id_override is not None:
        pool = LocalPinnedVecEnv(
            cfg.env_id_override,
            num_envs=cfg.num_envs,
            num_workers=cfg.num_env_workers,
            max_episode_steps=cfg.max_episode_steps,
        )
        seeds = [cfg.seed + i for i in range(cfg.num_envs)]
        obs_np, _ = pool.reset(seed=seeds)
        fixed_mask = action_mask_for(cfg.env_id_override)
        return pool, fixed_mask, obs_np
    pool = LocalCurriculumVecEnv(
        cfg.sid,
        num_envs=cfg.num_envs,
        p_review=cfg.p_review,
        seed=cfg.seed,
    )
    obs_np, _ = pool.reset()
    return pool, None, obs_np


def _make_pinned_action_buf(num_envs: int) -> torch.Tensor:
    """Pinned host buffer for actions to enable async D2H copies."""
    return torch.empty(num_envs, dtype=torch.int64, pin_memory=True)


# =========================================================================
# LR schedule + entropy-coef anneal helpers (v2)
# =========================================================================


def _current_lr(env_step: int, cfg: "TrainConfig") -> float:
    """Return the learning rate for the given env-step count.

    Default (``lr_warmup_steps == 0 and lr_min is None``): constant LR.
    With either set, applies linear warmup → cosine decay per Huang 2022
    §4 ("37 Implementation Details of PPO") and Schulman 2017 Atari:

      - env_step < warmup:             lr_max * (env_step / warmup)
      - env_step >= warmup:            lr_min + 0.5 * (lr_max - lr_min) *
                                       (1 + cos(pi * progress))
        where ``progress = (env_step - warmup) / (total_steps - warmup)``

    LR stays constant across the K PPO epochs of one update — the
    scheduler is keyed on env-steps, not optimiser steps, so per-update
    LR is fixed for all minibatches of that update.
    """
    lr_max = cfg.learning_rate
    if cfg.lr_warmup_steps <= 0 and cfg.lr_min is None:
        return lr_max
    if cfg.lr_warmup_steps > 0 and env_step < cfg.lr_warmup_steps:
        # Don't ramp from exactly zero — the first step would do nothing.
        return lr_max * max(env_step / cfg.lr_warmup_steps, 1e-3)
    if cfg.lr_min is None:
        return lr_max
    remain = max(cfg.total_steps - cfg.lr_warmup_steps, 1)
    progress = min((env_step - cfg.lr_warmup_steps) / remain, 1.0)
    return cfg.lr_min + 0.5 * (lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))


def _current_entropy_coef(env_step: int, cfg: "TrainConfig") -> float:
    """Return the entropy coefficient for the given env-step count.

    Default (``entropy_coef_final is None`` or anneal steps 0): constant
    ``cfg.entropy_coef``. Otherwise linear anneal from
    ``cfg.entropy_coef`` to ``cfg.entropy_coef_final`` over
    ``cfg.entropy_coef_anneal_steps`` env-steps.
    """
    if cfg.entropy_coef_final is None or cfg.entropy_coef_anneal_steps <= 0:
        return cfg.entropy_coef
    t = min(1.0, env_step / cfg.entropy_coef_anneal_steps)
    return cfg.entropy_coef + t * (cfg.entropy_coef_final - cfg.entropy_coef)


# =========================================================================
# Specialist trainer (single-spec)
# =========================================================================


def train_specialist(cfg: TrainConfig) -> dict:
    """Train one specialist locally.

    Per iteration:
      1. Step ``rollout_len`` env steps. Each step: ``obs_to_tensor`` (async
         H2D) → bf16 forward → categorical sample → pinned D2H of actions
         → ``vec.step``. State reset on done is folded into the step.
      2. Bootstrap V at obs[T+1], build a :class:`Rollout`, hand to
         :func:`_learn_specialist` (V-trace + PG + value + entropy).

    On-policy: rollout collection uses the *current* weights — no actor
    staleness, so V-trace clipping rarely activates.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    rd = RunDir(cfg.run_id, root=Path(cfg.root))
    kb = KBConditioner(num_rules=cfg.kb_num_rules, kb_dim=cfg.kb_dim) if cfg.use_kb else None
    model = Agent(num_actions=NUM_ACTIONS, kb_conditioner=kb).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                              eps=cfg.adam_eps)

    rnd: RNDModule | None = None
    if cfg.rnd_coef > 0:
        rnd = RNDModule(in_dim=model.hidden_dim,
                        feat_dim=cfg.rnd_feat_dim).to(device)

    start_step, start_update = _maybe_resume_specialist(model, optim, rd, cfg.sid)
    if cfg.reset_step_counter:
        start_step, start_update = 0, 0

    forward = _maybe_compile(model, enabled=use_amp)

    pool, fixed_mask, obs_np = _make_env_pool(cfg)
    state = model.initial_state(cfg.num_envs, device)

    summary = _run_single_spec_loop(
        cfg=cfg, sid=cfg.sid, model=model, optim=optim, rnd=rnd,
        forward=forward, pool=pool, fixed_mask=fixed_mask,
        obs_np=obs_np, state=state, device=device, use_amp=use_amp,
        rd=rd, start_step=start_step, start_update=start_update,
    )
    pool.close()
    return summary


def _run_single_spec_loop(
    *,
    cfg: TrainConfig,
    sid: str,
    model: Agent,
    optim: torch.optim.Optimizer,
    rnd: RNDModule | None,
    forward,
    pool: Any,
    fixed_mask: np.ndarray | None,
    obs_np: dict[str, np.ndarray],
    state: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    use_amp: bool,
    rd: RunDir,
    start_step: int,
    start_update: int,
) -> dict:
    """Inner per-rollout loop. Calls ``pool.step`` synchronously; the
    pool API is the same shape for both backends so this loop is
    backend-agnostic."""
    amp_dtype = torch.bfloat16
    T = cfg.rollout_len
    B = cfg.num_envs
    A = NUM_ACTIONS

    # Static or dynamic mask (B, A). For curriculum mode the pool exposes
    # a per-step mask which we re-fetch every step.
    if fixed_mask is not None:
        mask_b = (torch.as_tensor(fixed_mask, dtype=torch.bool, device=device)
                  .unsqueeze(0).expand(B, -1).contiguous())
    else:
        mask_b = torch.as_tensor(pool.action_mask(),
                                 dtype=torch.bool, device=device)

    # Pre-allocated GPU rollout buffers.
    ot0 = obs_to_tensor(obs_np, device)
    obs_buf_t: dict[str, torch.Tensor] = {
        k: torch.empty((T, B) + tuple(v.shape[1:]),
                       dtype=v.dtype, device=device)
        for k, v in ot0.items()
    }
    actions_buf_t = torch.empty((T, B), dtype=torch.int64, device=device)
    lp_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    value_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    reward_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    done_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    feat_dim = model.hidden_dim + (model.kb.kb_dim if model.kb is not None else 0)
    feat_buf_t = torch.empty((T, B, feat_dim),
                             dtype=torch.float32, device=device)
    mask_buf_t = (torch.empty((T, B, A), dtype=torch.bool, device=device)
                  if fixed_mask is None else None)
    pinned_action = _make_pinned_action_buf(B)

    ep_rewards = np.zeros(B, dtype=np.float64)
    ep_lengths = np.zeros(B, dtype=np.int64)

    step = start_step
    update = start_update
    t0 = time.time()
    steps_window = 0
    recent_ep_rewards: deque[float] = deque(maxlen=1000)
    best_mean_reward = float('-inf')
    last_mean_reward = 0.0

    with rd.metrics() as db:
        db.log_event(sid, step, "start_local", {
            "device": str(device), "num_envs": B, "rollout_len": T,
            "amp": use_amp, "amp_dtype": str(amp_dtype),
            "env_id_override": cfg.env_id_override,
        })

        while step < cfg.total_steps:
            init_h = state[0].clone()
            init_c = state[1].clone()

            outcomes: list[tuple[int, bool, float, int]] = []

            for t in range(T):
                ot = obs_to_tensor(obs_np, device)
                if fixed_mask is None:
                    # Pull current per-slot mask from the curriculum pool.
                    mask_b = torch.as_tensor(pool.action_mask(),
                                             dtype=torch.bool, device=device)
                    mask_buf_t[t].copy_(mask_b)
                with torch.no_grad(), torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = forward(ot, state, mask_b)
                logits = out["logits"].float()
                # bf16 autocast + a rare fp32 head overflow can still emit
                # inf/NaN; nan_to_num is a cheap bandaid so Categorical
                # never sees them. Pair with fp32 heads (see models.py).
                logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                lps = dist.log_prob(actions)

                for k in OBS_KEYS:
                    obs_buf_t[k][t].copy_(ot[k])
                actions_buf_t[t].copy_(actions)
                lp_buf_t[t].copy_(lps)
                value_buf_t[t].copy_(out["value"].float())
                feat_buf_t[t].copy_(out["features"].float())

                pinned_action.copy_(actions, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                actions_np = pinned_action.numpy().astype(np.int64, copy=False)

                obs_np, rewards_np, term_np, trunc_np, infos = pool.step(actions_np)
                done_np = np.asarray(term_np, dtype=bool) | np.asarray(trunc_np, dtype=bool)

                ep_rewards += np.asarray(rewards_np, dtype=np.float64)
                ep_lengths += 1

                reward_buf_t[t].copy_(
                    torch.from_numpy(np.ascontiguousarray(rewards_np, dtype=np.float32)),
                    non_blocking=True)
                done_buf_t[t].copy_(
                    torch.from_numpy(done_np.astype(np.float32)),
                    non_blocking=True)

                next_state = out["state"]
                if done_np.any():
                    done_t = torch.from_numpy(done_np).to(device, non_blocking=True)
                    keep = (~done_t).float().unsqueeze(-1)
                    next_state = (next_state[0] * keep, next_state[1] * keep)

                    # Pull episode outcomes from whichever backend produced them.
                    # outcomes: list[(slot, success, ep_r, ep_l, env_id, level)]
                    pool_outcomes = infos.get("episode_outcomes")
                    if pool_outcomes is not None:
                        # LocalCurriculumVecEnv path — env_id captured per-ep.
                        for slot, success, _ep_r, _ep_l, ep_env_id, ep_level in pool_outcomes:
                            outcomes.append((int(slot), bool(success),
                                             float(ep_rewards[slot]),
                                             int(ep_lengths[slot]),
                                             str(ep_env_id), int(ep_level)))
                            ep_rewards[slot] = 0.0
                            ep_lengths[slot] = 0
                    else:
                        # BatchedAsyncVectorEnv path: pinned to one env_id
                        # for the whole run (the override). Reuse cfg.env_id_override.
                        per_env = _gym_async_per_env_infos(infos, B)
                        ev = cfg.env_id_override or "(pinned)"
                        for i in np.where(done_np)[0]:
                            info_i = per_env[i] if per_env else {}
                            # BatchedAsyncVectorEnv nests terminal info
                            # under "final_info" (gym auto-reset semantics).
                            if "final_info" in info_i and isinstance(info_i["final_info"], dict):
                                info_i = info_i["final_info"]
                            success = bool(info_i.get("episode_success",
                                          info_i.get("is_ascended", False)))
                            outcomes.append((int(i), success,
                                             float(ep_rewards[i]),
                                             int(ep_lengths[i]),
                                             ev, 0))
                            ep_rewards[i] = 0.0
                            ep_lengths[i] = 0

                state = next_state

            # Bootstrap value at obs[T] under the current state.
            ot = obs_to_tensor(obs_np, device)
            if fixed_mask is None:
                mask_b = torch.as_tensor(pool.action_mask(),
                                         dtype=torch.bool, device=device)
            with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model.forward(ot, state, mask_b)  # uncompiled — once per rollout
            bootstrap_value = out["value"].float()

            rollout = Rollout(
                obs=obs_buf_t,
                actions=actions_buf_t,
                behavior_log_probs=lp_buf_t,
                rewards=reward_buf_t,
                dones=done_buf_t,
                values=value_buf_t,
                bootstrap_value=bootstrap_value,
                action_mask=mask_buf_t if fixed_mask is None else mask_b,
                init_state=(init_h, init_c),
            )
            rollout.behavior_features = feat_buf_t

            stats = _learn_specialist(model, optim, rollout, cfg,
                                       rnd=rnd, current_step=step)

            steps_consumed = T * B
            step += steps_consumed
            steps_window += steps_consumed
            update += 1

            for slot, success, ep_r, ep_l, ep_env_id, ep_level in outcomes:
                db.log_episode(sid, step, slot, ep_env_id, ep_r, ep_l, success)
                recent_ep_rewards.append(ep_r)
            if len(recent_ep_rewards) >= 100:
                window_mean = float(np.mean(recent_ep_rewards))
                last_mean_reward = window_mean
                if window_mean > best_mean_reward:
                    best_mean_reward = window_mean
                    _save_specialist(model, optim, rd, sid, step, update,
                                     path_override=rd.spec_best_ckpt(sid))
                    logger.info("new best mean_r=%.1f at step %d, saved best ckpt",
                                best_mean_reward, step)
            # Per-env-id rolling stats so the metrics DB can show
            # SR-per-curriculum-level over time without us having to
            # parse out env_ids in post-hoc analysis.
            if outcomes:
                from collections import defaultdict
                per_env_stats: dict[str, list] = defaultdict(list)
                for _slot, success, ep_r, _ep_l, ep_env_id, _ep_level in outcomes:
                    per_env_stats[ep_env_id].append((bool(success), float(ep_r)))
                env_stats: dict[str, float] = {}
                for ev, rows in per_env_stats.items():
                    n = len(rows)
                    sr = sum(1 for s, _ in rows if s) / n
                    rmean = sum(r for _, r in rows) / n
                    # Sanitise env_id for use as a metric key.
                    safe = ev.replace("MiniHack-", "MH-").replace("/", "_")
                    env_stats[f"per_env/{safe}/sr"] = sr
                    env_stats[f"per_env/{safe}/r_mean"] = rmean
                    env_stats[f"per_env/{safe}/eps"] = float(n)
                if env_stats:
                    db.log_scalars(sid, step, env_stats, "per_env")
                # Rollout-level aggregate: mean / max episode return
                # streamed as scalars (cheap to query without grouping
                # the episodes table). Useful for trend plots.
                ep_rewards_roll = [r for _, _, r, _, _, _ in outcomes]
                ep_lens_roll = [L for _, _, _, L, _, _ in outcomes]
                n_success = sum(1 for _, s, _, _, _, _ in outcomes if s)
                db.log_scalars(sid, step, {
                    "episodes/n_completed": float(len(outcomes)),
                    "episodes/mean_return": float(np.mean(ep_rewards_roll)),
                    "episodes/max_return": float(np.max(ep_rewards_roll)),
                    "episodes/min_return": float(np.min(ep_rewards_roll)),
                    "episodes/mean_length": float(np.mean(ep_lens_roll)),
                    "episodes/success_rate": float(n_success) / len(outcomes),
                }, "episodes")

            if update % cfg.log_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                stats["fps"] = steps_window / elapsed
                stats["num_envs"] = B
                db.log_scalars(sid, step, stats, "specialist")
                t0 = time.time()
                steps_window = 0

            if update % cfg.ckpt_every == 0:
                _save_specialist(model, optim, rd, sid, step, update)
                _save_scheduler_snapshot(pool, rd, sid)

    _save_specialist(model, optim, rd, sid, step, update)
    _save_scheduler_snapshot(pool, rd, sid)
    return {
        "sid": sid, "total_steps": step, "updates": update,
        "mean_reward": last_mean_reward,
        "best_mean_reward": best_mean_reward if best_mean_reward > float('-inf') else 0.0,
    }


def _save_scheduler_snapshot(pool: Any, rd: RunDir, sid: str) -> None:
    """Persist each slot's scheduler state (visited_levels, mastered_levels,
    level_success_window, current_level, current_env_id) to a JSON sidecar
    alongside the agent checkpoint. For pinned-env pools (finetune) the
    scheduler is trivial (frozen to one env); we skip the snapshot there.
    """
    if not hasattr(pool, "scheduler_snapshots"):
        return  # pinned-env BatchedAsync has no per-slot schedulers
    try:
        snaps = pool.scheduler_snapshots()
    except Exception as e:
        logger.warning("scheduler snapshot failed for %s: %r", sid, e)
        return
    import json
    out = rd.spec_sched(sid)
    out.write_text(json.dumps({"sid": sid, "slots": snaps}, indent=2,
                              default=lambda o: list(o) if isinstance(o, (set, tuple)) else repr(o)))


def _gym_async_per_env_infos(infos: Any, num_envs: int) -> list[dict]:
    """Reconstruct a per-env list of info dicts from gym's collated format.

    ``BatchedAsyncVectorEnv`` returns ``{"final_info": [...]}`` — that
    list already has one entry per env (final entries for done envs,
    empty otherwise). Other backends may use the dict-of-arrays mask
    format; we handle both.
    """
    if isinstance(infos, list):
        return list(infos)
    if not isinstance(infos, dict):
        return [{} for _ in range(num_envs)]
    fin = infos.get("final_info")
    if isinstance(fin, list) and len(fin) == num_envs:
        return [(d if isinstance(d, dict) else {}) for d in fin]
    out: list[dict] = [{} for _ in range(num_envs)]
    if fin is not None:
        for i in range(num_envs):
            if i < len(fin) and isinstance(fin[i], dict):
                out[i] = fin[i]
    for k, v in infos.items():
        if k.startswith("_") or k == "final_info":
            continue
        mask = infos.get(f"_{k}")
        if mask is None:
            continue
        try:
            arr = np.asarray(v)
            for i in range(num_envs):
                if i < len(mask) and bool(mask[i]):
                    out[i][k] = arr[i]
        except Exception:
            continue
    return out


# =========================================================================
# Specialist learn helpers
# =========================================================================


def _slice_mask(action_mask: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    """Slice the action mask along its B dim, handling both the pinned
    ``(B, A)`` and curriculum ``(T, B, A)`` shapes."""
    if action_mask.dim() == 2:
        return action_mask[b_idx]
    return action_mask[:, b_idx]


def _learn_specialist(model: Agent, optim, rollout: Rollout,
                      cfg: TrainConfig,
                      rnd: RNDModule | None = None,
                      current_step: int = 0) -> dict:
    """PPO+GAE update step for a single specialist.

    Replaces the V-trace learn step (HANDOFF_PPO.md §4). Once per
    rollout we compute advantages + returns with GAE on the frozen
    behavior values; then ``ppo_epochs`` passes over the rollout split
    into ``ppo_minibatches`` minibatches along the B axis. Each
    minibatch replays the model forward, computes the clipped PPO
    policy + value losses + entropy bonus, optimiser-steps, and logs
    the clip/KL diagnostics.

    ``current_step`` is the env-step counter; used to drive the LR
    warmup/cosine schedule and the entropy-coef anneal.
    """
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    T, B = rollout.actions.shape

    # ---- apply LR schedule + entropy anneal for this update ----
    cur_lr = _current_lr(current_step, cfg)
    for g in optim.param_groups:
        g["lr"] = cur_lr
    cur_ent_coef = _current_entropy_coef(current_step, cfg)

    # -------- once per rollout: GAE on frozen behavior values --------
    rewards = rollout.rewards
    if rnd is not None and cfg.rnd_intrinsic_scale > 0:
        with torch.no_grad():
            feat = rollout.behavior_features.reshape(-1, model.hidden_dim)
            bonus = rnd.intrinsic_reward(feat).reshape(rewards.shape)
        rewards = rewards + cfg.rnd_intrinsic_scale * bonus
    if cfg.reward_clip_tanh_scale > 0:
        rewards = torch.tanh(rewards / cfg.reward_clip_tanh_scale)
    if cfg.reward_clip is not None:
        rewards = torch.clamp(rewards, -cfg.reward_clip, cfg.reward_clip)

    old_log_probs = rollout.behavior_log_probs.detach()
    old_values = rollout.values.detach()
    bootstrap = rollout.bootstrap_value.detach()
    advantages, returns = losses.compute_gae(
        rewards, old_values, bootstrap, rollout.dones,
        gamma=cfg.discount, lam=cfg.gae_lambda,
    )
    # Rollout-level advantage normalisation (Sample-Factory default).
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    params = [p for p in model.parameters() if p.requires_grad]
    if rnd is not None:
        params = params + [p for p in rnd.predictor.parameters()]

    # PPO loop accumulators for logging.
    acc: dict[str, list[float]] = {
        "loss/total": [], "loss/pg": [], "loss/value": [], "loss/entropy": [],
        "loss/rnd": [],
        "ppo/clip_frac": [], "ppo/value_clip_frac": [],
        "ppo/approx_kl": [], "ppo/ratio_mean": [], "ppo/ratio_max": [],
        "grad_norm": [],
    }
    per_epoch_kl: list[float] = []

    for epoch in range(cfg.ppo_epochs):
        perm = torch.randperm(B, device=device)
        mb_B = B // cfg.ppo_minibatches
        epoch_kls: list[float] = []
        for mb in range(cfg.ppo_minibatches):
            b_idx = perm[mb * mb_B:(mb + 1) * mb_B]

            obs_mb = {k: v[:, b_idx] for k, v in rollout.obs.items()}
            init_mb = (rollout.init_state[0][b_idx], rollout.init_state[1][b_idx])
            dones_mb = rollout.dones[:, b_idx]
            mask_mb = _slice_mask(rollout.action_mask, b_idx)
            actions_mb = rollout.actions[:, b_idx]
            old_lp_mb = old_log_probs[:, b_idx]
            old_v_mb = old_values[:, b_idx]
            adv_mb = adv_norm[:, b_idx]
            ret_mb = returns[:, b_idx]

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                out = model.forward_sequence(
                    obs_seq=obs_mb, init_state=init_mb,
                    dones=dones_mb, action_mask=mask_mb,
                )
            logits = out["logits"].float()  # heads already fp32 — cheap no-op
            values_new = out["value"].float()
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

            log_probs_all = F.log_softmax(logits, dim=-1)
            new_lp = log_probs_all.gather(-1, actions_mb.unsqueeze(-1)).squeeze(-1)

            pg_loss, pg_stats = losses.ppo_policy_loss(
                new_lp, old_lp_mb, adv_mb, clip_eps=cfg.ppo_clip)
            v_loss, v_clip_frac = losses.ppo_value_loss(
                values_new, old_v_mb, ret_mb, clip_eps=cfg.value_clip)
            ent_bonus = losses.entropy_loss(logits)

            if rnd is not None and cfg.rnd_coef > 0:
                features = out["features"].float()
                rnd_loss = rnd.distill_loss(features.reshape(-1, model.hidden_dim))
            else:
                rnd_loss = torch.zeros((), device=logits.device)

            total = (pg_loss + cfg.value_coef * v_loss
                     - cur_ent_coef * ent_bonus
                     + cfg.rnd_coef * rnd_loss)

            optim.zero_grad(set_to_none=True)
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            acc["loss/total"].append(float(total.detach()))
            acc["loss/pg"].append(float(pg_loss.detach()))
            acc["loss/value"].append(float(v_loss.detach()))
            acc["loss/entropy"].append(float(ent_bonus.detach()))
            acc["loss/rnd"].append(float(rnd_loss.detach()))
            acc["ppo/clip_frac"].append(pg_stats["clip_frac"])
            acc["ppo/value_clip_frac"].append(v_clip_frac)
            acc["ppo/approx_kl"].append(pg_stats["approx_kl"])
            acc["ppo/ratio_mean"].append(pg_stats["ratio_mean"])
            acc["ppo/ratio_max"].append(pg_stats["ratio_max"])
            acc["grad_norm"].append(float(gn))
            epoch_kls.append(pg_stats["approx_kl"])
        per_epoch_kl.append(float(np.mean(epoch_kls)))

    # ---------- rollout-level diagnostics on behavior buffers ----------
    with torch.no_grad():
        values_d = old_values
        rw = rollout.rewards
        ev = losses.explained_variance(returns, old_values)

    stats: dict[str, float] = {k: float(np.mean(v)) for k, v in acc.items()}
    for i, kl in enumerate(per_epoch_kl):
        stats[f"ppo/approx_kl_epoch_{i}"] = kl
    stats.update({
        "lr/current": cur_lr,
        "entropy_coef/current": cur_ent_coef,
        "value/mean": float(values_d.mean()),
        "value/std": float(values_d.std()),
        "value/min": float(values_d.min()),
        "value/max": float(values_d.max()),
        "value/explained_variance": ev,
        "adv/mean_pre_norm": float(advantages.mean()),
        "adv/std_pre_norm": float(advantages.std()),
        "adv/min": float(advantages.min()),
        "adv/max": float(advantages.max()),
        "returns/mean": float(returns.mean()),
        "returns/std": float(returns.std()),
        "reward/mean": float(rw.mean()),
        "reward/std": float(rw.std()),
        "reward/min": float(rw.min()),
        "reward/max": float(rw.max()),
    })
    return stats


# =========================================================================
# Specialist ckpt I/O
# =========================================================================


def _maybe_resume_specialist(model, optim, rd: RunDir, sid: str
                             ) -> tuple[int, int]:
    ckpt = rd.spec_ckpt(sid)
    if not ckpt.exists():
        return 0, 0
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    if "optim" in blob:
        try:
            optim.load_state_dict(blob["optim"])
        except Exception:
            pass
    return int(blob.get("step", 0)), int(blob.get("update", 0))


def _save_specialist(model, optim, rd: RunDir, sid: str,
                     step: int, update: int,
                     path_override: Path | None = None) -> None:
    blob = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "update": update,
        "saved_at": time.time(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "torch_version": torch.__version__,
    }
    torch.save(blob, path_override or rd.spec_ckpt(sid))


# =========================================================================
# Consensus rollout collection (shared by sync + APPO async paths)
# =========================================================================


def _collect_consensus_rollout(
    cmodel: ConsensusHMoE,           # actor model (may be the learner's cmodel
                                     # in sync mode, or a shadow copy in async)
    forward,                          # compiled forward callable for ``cmodel``
    state: dict,                      # (in) initial rollout state
    obs_np: dict,                     # (in) initial rollout observation
    ep_rewards: np.ndarray,           # (in/out) per-env running ep-return
    ep_lengths: np.ndarray,           # (in/out) per-env running ep-length
    vec,                              # env pool (BatchedAsyncVectorEnv)
    mask_b: torch.Tensor,             # (B, A) action mask
    device: torch.device,
    use_amp: bool,
    amp_dtype,
    T: int,
    B: int,
) -> tuple[Rollout, dict, dict, np.ndarray, np.ndarray,
           list[tuple[int, bool, float, int]], torch.Tensor]:
    """Collect one T-step rollout on ``vec`` using ``cmodel`` as the actor.

    Allocates its own GPU buffers (the caller's Rollout references them, so
    the caller's lifetime controls GC). Updates ``state``, ``obs_np``,
    ``ep_rewards``, ``ep_lengths`` via returned values.

    Returns
    -------
    rollout : Rollout
        Ready for ``_learn_consensus``; attrs ``teacher_logits``,
        ``cons_init_state``, ``spec_logits_cache``, ``spec_features_cache``
        are populated.
    next_state : dict
        Post-rollout LSTM state (core + spec + prev_option).
    next_obs_np : dict
        Post-rollout observation (for the next rollout or bootstrap).
    ep_rewards, ep_lengths : np.ndarray
        Updated running accumulators.
    outcomes : list
        Episode completions during this rollout; same shape as the
        sync path used downstream (slot, success, ep_r, ep_l).
    opt_buf_t : torch.Tensor
        (T, B, K) per-step option logits — kept separately so the
        caller can compute per-option attribution metrics.
    """
    K = cmodel.K
    A = cmodel.num_actions
    spec_D = cmodel.specialists[0].hidden_dim

    ot0 = obs_to_tensor(obs_np, device)
    obs_buf_t: dict[str, torch.Tensor] = {
        k: torch.empty((T, B) + tuple(v.shape[1:]), dtype=v.dtype, device=device)
        for k, v in ot0.items()
    }
    actions_buf_t = torch.empty((T, B), dtype=torch.int64, device=device)
    lp_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    value_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    reward_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    done_buf_t = torch.empty((T, B), dtype=torch.float32, device=device)
    mix_buf_t = torch.empty((T, B, A), dtype=torch.float32, device=device)
    opt_buf_t = torch.empty((T, B, K), dtype=torch.float32, device=device)
    spec_logits_buf_t = torch.empty((T, B, K, A), dtype=torch.float32, device=device)
    spec_feat_buf_t = torch.empty((T, B, K, spec_D), dtype=torch.float32, device=device)
    pinned_action = _make_pinned_action_buf(B)

    init_h = state["core"][0].clone()
    init_c = state["core"][1].clone()
    init_spec = [(s_h.clone(), s_c.clone()) for (s_h, s_c) in state["spec"]]
    init_prev = state["prev_option"].clone()

    outcomes: list[tuple[int, bool, float, int]] = []

    for t in range(T):
        ot = obs_to_tensor(obs_np, device)
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = forward(ot, state, mask_b, deterministic=False)
        logits = out["logits"].float()
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        lps = dist.log_prob(actions)

        for k in OBS_KEYS:
            obs_buf_t[k][t].copy_(ot[k])
        actions_buf_t[t].copy_(actions)
        lp_buf_t[t].copy_(lps)
        value_buf_t[t].copy_(out["value"].float())
        mix_buf_t[t].copy_(out["mixture_logits"].float())
        opt_buf_t[t].copy_(out["option_logits"].float())
        spec_logits_buf_t[t].copy_(out["spec_logits"].float())
        spec_feat_buf_t[t].copy_(out["spec_features"].float())

        pinned_action.copy_(actions, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        actions_np = pinned_action.numpy().astype(np.int64, copy=False)

        obs_np, rewards_np, term_np, trunc_np, infos = vec.step(actions_np)
        done_np = np.asarray(term_np) | np.asarray(trunc_np)

        ep_rewards += np.asarray(rewards_np, dtype=np.float64)
        ep_lengths += 1

        reward_buf_t[t].copy_(
            torch.from_numpy(np.ascontiguousarray(rewards_np, dtype=np.float32)),
            non_blocking=True)
        done_buf_t[t].copy_(torch.from_numpy(done_np.astype(np.float32)),
                            non_blocking=True)

        next_state = out["state"]
        if done_np.any():
            done_t = torch.from_numpy(done_np).to(device, non_blocking=True)
            keep = (~done_t).float().unsqueeze(-1)
            keep_core = keep.unsqueeze(0)
            next_state["core"] = (next_state["core"][0] * keep_core,
                                  next_state["core"][1] * keep_core)
            next_state["spec"] = [(h * keep, c * keep)
                                  for (h, c) in next_state["spec"]]
            next_state["prev_option"] = torch.where(
                done_t, torch.zeros_like(next_state["prev_option"]),
                next_state["prev_option"])

            per_env_infos = _gym_async_per_env_infos(infos, B)
            for i in np.where(done_np)[0]:
                info_i = per_env_infos[i]
                success = bool(info_i.get("episode_success",
                               info_i.get("is_ascended", False)))
                outcomes.append((int(i), success,
                                 float(ep_rewards[i]),
                                 int(ep_lengths[i])))
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0

        state = next_state

    # Bootstrap V at obs[T] under current state — use uncompiled forward
    # for the tail so the compile cache doesn't get a spurious shape call.
    ot = obs_to_tensor(obs_np, device)
    with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        out = cmodel.forward(ot, state, action_mask=mask_b, deterministic=False)
    bootstrap_value = out["value"].float()

    rollout = Rollout(
        obs=obs_buf_t,
        actions=actions_buf_t,
        behavior_log_probs=lp_buf_t,
        rewards=reward_buf_t,
        dones=done_buf_t,
        values=value_buf_t,
        bootstrap_value=bootstrap_value,
        action_mask=mask_b,
        init_state=(init_h, init_c),
    )
    rollout.teacher_logits = mix_buf_t
    rollout.cons_init_state = {
        "core": (init_h, init_c),
        "spec": init_spec,
        "prev_option": init_prev,
    }
    rollout.spec_logits_cache = spec_logits_buf_t
    rollout.spec_features_cache = spec_feat_buf_t

    return rollout, state, obs_np, ep_rewards, ep_lengths, outcomes, opt_buf_t


# =========================================================================
# Consensus trainer (HO-MoE)
# =========================================================================


def train_consensus(cfg: TrainConfig, specialist_paths: dict[str, Path],
                    specialist_masks: dict[str, np.ndarray],
                    env_id: str = "NetHackScore-v0") -> dict:
    """Train the HO-MoE consensus locally on a single env_id.

    K frozen specialists are loaded from ``specialist_paths`` and baked
    into the consensus model. Only the consensus head + option router +
    adapters + lambda head are trainable.

    Same throughput recipe as the specialist trainer: bf16 autocast,
    ``torch.compile`` of the per-step forward, pre-allocated GPU rollout
    buffers, pinned-host action buffer, ``BatchedAsyncVectorEnv`` for the
    env pool.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    rd = RunDir(cfg.run_id, root=Path(cfg.root))
    sids = sorted(specialist_paths.keys())
    specs: list[Agent] = []
    masks_list: list[torch.Tensor] = []
    for sid in sids:
        a = Agent(num_actions=NUM_ACTIONS)
        blob = torch.load(specialist_paths[sid], map_location="cpu", weights_only=False)
        a.load_state_dict(blob["model"])
        a.eval()
        specs.append(a)
        masks_list.append(torch.as_tensor(specialist_masks[sid], dtype=torch.bool))

    cmodel = ConsensusHMoE(specs, num_actions=NUM_ACTIONS,
                           specialist_masks=masks_list).to(device)

    # Resume the consensus head if a prior checkpoint exists. Specialists
    # are reloaded from the canonical paths above so they stay frozen at
    # the right weights regardless of what the snapshot contained.
    # The step counter is restored from the checkpoint so additional
    # training appends cleanly to the metrics DB (e.g. resuming from
    # 200M with cfg.total_steps=300M trains another 100M and logs to
    # step ranges 200M..300M without overwriting the prior history).
    resume_step = 0
    resume_blob: dict | None = None
    resume_path = rd.consensus_ckpt()
    if resume_path.exists():
        resume_blob = torch.load(resume_path, map_location=device,
                                  weights_only=False)
        sd = (resume_blob["model"]
              if isinstance(resume_blob, dict) and "model" in resume_blob
              else resume_blob)
        missing, unexpected = cmodel.load_state_dict(sd, strict=False)
        if isinstance(resume_blob, dict):
            resume_step = int(resume_blob.get("step", 0))
        logger.info(
            "resumed consensus from %s (missing=%d unexpected=%d ckpt_step=%d)",
            resume_path, len(missing), len(unexpected), resume_step,
        )

    forward = _maybe_compile(cmodel, enabled=use_amp)

    params = [p for p in cmodel.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=cfg.learning_rate,
                              eps=cfg.adam_eps)

    # Restore Adam moment estimates (m, v) if the ckpt has them. Otherwise
    # the resume causes a temporary loss spike + slow first ~1000 updates
    # while Adam re-warms. Same for the returns-normalization stats —
    # without these, value/explained_variance dips for ~50M steps until
    # RunningMeanStd re-converges.
    if resume_blob is not None and isinstance(resume_blob, dict):
        if "optimizer" in resume_blob:
            try:
                optim.load_state_dict(resume_blob["optimizer"])
                logger.info("resumed optimizer state from ckpt")
            except Exception as e:
                logger.warning("failed to load optimizer state: %s", e)
        if ("return_rms" in resume_blob
                and getattr(cfg, "normalize_returns", False)):
            if not hasattr(cmodel, "_return_rms"):
                cmodel._return_rms = losses.RunningMeanStd()
            cmodel._return_rms.load_state_dict(resume_blob["return_rms"])
            logger.info(
                "resumed return RMS from ckpt: mean=%.3f var=%.3f count=%.0f",
                cmodel._return_rms.mean, cmodel._return_rms.var,
                cmodel._return_rms._count,
            )

    vec = make_vec_env(env_id, num_envs=cfg.num_envs, seed=cfg.seed,
                       num_workers=cfg.num_env_workers)
    obs_np, _ = vec.reset(seed=[cfg.seed + i for i in range(cfg.num_envs)])

    state = cmodel.initial_state(cfg.num_envs, device)
    mask_np = action_mask_for(env_id)
    mask_b = (torch.as_tensor(mask_np, dtype=torch.bool, device=device)
              .unsqueeze(0).expand(cfg.num_envs, -1).contiguous())

    ep_rewards = np.zeros(cfg.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(cfg.num_envs, dtype=np.int64)

    T = cfg.rollout_len
    B = cfg.num_envs
    K = cmodel.K

    step = resume_step
    update = 0
    t0 = time.time()
    steps_window = 0

    with rd.metrics() as db:
        db.log_event("consensus", step, "start_local", {
            "env_id": env_id, "K": cmodel.K, "device": str(device),
            "num_envs": B, "rollout_len": T,
            "amp": use_amp, "amp_dtype": str(amp_dtype),
            "resume_step": resume_step,
        })

        while step < cfg.total_steps:
            rollout, state, obs_np, ep_rewards, ep_lengths, outcomes, opt_buf_t = \
                _collect_consensus_rollout(
                    cmodel, forward, state, obs_np,
                    ep_rewards, ep_lengths, vec, mask_b,
                    device, use_amp, amp_dtype, T, B,
                )

            stats = _learn_consensus(cmodel, optim, rollout, cfg,
                                      rnd=None, current_step=step)

            # Per-option attribution: which option fired most, what reward did it earn?
            with torch.no_grad():
                opt_argmax = opt_buf_t.argmax(dim=-1)            # (T, B)
                rew_flat = rollout.rewards.reshape(-1)
                arg_flat = opt_argmax.reshape(-1)
                cnt = torch.bincount(arg_flat, minlength=K).float()
                rsum = torch.bincount(arg_flat, weights=rew_flat, minlength=K)
                frac = cnt / max(cnt.sum().item(), 1.0)
                rmean = rsum / cnt.clamp(min=1.0)
                rmean_overall = rew_flat.mean()
                fired = cnt > 0
                if fired.any():
                    best_k = int(rmean.where(fired, torch.full_like(rmean, float("-inf"))).argmax())
                    spread = float(rmean[best_k]) - float(rmean_overall)
                else:
                    best_k = -1
                    spread = 0.0
                stats["option/router_entropy"] = float(
                    losses.entropy_loss(opt_buf_t).detach())
                stats["option/argmax_unique"] = int(fired.sum())
                stats["option/best_idx"] = best_k
                stats["option/best_minus_mean_reward"] = spread
                for k in range(K):
                    stats[f"option/frac_{k:02d}"] = float(frac[k])
                    stats[f"option/reward_{k:02d}"] = float(rmean[k])

                # Per-option softmax distribution (not just argmax). The
                # argmax fraction can look balanced while the underlying
                # softmax is near-uniform — i.e., the router barely
                # commits. argmax_prob_mean tells us the actual
                # commitment level: ~1/K = no preference, →1.0 = peaked.
                opt_probs_buf = F.softmax(opt_buf_t.float(), dim=-1)
                opt_probs_flat = opt_probs_buf.reshape(-1, K)
                mean_probs = opt_probs_flat.mean(dim=0)
                amax_prob = opt_probs_flat.max(dim=-1).values
                stats["option/argmax_prob_mean"] = float(amax_prob.mean())
                stats["option/argmax_prob_p10"] = float(
                    amax_prob.quantile(0.10))
                stats["option/argmax_prob_p99"] = float(
                    amax_prob.quantile(0.99))
                for k in range(K):
                    stats[f"option/mean_prob_{k:02d}"] = float(mean_probs[k])
                    stats[f"option/prob_std_{k:02d}"] = float(
                        opt_probs_flat[:, k].std())

            steps_consumed = T * B
            step += steps_consumed
            steps_window += steps_consumed
            update += 1

            for slot, success, ep_r, ep_l in outcomes:
                db.log_episode("consensus", step, slot, env_id,
                               ep_r, ep_l, success)
            if outcomes:
                ep_rs = [r for _, _, r, _ in outcomes]
                ep_ls = [L for _, _, _, L in outcomes]
                n_succ = sum(1 for _, s, _, _ in outcomes if s)
                db.log_scalars("consensus", step, {
                    "episodes/n_completed": float(len(outcomes)),
                    "episodes/mean_return": float(np.mean(ep_rs)),
                    "episodes/max_return": float(np.max(ep_rs)),
                    "episodes/min_return": float(np.min(ep_rs)),
                    "episodes/mean_length": float(np.mean(ep_ls)),
                    "episodes/success_rate": float(n_succ) / len(outcomes),
                }, "episodes")

            if update % cfg.log_every == 0:
                elapsed = max(time.time() - t0, 1e-6)
                stats["fps"] = steps_window / elapsed
                stats["num_envs"] = B
                db.log_scalars("consensus", step, stats, "consensus")
                t0 = time.time()
                steps_window = 0
            if update % cfg.ckpt_every == 0:
                _save_consensus_ckpt(cmodel, optim, step, update,
                                       rd.consensus_ckpt())

    _save_consensus_ckpt(cmodel, optim, step, update, rd.consensus_ckpt())
    vec.close()
    return {"total_steps": step, "K": cmodel.K}


def _save_consensus_ckpt(cmodel, optim, step, update, path) -> None:
    """Save consensus + optimizer + RunningMeanStd in one blob.

    The optimizer's Adam ``(m, v)`` moments and the return-normalization
    stats are training-loop state that the older ckpts dropped, causing
    a visible value/exp_var dip and a brief retraining transient on
    every resume (see RUN_V4_BIG_2B_REPORT.md notes).
    """
    blob: dict = {
        "model": cmodel.state_dict(),
        "optimizer": optim.state_dict(),
        "step": step,
        "update": update,
        "saved_at": time.time(),
    }
    rms = getattr(cmodel, "_return_rms", None)
    if rms is not None:
        blob["return_rms"] = rms.state_dict()
    torch.save(blob, path)


def _slice_cons_init_state(init_state: dict, b_idx: torch.Tensor) -> dict:
    """Slice the consensus model's init state along its B dim per minibatch.

    - ``core``: tuple of ``(num_layers, B, D)`` tensors → slice dim=1.
    - ``spec``: list of per-specialist ``(h, c)`` with each ``(B, D)``.
    - ``prev_option``: ``(B,)`` long.
    """
    h0, c0 = init_state["core"]
    return {
        "core": (h0[:, b_idx], c0[:, b_idx]),
        "spec": [(h[b_idx], c[b_idx]) for (h, c) in init_state["spec"]],
        "prev_option": init_state["prev_option"][b_idx],
    }


def _learn_consensus(cmodel, optim, rollout, cfg: TrainConfig,
                     rnd: RNDModule | None = None,
                     current_step: int = 0) -> dict:
    """PPO+GAE update step for the HO-MoE consensus.

    Replaces the V-trace learn step. Per rollout: compute GAE
    advantages + returns on the frozen behavior values, then
    ``ppo_epochs`` passes of ``ppo_minibatches`` minibatches along the
    B axis. Each minibatch: replay the consensus forward_sequence
    (specialists still frozen / no_grad inside), compute clipped PPO
    policy + value losses + entropy + router aux losses (kickstart,
    option-entropy, load-balance, router-z), backward, clip-grad, step.

    The router-coef anneal (``option_entropy_coef``, ``load_balance_coef``
    → 1/3× over ``router_coef_anneal_steps``) is computed **once** at
    the start of the update using ``current_step``; the scale is
    constant across the PPO epochs for one update (matching the prior
    V-trace behavior).
    """
    device = next(cmodel.parameters()).device
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    T, B = rollout.actions.shape
    K = cmodel.K

    # ---- apply LR schedule + entropy anneal for this update ----
    cur_lr = _current_lr(current_step, cfg)
    for g in optim.param_groups:
        g["lr"] = cur_lr
    cur_ent_coef = _current_entropy_coef(current_step, cfg)

    # -------- once per rollout: intrinsic reward + GAE --------
    rewards = rollout.rewards
    if rnd is not None and cfg.rnd_intrinsic_scale > 0:
        # Features come from the consensus encoder; for intrinsic reward we
        # want to use whatever the behavior policy saw — the trainer already
        # stores nothing for consensus, so recompute from obs under no_grad.
        # For now: RND is off in consensus (rnd_coef=0 default) so this
        # branch is inert. If enabled, populate rollout.behavior_features
        # in the consensus rollout loop.
        pass
    if cfg.reward_clip_tanh_scale > 0:
        rewards = torch.tanh(rewards / cfg.reward_clip_tanh_scale)
    if cfg.reward_clip is not None:
        rewards = torch.clamp(rewards, -cfg.reward_clip, cfg.reward_clip)

    old_log_probs = rollout.behavior_log_probs.detach()
    old_values = rollout.values.detach()
    bootstrap = rollout.bootstrap_value.detach()

    # Returns normalization (APPO/SF default, Huang 2022 §4). The value
    # head is trained to predict NORMALIZED returns — old_values is
    # therefore in normalized scale. Denormalize V for GAE math (which
    # needs V in the same scale as r), compute GAE in raw scale, then
    # re-normalize the target returns for the value loss. old_values
    # stays in normalized scale (matches the value head output).
    if getattr(cfg, "normalize_returns", False):
        if not hasattr(cmodel, "_return_rms"):
            cmodel._return_rms = losses.RunningMeanStd()
        rms = cmodel._return_rms
        std = rms.std
        old_values_raw = old_values * std
        bootstrap_raw = bootstrap * std
        advantages, returns_raw = losses.compute_gae(
            rewards, old_values_raw, bootstrap_raw, rollout.dones,
            gamma=cfg.discount, lam=cfg.gae_lambda,
        )
        rms.update(returns_raw)
        returns = returns_raw / rms.std
    else:
        advantages, returns = losses.compute_gae(
            rewards, old_values, bootstrap, rollout.dones,
            gamma=cfg.discount, lam=cfg.gae_lambda,
        )
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Router-coef anneal: 1.0× → 1/3× over the first router_coef_anneal_steps.
    if cfg.router_coef_anneal_steps > 0:
        anneal_t = min(1.0, current_step / cfg.router_coef_anneal_steps)
        scale = 1.0 - (2.0 / 3.0) * anneal_t
    else:
        scale = 1.0
    oe_coef = cfg.option_entropy_coef * scale
    lb_coef = cfg.load_balance_coef * scale

    params = [p for p in cmodel.parameters() if p.requires_grad]
    if rnd is not None:
        params = params + [p for p in rnd.predictor.parameters()]

    acc: dict[str, list[float]] = {
        "loss/total": [], "loss/pg": [], "loss/value": [], "loss/entropy": [],
        "loss/kickstart": [], "loss/option_entropy": [],
        "loss/load_balance": [], "loss/router_z": [], "loss/rnd": [],
        "loss/option_stickiness": [],
        "ppo/clip_frac": [], "ppo/value_clip_frac": [],
        "ppo/approx_kl": [], "ppo/ratio_mean": [], "ppo/ratio_max": [],
        "grad_norm": [],
        "mixing_lambda": [], "mixing_lambda/min": [], "mixing_lambda/max": [],
        "lambda/frac_lt02": [], "lambda/frac_02_05": [],
        "lambda/frac_05_08": [], "lambda/frac_gt08": [],
    }
    per_epoch_kl: list[float] = []

    for epoch in range(cfg.ppo_epochs):
        perm = torch.randperm(B, device=device)
        mb_B = B // cfg.ppo_minibatches
        epoch_kls: list[float] = []
        for mb in range(cfg.ppo_minibatches):
            b_idx = perm[mb * mb_B:(mb + 1) * mb_B]

            obs_mb = {k: v[:, b_idx] for k, v in rollout.obs.items()}
            init_mb = _slice_cons_init_state(rollout.cons_init_state, b_idx)
            dones_mb = rollout.dones[:, b_idx]
            mask_mb = _slice_mask(rollout.action_mask, b_idx)
            actions_mb = rollout.actions[:, b_idx]
            old_lp_mb = old_log_probs[:, b_idx]
            old_v_mb = old_values[:, b_idx]
            adv_mb = adv_norm[:, b_idx]
            ret_mb = returns[:, b_idx]

            cached_spec = None
            if getattr(rollout, "spec_logits_cache", None) is not None:
                cached_spec = {
                    "logits": rollout.spec_logits_cache[:, b_idx],
                    "features": rollout.spec_features_cache[:, b_idx],
                }

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                out = cmodel.forward_sequence(
                    obs_seq=obs_mb, init_state=init_mb,
                    dones=dones_mb, action_mask=mask_mb,
                    cached_spec=cached_spec,
                )
            logits = out["logits"].float()
            values_new = out["value"].float()
            opt_logits = out["option_logits"].float()
            lam = out["lambda"].float()
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)

            log_probs_all = F.log_softmax(logits, dim=-1)
            new_lp = log_probs_all.gather(-1, actions_mb.unsqueeze(-1)).squeeze(-1)

            pg_loss, pg_stats = losses.ppo_policy_loss(
                new_lp, old_lp_mb, adv_mb, clip_eps=cfg.ppo_clip)
            v_loss, v_clip_frac = losses.ppo_value_loss(
                values_new, old_v_mb, ret_mb, clip_eps=cfg.value_clip)
            ent_bonus = losses.entropy_loss(logits)

            # Router aux losses on this minibatch's replay outputs.
            oh = losses.option_entropy_bonus(opt_logits)
            opt_probs_mb = F.softmax(opt_logits, dim=-1)
            opt_assign_mb = opt_logits.argmax(dim=-1)
            lb = losses.load_balance_loss(opt_probs_mb, opt_assign_mb, K)
            rz = losses.router_z_loss(opt_logits)
            if cfg.option_stickiness_coef > 0 and "prev_option" in out:
                stick = losses.option_stickiness_loss(
                    opt_logits, out["prev_option"], dones_mb)
            else:
                stick = torch.zeros((), device=logits.device)

            if cfg.kickstart_coef > 0 and hasattr(rollout, "teacher_logits") \
               and rollout.teacher_logits is not None:
                teacher_mb = rollout.teacher_logits[:, b_idx]
                ks = losses.kickstart_loss(logits, teacher_mb, mask_mb)
            else:
                ks = torch.zeros((), device=logits.device)

            if rnd is not None and cfg.rnd_coef > 0:
                features = out["features"].float()
                rnd_loss = rnd.distill_loss(
                    features.reshape(-1, features.shape[-1]))
            else:
                rnd_loss = torch.zeros((), device=logits.device)

            total = (pg_loss + cfg.value_coef * v_loss
                     - cur_ent_coef * ent_bonus
                     + cfg.kickstart_coef * ks - oe_coef * oh
                     + lb_coef * lb + cfg.router_z_coef * rz
                     + cfg.option_stickiness_coef * stick
                     + cfg.rnd_coef * rnd_loss)

            optim.zero_grad(set_to_none=True)
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()

            with torch.no_grad():
                lam_d = lam.detach()
                lam_flat = lam_d.reshape(-1)
                frac_lt02 = float((lam_flat < 0.2).float().mean())
                frac_02_05 = float(((lam_flat >= 0.2) & (lam_flat < 0.5))
                                   .float().mean())
                frac_05_08 = float(((lam_flat >= 0.5) & (lam_flat < 0.8))
                                   .float().mean())
                frac_gt08 = float((lam_flat >= 0.8).float().mean())

            acc["loss/total"].append(float(total.detach()))
            acc["loss/pg"].append(float(pg_loss.detach()))
            acc["loss/value"].append(float(v_loss.detach()))
            acc["loss/entropy"].append(float(ent_bonus.detach()))
            acc["loss/kickstart"].append(float(ks.detach()))
            acc["loss/option_entropy"].append(float(oh.detach()))
            acc["loss/load_balance"].append(float(lb.detach()))
            acc["loss/router_z"].append(float(rz.detach()))
            acc["loss/option_stickiness"].append(float(stick.detach()))
            acc["loss/rnd"].append(float(rnd_loss.detach()))
            acc["ppo/clip_frac"].append(pg_stats["clip_frac"])
            acc["ppo/value_clip_frac"].append(v_clip_frac)
            acc["ppo/approx_kl"].append(pg_stats["approx_kl"])
            acc["ppo/ratio_mean"].append(pg_stats["ratio_mean"])
            acc["ppo/ratio_max"].append(pg_stats["ratio_max"])
            acc["grad_norm"].append(float(gn))
            acc["mixing_lambda"].append(float(lam_d.mean()))
            acc["mixing_lambda/min"].append(float(lam_d.min()))
            acc["mixing_lambda/max"].append(float(lam_d.max()))
            acc["lambda/frac_lt02"].append(frac_lt02)
            acc["lambda/frac_02_05"].append(frac_02_05)
            acc["lambda/frac_05_08"].append(frac_05_08)
            acc["lambda/frac_gt08"].append(frac_gt08)
            epoch_kls.append(pg_stats["approx_kl"])
        per_epoch_kl.append(float(np.mean(epoch_kls)))

    # --------- rollout-level diagnostics ---------
    with torch.no_grad():
        ev = losses.explained_variance(returns, old_values)

    stats: dict[str, float] = {k: float(np.mean(v)) for k, v in acc.items()}
    for i, kl in enumerate(per_epoch_kl):
        stats[f"ppo/approx_kl_epoch_{i}"] = kl
    stats.update({
        "router_coef_scale": float(scale),
        "lr/current": cur_lr,
        "entropy_coef/current": cur_ent_coef,
        "value/mean": float(old_values.mean()),
        "value/std": float(old_values.std()),
        "value/min": float(old_values.min()),
        "value/max": float(old_values.max()),
        "value/explained_variance": ev,
        "adv/mean_pre_norm": float(advantages.mean()),
        "adv/std_pre_norm": float(advantages.std()),
        "adv/min": float(advantages.min()),
        "adv/max": float(advantages.max()),
        "returns/mean": float(returns.mean()),
        "returns/std": float(returns.std()),
        "reward/extrinsic_mean": float(rollout.rewards.mean()),
        "reward/extrinsic_std": float(rollout.rewards.std()),
        "reward/extrinsic_max": float(rollout.rewards.max()),
        "reward/extrinsic_min": float(rollout.rewards.min()),
    })
    return stats
