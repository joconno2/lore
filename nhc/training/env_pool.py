"""Local subprocess-based env pools for the in-process trainer.

Two backends:

- :class:`LocalCurriculumVecEnv` — one subprocess per env slot. Each slot
  owns its own :class:`nhc.curriculum.CurriculumScheduler` so the inner
  rollout already mixes the frontier level with previously-solved review
  levels. Used for **specialist pretraining** on MiniHack curricula.
  Ships a per-step ``(B, A)`` action mask because slots can be on
  different env_ids.

- :class:`nhc.env.BatchedAsyncVectorEnv` — W subprocesses, each owning
  ``num_envs / W`` envs of a *single* env_id. Used for **finetune** and
  **consensus** training where every slot steps the same env. Cheaper
  RAM (one NLE process holds many envs) and lower per-step latency.
  Re-exported here as :class:`LocalPinnedVecEnv` so the trainer picks a
  backend by name.

Both backends expose:

  ``num_envs``                              int
  ``reset(seed=...)``                       (obs_dict, info)
  ``step(actions: np.ndarray)``             (obs, rewards, terms, truncs, info)
  ``step_send(actions) / step_recv()``      split for the concurrent trainer
  ``close()``

The split send/recv is plumbing for any caller that wants to overlap
multiple pools (e.g. a future multi-spec trainer). The current
single-spec trainer just calls ``step()``.

Curriculum-only extras on :class:`LocalCurriculumVecEnv`:

  ``action_mask()``    -> ``(B, A) bool`` per current slot env_id
  ``slot_levels()``    -> ``list[int]`` per slot frontier index

Both are zero-cost cached calls; the masks come from the slot worker
on every reply, the parent just stacks the cached arrays.
"""
from __future__ import annotations

import multiprocessing as _mp
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

from nhc.curriculum import CurriculumScheduler
from nhc.env import (
    BatchedAsyncVectorEnv,  # re-exported below as LocalPinnedVecEnv
    NUM_ACTIONS,
    OBS_KEYS,
    action_mask_for,
    make_env,
)


# Convenience alias so the trainer can refer to the pinned-env backend by
# the same naming convention as the curriculum backend.
LocalPinnedVecEnv = BatchedAsyncVectorEnv


# ---------------------------------------------------------------- LRU env cache


class _EnvLRU:
    """Bounded LRU of warm env instances, keyed by ``env_id``.

    Closing an NLE env releases its C-side game state; reopening costs
    ~100 ms. With per-slot review sampling we'd otherwise reopen on every
    other reset. A cap of 3 keeps memory bounded but covers the typical
    {frontier, frontier-1, frontier-2} review window.
    """

    def __init__(self, max_size: int = 3):
        self.max_size = int(max_size)
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, env_id: str):
        env = self._cache.pop(env_id, None)
        if env is None:
            env = make_env(env_id)
        self._cache[env_id] = env
        while len(self._cache) > self.max_size:
            _, victim = self._cache.popitem(last=False)
            try:
                victim.close()
            except Exception:
                pass
        return env

    def close_all(self) -> None:
        while self._cache:
            _, env = self._cache.popitem()
            try:
                env.close()
            except Exception:
                pass


# ------------------------------------------------------------- per-slot state


class _Slot:
    """One env stream inside a pool: scheduler + current env + tracker.

    Owns the curriculum scheduler, a small warm-env LRU keyed by env_id,
    and the running episode reward/length so the parent can emit episode
    outcomes when the env terminates.
    """

    def __init__(self, sid: str, slot_idx: int, p_review: float, seed: int,
                 env_id_override: Optional[str] = None):
        self.sid = sid
        self.slot_idx = slot_idx
        self.scheduler = CurriculumScheduler(sid, p_review=p_review)
        if env_id_override is not None:
            # Pinned mode: every reset uses one env, no review/advance.
            self.scheduler.env_ids = [env_id_override]
            self.scheduler.p_review = 0.0
            self.scheduler.state.level_idx = 0
            self.scheduler.state.visited_levels = []
            self.scheduler.state.mastered_levels = []
        self.rng = np.random.default_rng(seed)
        self.envs = _EnvLRU(max_size=3)
        self._mask_cache: dict[str, np.ndarray] = {}
        self.current_level: int = self.scheduler.state.level_idx
        self.current_env_id: str = self.scheduler.env_ids[self.current_level]
        self.env = self.envs.get(self.current_env_id)
        self.obs, _ = self.env.reset(seed=seed)
        self.ep_reward = 0.0
        self.ep_len = 0
        self.last_episode_outcome: Optional[bool] = None
        self.last_episode_reward: Optional[float] = None
        self.last_episode_length: Optional[int] = None
        # Env id and curriculum level of the *just-completed* episode.
        # Captured before the post-reset re-roll (which may pick a new
        # frontier or review env), so the trainer can log SR against
        # the env that was actually being attempted.
        self.last_episode_env_id: Optional[str] = None
        self.last_episode_level: Optional[int] = None

    def mask_for(self, env_id: str) -> np.ndarray:
        m = self._mask_cache.get(env_id)
        if m is None:
            m = action_mask_for(env_id)
            self._mask_cache[env_id] = m
        return m

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """Step the current env. On termination, record the outcome,
        possibly advance/regress the scheduler, draw a new env_id (review
        or frontier), and reset.
        """
        obs, r, term, trunc, info = self.env.step(int(action))
        done = bool(term or trunc)
        self.ep_reward += float(r)
        self.ep_len += 1
        if done:
            success = bool(info.get("episode_success", False))
            self.last_episode_outcome = success
            self.last_episode_reward = float(info.get("episode_reward",
                                                      self.ep_reward))
            self.last_episode_length = int(info.get("episode_length",
                                                    self.ep_len))
            # Capture the env_id / level the agent was *attempting*,
            # before the scheduler reroll changes them for the next ep.
            self.last_episode_env_id = self.current_env_id
            self.last_episode_level = self.current_level
            self.scheduler.record(success, level_idx=self.current_level)
            new_idx, new_env_id = self.scheduler.sample_env_id(self.rng)
            if new_env_id != self.current_env_id:
                self.env = self.envs.get(new_env_id)
                self.current_env_id = new_env_id
            self.current_level = new_idx
            self.ep_reward = 0.0
            self.ep_len = 0
            obs, _ = self.env.reset()
        self.obs = obs
        return obs, float(r), done, info

    def close(self) -> None:
        self.envs.close_all()


# --------------------------------------------------------------- subproc body


def _slot_worker(slot_idx: int, sid: str, p_review: float, seed: int,
                 conn, env_id_override: Optional[str] = None) -> None:
    """Subprocess body: own one ``_Slot`` and respond to step/close commands.

    Parent → child: ``("step", action_int)`` or ``("close", None)``.

    Initial reply (immediately after construction): the *current* obs / env_id /
    level / mask so the parent can populate its first batch.

    Per-step reply: ``{obs, reward, done, episode_outcome, episode_reward,
    episode_length, env_id, level, mask}``.
    """
    try:
        slot = _Slot(sid, slot_idx, p_review=p_review, seed=seed,
                     env_id_override=env_id_override)
        conn.send({
            "obs": slot.obs,
            "env_id": slot.current_env_id,
            "level": slot.current_level,
            "mask": slot.mask_for(slot.current_env_id),
        })
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                slot.close()
                conn.close()
                return
            if cmd == "snapshot":
                # Return scheduler state so the parent can persist per-slot
                # curriculum progress (visited_levels, mastered_levels,
                # per-level success windows) alongside the agent checkpoint.
                conn.send({"scheduler": slot.scheduler.state_dict(),
                           "current_level": slot.current_level,
                           "current_env_id": slot.current_env_id})
                continue
            if cmd != "step":
                continue
            obs, r, done, _info = slot.step(int(payload))
            outcome = slot.last_episode_outcome
            ep_r = slot.last_episode_reward
            ep_l = slot.last_episode_length
            ep_env_id = slot.last_episode_env_id
            ep_level = slot.last_episode_level
            slot.last_episode_outcome = None
            slot.last_episode_reward = None
            slot.last_episode_length = None
            slot.last_episode_env_id = None
            slot.last_episode_level = None
            conn.send({
                "obs": obs,
                "reward": float(r),
                "done": bool(done),
                "episode_outcome": outcome,
                "episode_reward": ep_r,
                "episode_length": ep_l,
                "episode_env_id": ep_env_id,
                "episode_level": ep_level,
                "env_id": slot.current_env_id,
                "level": slot.current_level,
                "mask": slot.mask_for(slot.current_env_id),
            })
    except Exception as e:  # pragma: no cover — surface to parent
        try:
            conn.send({"error": repr(e)})
        finally:
            conn.close()


# --------------------------------------------------------------- slot pool


class _SlotPool:
    """Subprocess-per-slot env pool.

    Owns N child processes, each running :func:`_slot_worker` against one
    real env. The parent fans out one action per slot then drains; the
    inner per-step latency is ``max_i(env_i.step)`` instead of
    ``sum_i(env_i.step)``. NLE's ``env.step`` is a few-ms C call so the
    parallel speedup is roughly linear up to physical cores.

    The send/recv split (``step_send`` then ``step_recv``) is exposed so
    the concurrent multi-spec trainer can interleave sends across multiple
    pools, letting their subprocess env-step work overlap.
    """

    def __init__(self, sid: str, num_envs: int, p_review: float, seed: int,
                 env_id_override: Optional[str] = None):
        ctx = _mp.get_context("spawn")
        self.num_envs = int(num_envs)
        self.sid = sid
        self._procs = []
        self._conns = []
        # Last-seen state per slot so the parent can stack obs / masks
        # without round-tripping to the children.
        self._last_obs: list[dict] = [None] * self.num_envs  # type: ignore[list-item]
        self._last_env_ids: list[str] = [""] * self.num_envs
        self._last_levels: list[int] = [0] * self.num_envs
        self._last_masks: list[np.ndarray] = [None] * self.num_envs  # type: ignore[list-item]
        for i in range(self.num_envs):
            parent, child = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_slot_worker,
                args=(i, sid, float(p_review), int(seed) + i * 7919, child,
                      env_id_override),
                daemon=True,
            )
            p.start()
            child.close()
            self._procs.append(p)
            self._conns.append(parent)
        # Drain initial state from each child.
        for i, conn in enumerate(self._conns):
            msg = conn.recv()
            if "error" in msg:
                raise RuntimeError(f"slot {i} init failed: {msg['error']}")
            self._last_obs[i] = msg["obs"]
            self._last_env_ids[i] = msg["env_id"]
            self._last_levels[i] = msg["level"]
            self._last_masks[i] = msg["mask"]
        self._send_pending = False

    def stack_obs(self) -> dict[str, np.ndarray]:
        return {
            "glyphs": np.stack([o["glyphs"] for o in self._last_obs], axis=0),
            "blstats": np.stack([o["blstats"] for o in self._last_obs], axis=0),
            "message": np.stack([o["message"] for o in self._last_obs], axis=0),
        }

    def stack_masks(self) -> np.ndarray:
        return np.stack(self._last_masks, axis=0)

    def slot_levels(self) -> list[int]:
        return list(self._last_levels)

    def step_send(self, actions: np.ndarray) -> None:
        """Fan out one action per slot. Returns immediately; pair with
        ``step_recv``. Used by the concurrent trainer to interleave sends
        across multiple pools."""
        for i, conn in enumerate(self._conns):
            conn.send(("step", int(actions[i])))
        self._send_pending = True

    def step_recv(self) -> dict:
        """Drain replies for the most recent ``step_send``.

        Returns a dict with keys:
          ``rewards`` (B,) float32
          ``dones``   (B,) float32
          ``episode_outcomes`` list[(slot, success, ep_r, ep_l, env_id, level)]
            — env_id/level are the env the slot was *attempting* when
            the episode terminated, captured before the scheduler reroll.
        """
        if not self._send_pending:
            raise RuntimeError("step_recv called without a pending step_send")
        B = self.num_envs
        rewards = np.zeros(B, dtype=np.float32)
        dones = np.zeros(B, dtype=np.float32)
        outcomes: list[tuple[int, bool, float, int, str, int]] = []
        for i, conn in enumerate(self._conns):
            msg = conn.recv()
            if "error" in msg:
                raise RuntimeError(f"slot {i} step failed: {msg['error']}")
            self._last_obs[i] = msg["obs"]
            self._last_env_ids[i] = msg["env_id"]
            self._last_levels[i] = msg["level"]
            self._last_masks[i] = msg["mask"]
            rewards[i] = msg["reward"]
            dones[i] = float(msg["done"])
            if msg["done"] and msg["episode_outcome"] is not None:
                outcomes.append((
                    i,
                    bool(msg["episode_outcome"]),
                    float(msg.get("episode_reward") or 0.0),
                    int(msg.get("episode_length") or 0),
                    str(msg.get("episode_env_id") or ""),
                    int(msg.get("episode_level") or 0),
                ))
        self._send_pending = False
        return {"rewards": rewards, "dones": dones,
                "episode_outcomes": outcomes}

    def step_all(self, actions: np.ndarray) -> dict:
        """Synchronous send-then-drain. Equivalent to ``step_send`` then
        ``step_recv``. Kept for callers that don't need the split."""
        self.step_send(actions)
        return self.step_recv()

    def scheduler_snapshots(self) -> list[dict]:
        """Ask every slot for its scheduler state. Used to persist per-slot
        curriculum progress alongside the agent checkpoint so a post-hoc
        analyser can answer "which levels had this specialist mastered?"
        without re-deriving from the episodes table."""
        if self._send_pending:
            raise RuntimeError("cannot snapshot while a step is in flight")
        for conn in self._conns:
            conn.send(("snapshot", None))
        out = []
        for i, conn in enumerate(self._conns):
            try:
                msg = conn.recv()
                out.append({"slot": i, **msg})
            except Exception as e:
                out.append({"slot": i, "error": repr(e)})
        return out

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass


# --------------------------------------------------------------- public API


class LocalCurriculumVecEnv:
    """Gym-vector-style wrapper around :class:`_SlotPool` for curriculum training.

    Per-slot scheduler + warm-env LRU + per-step action mask. ``step``
    returns gym-style ``(obs, rewards, terms, truncs, info)`` where
    ``info["episode_outcomes"]`` carries the ``(slot, success, ep_reward,
    ep_length)`` tuples the trainer logs.

    Single-spec trainers call :meth:`step`; the concurrent trainer calls
    :meth:`step_send` / :meth:`step_recv` to overlap multiple pools.

    Action mask is per-slot per-step (slots can be on different curriculum
    levels with different valid action subsets). Use :meth:`action_mask`
    to fetch the current ``(B, A)`` mask before sampling actions; the
    next step's mask is updated by the slot worker after the step.
    """

    def __init__(self, sid: str, *, num_envs: int = 32, p_review: float = 0.25,
                 seed: int = 0, env_id_override: Optional[str] = None):
        self.sid = sid
        self.num_envs = int(num_envs)
        self.pool = _SlotPool(sid, num_envs=num_envs, p_review=p_review,
                              seed=seed, env_id_override=env_id_override)

    def reset(self, *, seed=None) -> tuple[dict[str, np.ndarray], dict]:
        # The pool already drained initial obs at construction. ``seed`` is
        # accepted for API parity but has no effect after init (each slot's
        # rng is seeded in `_Slot.__init__`).
        del seed
        return self.pool.stack_obs(), {}

    def step_send(self, actions: np.ndarray) -> None:
        self.pool.step_send(actions)

    def step_recv(self) -> tuple[dict[str, np.ndarray], np.ndarray,
                                  np.ndarray, np.ndarray, dict]:
        result = self.pool.step_recv()
        obs = self.pool.stack_obs()
        terms = result["dones"].astype(bool)
        # Slot worker collapses term and trunc into a single ``done`` flag.
        # Surface as terms-only; truncs is always-False so V-trace's
        # done handling (gamma * (1 - done)) works the same way.
        truncs = np.zeros_like(terms, dtype=bool)
        info = {"episode_outcomes": result["episode_outcomes"]}
        return obs, result["rewards"], terms, truncs, info

    def step(self, actions: np.ndarray):
        self.step_send(actions)
        return self.step_recv()

    def action_mask(self) -> np.ndarray:
        """Current per-slot action mask, shape ``(B, A)`` bool."""
        return self.pool.stack_masks()

    def slot_levels(self) -> list[int]:
        return self.pool.slot_levels()

    def scheduler_snapshots(self) -> list[dict]:
        return self.pool.scheduler_snapshots()

    def close(self) -> None:
        self.pool.close()


__all__ = ["LocalCurriculumVecEnv", "LocalPinnedVecEnv", "BatchedAsyncVectorEnv"]
