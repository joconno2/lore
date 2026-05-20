"""Real NLE / MiniHack env factory and action-space plumbing.

Single source of truth for environment construction. No mock env exists —
callers must pass a real gymnasium id. The agent's policy head is indexed
by the canonical NLE action order (`nle.nethack.ACTIONS`); per-env masks
translate to whatever subset that env actually exposes.
"""
from __future__ import annotations

import warnings

import numpy as np
import gymnasium as gym
import torch

from nle import nethack as _nethack

CANONICAL_ACTIONS: tuple = tuple(_nethack.ACTIONS)
NUM_ACTIONS = len(CANONICAL_ACTIONS)
OBS_KEYS = ("glyphs", "blstats", "message")


def _register_envs(env_id: str) -> None:
    """Trigger env-registration side effects. Subprocess workers need this."""
    if env_id.startswith("MiniHack"):
        import minihack  # noqa: F401
        _patch_minigrid_once()
        _register_custom_minihack()
    if env_id.startswith("NetHack"):
        import nle  # noqa: F401


_CUSTOM_REGISTERED = False
_DES_DIR = (__file__.rsplit("/", 1)[0] + "/des") if "/" in __file__ else "des"

_CUSTOM_ENVS = {
    "MiniHack-Inventory-Simple-v0": ("inventory_manager.des", 500),
    "MiniHack-Retreat-SlowMonster-v0": ("tactical_retreat.des", 500),
    "MiniHack-Prayer-Aligned-v0": ("prayer_altar.des", 300),
}


def _register_custom_minihack() -> None:
    """Idempotent registration of the C1/C2/C3 .des-defined envs."""
    global _CUSTOM_REGISTERED
    if _CUSTOM_REGISTERED:
        return
    import os
    existing = set(gym.envs.registry.keys())
    for env_id, (fname, max_steps) in _CUSTOM_ENVS.items():
        if env_id in existing:
            continue
        des_path = os.path.join(_DES_DIR, fname)
        if not os.path.exists(des_path):
            continue
        try:
            gym.register(
                id=env_id,
                entry_point="minihack.navigation:MiniHackNavigation",
                kwargs={
                    "des_file": des_path,
                    "max_episode_steps": max_steps,
                    "observation_keys": OBS_KEYS,
                },
            )
        except Exception:
            pass
    _CUSTOM_REGISTERED = True


_MINIGRID_PATCHED = False


def _patch_minigrid_once() -> None:
    """MiniHack 1.0.2 + minigrid 2.x: MultiRoomEnv hard-codes width/height,
    collides with MiniHack's kwargs. Patch to honor them."""
    global _MINIGRID_PATCHED
    if _MINIGRID_PATCHED:
        return
    try:
        import minigrid
        reg = getattr(minigrid, "register_minigrid_envs", None)
        if reg is not None:
            try:
                reg()
            except Exception:
                pass
        from minigrid.envs.multiroom import MultiRoomEnv
        _orig = MultiRoomEnv.__init__

        def _patched(self, minNumRooms, maxNumRooms, maxRoomSize=10, max_steps=None, **kw):
            w = kw.pop("width", None)
            h = kw.pop("height", None)
            _orig(self, minNumRooms, maxNumRooms, maxRoomSize=maxRoomSize,
                  max_steps=max_steps, **kw)
            if w is not None and h is not None:
                self.size = min(int(w), int(h))
                self.width = int(w); self.height = int(h)

        MultiRoomEnv.__init__ = _patched
    except ModuleNotFoundError:
        pass
    _MINIGRID_PATCHED = True


def _env_action_mask(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    """Return (canonical_mask, canonical_to_tuple_idx)."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    canon_to_tuple = -np.ones(NUM_ACTIONS, dtype=np.int64)
    env_actions = getattr(env.unwrapped, "actions", None)
    if env_actions is None:
        mask[:] = True
        canon_to_tuple[:] = np.arange(NUM_ACTIONS)
        return mask, canon_to_tuple
    tuple_by_val = {
        (a.value if hasattr(a, "value") else int(a)): i
        for i, a in enumerate(env_actions)
    }
    for i, canon in enumerate(CANONICAL_ACTIONS):
        v = canon.value if hasattr(canon, "value") else int(canon)
        if v in tuple_by_val:
            mask[i] = True
            canon_to_tuple[i] = tuple_by_val[v]
    return mask, canon_to_tuple


class _Wrap(gym.Wrapper):
    """Combined wrapper: normalize obs dtype, translate canonical action
    index to env tuple index, detect success on done."""

    def __init__(self, env: gym.Env, success_threshold: float = 0.5):
        super().__init__(env)
        self._mask, self._canon_to_tuple = _env_action_mask(env)
        self._valid = np.where(self._mask)[0]
        self._success_threshold = success_threshold
        self._ep_reward = 0.0
        self._ep_len = 0

    @property
    def action_mask(self) -> np.ndarray:
        return self._mask.copy()

    def reset(self, **kw):
        self._ep_reward = 0.0
        self._ep_len = 0
        obs, info = self.env.reset(**kw)
        return self._norm(obs), info

    def step(self, action):
        a = int(action)
        if 0 <= a < NUM_ACTIONS and self._canon_to_tuple[a] >= 0:
            tidx = int(self._canon_to_tuple[a])
        elif self._valid.size:
            tidx = int(self._canon_to_tuple[int(self._valid[0])])
        else:
            tidx = 0
        obs, r, term, trunc, info = self.env.step(tidx)
        self._ep_reward += float(r)
        self._ep_len += 1
        if term or trunc:
            # Principled, env-defined success signal. No reward heuristics.
            #
            # MiniHack envs (the entire pretraining curriculum) populate
            # ``info["end_status"]`` from a ``StepStatus`` enum. The
            # ``TASK_SUCCESSFUL`` value is fired only when the env's
            # ``RewardManager`` confirms the task-success event — for
            # navigation envs that means the agent stepped onto the goal
            # tile, etc. Anything else (DEATH, TASK_PENDING from timeout,
            # ABORT) is a failure.
            #
            # NetHack envs (NetHackScore, Challenge) populate
            # ``info["is_ascended"]``. Ascension is essentially never True
            # at our scale, but it's the only env-defined success signal
            # — a 0/1 marker, not a reward heuristic.
            #
            # If both are absent (truly unknown env): success=False. We
            # never invent success from reward magnitudes. The curriculum
            # scheduler relies on this signal being honest; a borderline
            # specialist that can't cross tau=0.7 should be advanced by
            # the scheduler's bounded-time rule (max_episodes), not by a
            # reward-threshold hack here.
            es = info.get("end_status", None)
            es_name = getattr(es, "name", None)
            if es_name is not None:
                success = (es_name == "TASK_SUCCESSFUL")
            elif "is_ascended" in info:
                success = bool(info["is_ascended"])
            else:
                success = False
            info["episode_success"] = success
            info["episode_reward"] = self._ep_reward
            info["episode_length"] = self._ep_len
        return self._norm(obs), r, term, trunc, info

    @staticmethod
    def _norm(obs: dict) -> dict:
        glyphs = np.asarray(obs["glyphs"], dtype=np.int16)
        blstats = np.asarray(obs["blstats"], dtype=np.float32)
        if blstats.shape[-1] < 27:
            blstats = np.pad(blstats, (0, 27 - blstats.shape[-1]))
        msg = np.asarray(obs.get("message", np.zeros(256)), dtype=np.uint8)
        return {"glyphs": glyphs, "blstats": blstats, "message": msg}


def make_env(env_id: str, max_episode_steps: int | None = None) -> gym.Env:
    _register_envs(env_id)
    # For NLE tasks (NetHackScore, NetHackChallenge, etc.) pin the agent
    # character to Human Valkyrie Lawful Female. NLE's default
    # "mon-hum-neu-mal" is a Human Monk: no starting weapon, low HP, and
    # the intrinsic hunger-survival playstyle is a notoriously hard
    # starting class for RL. Valkyrie starts with a +3 long sword, +3
    # small shield, 16 HP, cold resistance — the canonical "easy" class
    # that also benefits directly from our melee specialist. The specific
    # "val-hum-law-fem" combo gives Human (slightly less raw strength
    # than Dwarf but the standard RL-friendly choice) + Lawful (fits the
    # sword-and-prayer playstyle the prayer specialist can lean on).
    kwargs: dict = {"observation_keys": OBS_KEYS}
    if env_id.startswith("NetHack"):
        kwargs["character"] = "val-hum-law-fem"
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    try:
        env = gym.make(env_id, **kwargs)
    except (TypeError, ValueError):
        # Fallback for env registrations that don't accept both kwargs
        # (older MiniHack subclasses used custom entry points that reject
        # `character`). Retry with just observation_keys.
        try:
            kw2: dict = {"observation_keys": OBS_KEYS}
            if max_episode_steps is not None:
                kw2["max_episode_steps"] = max_episode_steps
            env = gym.make(env_id, **kw2)
        except (TypeError, ValueError):
            env = gym.make(env_id)
    return _Wrap(env)


def make_vec_env(env_id: str, num_envs: int, seed: int | None = None,
                 num_workers: int | None = None) -> "gym.vector.VectorEnv | BatchedAsyncVectorEnv":
    """Vectorised env pool.

    - ``num_workers is None or 1``: classic ``gym.vector.AsyncVectorEnv`` with
      one NLE subprocess per env. Fastest per-step but pays ~140 MB/env in
      RSS for the subproc, so caps at ~32 envs on a 14 GiB box.
    - ``num_workers >= 1`` and ``num_envs > num_workers``: chunked subproc
      backend (``BatchedAsyncVectorEnv``) where each of the W workers owns
      ``num_envs / W`` envs in a single process. NLE state is mostly C-side
      (~3 MB marginal/env, ~511 MB process baseline), so 8 workers × 16 envs
      = 128 envs fits in <6 GB instead of 18 GB. Step latency is the
      slowest worker's serial scan over its chunk, but still pipelines with
      the GPU forward.
    """
    if num_workers is None or num_workers <= 1 or num_envs <= num_workers:
        fns = [(lambda: make_env(env_id)) for _ in range(num_envs)]
        vec = gym.vector.AsyncVectorEnv(fns)
        if seed is not None:
            vec.reset(seed=[seed + i for i in range(num_envs)])
        return vec
    return BatchedAsyncVectorEnv(env_id, num_envs=num_envs,
                                 num_workers=num_workers, seed=seed)


def action_mask_for(env_id: str) -> np.ndarray:
    """Probe the env once to get its canonical-action mask."""
    env = make_env(env_id)
    try:
        return env.action_mask
    finally:
        env.close()


def _chunk_worker(env_id: str, chunk_size: int, base_seed: int,
                  conn, max_episode_steps: int | None = None) -> None:
    """Subprocess body for ``BatchedAsyncVectorEnv``.

    Owns ``chunk_size`` real envs in one process. Receives commands over a
    duplex Pipe; replies on the same Pipe.

    Protocol:
      ('reset', seeds: list[int])           -> (obs_chunk: dict, infos: list)
      ('step',  actions: ndarray (M,))      -> (obs_chunk, rewards (M,) f32,
                                                terms (M,) bool, truncs (M,) bool,
                                                infos: list[dict])  with auto-reset
                                                on done (gym AsyncVec parity)
      ('close', None)                        -> None  then exit
    """
    _register_envs(env_id)
    envs = [make_env(env_id, max_episode_steps=max_episode_steps) for _ in range(chunk_size)]
    obs_states: list[dict] = [None] * chunk_size  # last obs per env after auto-reset
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                seeds = payload
                obss = []
                infos = []
                for i, env in enumerate(envs):
                    s = int(seeds[i]) if seeds is not None else None
                    o, inf = env.reset(seed=s)
                    obss.append(o); infos.append(inf)
                    obs_states[i] = o
                obs_chunk = {k: np.stack([o[k] for o in obss], axis=0)
                             for k in OBS_KEYS}
                conn.send((obs_chunk, infos))
            elif cmd == "step":
                actions = payload
                rewards = np.empty(chunk_size, dtype=np.float32)
                terms = np.empty(chunk_size, dtype=bool)
                truncs = np.empty(chunk_size, dtype=bool)
                obss = [None] * chunk_size
                infos: list[dict] = [{} for _ in range(chunk_size)]
                for i, env in enumerate(envs):
                    o, r, term, trunc, info = env.step(int(actions[i]))
                    rewards[i] = r
                    terms[i] = bool(term)
                    truncs[i] = bool(trunc)
                    if term or trunc:
                        # Auto-reset to match gym AsyncVectorEnv semantics: on
                        # done the returned obs is the *first obs of the next
                        # episode*, with the terminal info bundled.
                        info_final = info
                        for _reset_attempt in range(3):
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                o, _info_reset = env.reset()
                            if not any("moveloop" in str(w.message) for w in caught):
                                break
                        info = {"final_observation": None, "final_info": info_final}
                    obss[i] = o
                    obs_states[i] = o
                    infos[i] = info
                obs_chunk = {k: np.stack([o[k] for o in obss], axis=0)
                             for k in OBS_KEYS}
                conn.send((obs_chunk, rewards, terms, truncs, infos))
            elif cmd == "close":
                break
            else:
                raise RuntimeError(f"unknown cmd {cmd!r}")
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        conn.close()


class BatchedAsyncVectorEnv:
    """W subprocs × M envs vec env. Mimics the slice of gymnasium vector API
    the trainer actually uses: ``num_envs``, ``reset(seed=[...])``,
    ``step(actions_np)``, ``close()``.

    Returns the gymnasium-style "dict-of-arrays + final_info" info bundle
    via the ``final_info`` list, so the trainer's ``_gym_async_per_env_infos``
    keeps working.
    """
    def __init__(self, env_id: str, *, num_envs: int, num_workers: int,
                 seed: int | None = None,
                 max_episode_steps: int | None = None):
        import multiprocessing as mp
        if num_envs % num_workers != 0:
            raise ValueError(f"num_envs ({num_envs}) must divide num_workers "
                             f"({num_workers})")
        self.env_id = env_id
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.chunk = num_envs // num_workers
        ctx = mp.get_context("spawn")
        self._procs = []
        self._conns = []
        for w in range(num_workers):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(target=_chunk_worker,
                            args=(env_id, self.chunk,
                                  (seed or 0) + w * self.chunk, child_conn,
                                  max_episode_steps),
                            daemon=True)
            p.start()
            child_conn.close()
            self._procs.append(p)
            self._conns.append(parent_conn)
        if seed is not None:
            self.reset(seed=[seed + i for i in range(num_envs)])
        self._send_pending = False

    def reset(self, *, seed=None):
        if seed is None:
            seed_chunks = [None] * self.num_workers
        else:
            assert len(seed) == self.num_envs
            seed_chunks = [seed[w*self.chunk:(w+1)*self.chunk]
                           for w in range(self.num_workers)]
        for c, sc in zip(self._conns, seed_chunks):
            c.send(("reset", sc))
        chunks = [c.recv() for c in self._conns]
        obs_chunks = [ck[0] for ck in chunks]
        infos = [inf for ck in chunks for inf in ck[1]]
        obs = {k: np.concatenate([oc[k] for oc in obs_chunks], axis=0)
               for k in OBS_KEYS}
        return obs, {"final_info": infos}

    def step_send(self, actions: np.ndarray) -> None:
        """Fan out one step command per worker chunk. Returns immediately;
        pair with :meth:`step_recv`. Used by the concurrent multi-spec
        trainer to overlap subprocess env-step work across multiple pools.
        """
        a = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        for w, c in enumerate(self._conns):
            c.send(("step", a[w*self.chunk:(w+1)*self.chunk]))
        self._send_pending = True

    def step_recv(self):
        """Drain replies for the most recent :meth:`step_send`."""
        if not self._send_pending:
            raise RuntimeError("step_recv called without a pending step_send")
        chunks = [c.recv() for c in self._conns]
        obs = {k: np.concatenate([ck[0][k] for ck in chunks], axis=0)
               for k in OBS_KEYS}
        rewards = np.concatenate([ck[1] for ck in chunks], axis=0)
        terms = np.concatenate([ck[2] for ck in chunks], axis=0)
        truncs = np.concatenate([ck[3] for ck in chunks], axis=0)
        # gym-style infos: per-env terminal entries go into final_info.
        per_env_infos: list[dict] = []
        for ck in chunks:
            per_env_infos.extend(ck[4])
        self._send_pending = False
        return obs, rewards, terms, truncs, {"final_info": per_env_infos}

    def step(self, actions: np.ndarray):
        """Synchronous step: send-then-drain. Equivalent to
        :meth:`step_send` then :meth:`step_recv`."""
        self.step_send(actions)
        return self.step_recv()

    def close(self):
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        for c in self._conns:
            try: c.close()
            except Exception: pass


def obs_to_tensor(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Batched numpy obs → tensors on device, no extra copies."""
    return {
        "glyphs": torch.from_numpy(np.ascontiguousarray(obs["glyphs"], dtype=np.int16)).to(device, non_blocking=True),
        "blstats": torch.from_numpy(np.ascontiguousarray(obs["blstats"], dtype=np.float32)).to(device, non_blocking=True),
        "message": torch.from_numpy(np.ascontiguousarray(obs["message"], dtype=np.uint8)).to(device, non_blocking=True),
    }
