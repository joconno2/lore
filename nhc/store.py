"""Run-scoped artifact store: one SQLite DB, one checkpoint tree per run.

Replaces the legacy per-chunk `.db` mess (794 files found in one run) and
the ad-hoc `outputs/` layout with a single `runs/<run_id>/` directory:

    runs/<run_id>/
        metrics.db                # scalars, episodes, events
        checkpoints/
            specialists/<sid>.pt
            consensus/latest.pt
        registry.json
        config.json
        log.txt

All writers use atomic rename; concurrent chunk writers serialize through
SQLite's default journaling.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Process-wide lock serializing cross-DB ATTACH+INSERT merges. The driver
# unpacks run artifacts as Ray workers finish, potentially from a threadpool
# (e.g. `ray.wait` callbacks, concurrent.futures), so even though SQLite
# itself serializes writers via WAL, ATTACH-based multi-table INSERTs across
# connections must not interleave on the same destination.
_MERGE_LOCK = threading.Lock()


class MetricsDB:
    """SQLite metrics store with the schema the trainer actually emits.

    Tables:
      scalars(t REAL, run_scope TEXT, sid TEXT, step INTEGER, key TEXT, value REAL)
      episodes(t REAL, sid TEXT, step INTEGER, episode INTEGER, env_id TEXT,
               reward REAL, length INTEGER, success INTEGER)
      events(t REAL, sid TEXT, step INTEGER, kind TEXT, payload TEXT)
    """

    SCHEMA = [
        """CREATE TABLE IF NOT EXISTS scalars(
            t REAL, run_scope TEXT, sid TEXT, step INTEGER, key TEXT, value REAL
        )""",
        """CREATE TABLE IF NOT EXISTS episodes(
            t REAL, sid TEXT, step INTEGER, episode INTEGER, env_id TEXT,
            reward REAL, length INTEGER, success INTEGER
        )""",
        """CREATE TABLE IF NOT EXISTS events(
            t REAL, sid TEXT, step INTEGER, kind TEXT, payload TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS ix_scalars_key ON scalars(sid, key, step)",
        "CREATE INDEX IF NOT EXISTS ix_episodes_sid ON episodes(sid, step)",
    ]

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in self.SCHEMA:
            self.conn.execute(stmt)

    def log_scalars(self, sid: str, step: int, metrics: dict[str, float],
                    run_scope: str = "specialist") -> None:
        t = time.time()
        rows = [(t, run_scope, sid, step, k, float(v)) for k, v in metrics.items()
                if isinstance(v, (int, float))]
        if rows:
            self.conn.executemany(
                "INSERT INTO scalars VALUES (?,?,?,?,?,?)", rows
            )

    def log_episode(self, sid: str, step: int, episode: int, env_id: str,
                    reward: float, length: int, success: bool) -> None:
        self.conn.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), sid, step, episode, env_id,
             float(reward), int(length), int(bool(success))),
        )

    def log_event(self, sid: str, step: int, kind: str, payload: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?)",
            (time.time(), sid, step, kind, json.dumps(payload or {})),
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class RunDir:
    """One-stop handle to a run's artifact tree.

    Convention:
      runs/<run_id>/
          metrics.db
          registry.json
          config.json
          checkpoints/specialists/<sid>.pt
          checkpoints/specialists/<sid>.scheduler.json
          checkpoints/consensus/latest.pt
          log.txt
    """
    def __init__(self, run_id: str, root: Path = Path("runs")):
        self.run_id = run_id
        self.root = Path(root) / run_id
        self.ckpt_dir = self.root / "checkpoints"
        self.spec_dir = self.ckpt_dir / "specialists"
        self.cons_dir = self.ckpt_dir / "consensus"
        self.db_path = self.root / "metrics.db"
        self.registry_path = self.root / "registry.json"
        self.config_path = self.root / "config.json"
        self.log_path = self.root / "log.txt"
        for d in (self.root, self.spec_dir, self.cons_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resolve(cls, run_id: str, root: Path | str = Path("runs")) -> "RunDir":
        return cls(run_id, Path(root))

    @contextmanager
    def metrics(self) -> Iterator[MetricsDB]:
        db = MetricsDB(self.db_path)
        try:
            yield db
        finally:
            db.close()

    def write_config(self, cfg: dict) -> None:
        self.config_path.write_text(json.dumps(cfg, indent=2, default=str))

    def spec_ckpt(self, sid: str) -> Path:
        return self.spec_dir / f"{sid}.pt"

    def spec_sched(self, sid: str) -> Path:
        return self.spec_dir / f"{sid}.scheduler.json"

    def spec_aux(self, sid: str) -> Path:
        """Extra state (running-mean reward, optimizer, rng)."""
        return self.spec_dir / f"{sid}.aux.json"

    def consensus_ckpt(self) -> Path:
        return self.cons_dir / "latest.pt"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    tmp.replace(path)


def pack_run_dir(rd: RunDir) -> bytes:
    """Tar.gz the run-dir tree so it can be shipped back through Ray.

    Workers write to a local (per-node) run dir, then return the packed
    bytes to the driver, which unpacks into the shared driver-side tree.
    """
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if rd.root.exists():
            tf.add(str(rd.root), arcname=rd.run_id)
    return buf.getvalue()


def unpack_run_dir(blob: bytes, dest_root: Path) -> None:
    """Inverse of `pack_run_dir`. Extracts under `dest_root`.

    If the packed archive contains a `metrics.db` and the destination
    already has one, rows are merged (scalars/episodes/events all append).
    Checkpoint files simply overwrite — they're per-sid keys in their own
    sub-paths so there's no collision."""
    import io
    import tarfile
    import tempfile
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nhc_unpack_") as td:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(td)
        stage = Path(td)
        for p in sorted(stage.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(stage)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.name == "metrics.db" and dest.exists():
                _merge_metrics_db(src=p, dst=dest)
            else:
                p.replace(dest)


def _hash_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _merge_metrics_db(src: Path, dst: Path) -> None:
    """Append rows from src.metrics.db into dst.metrics.db and delete src.

    Thread-safe: serialized by a process-wide lock (``_MERGE_LOCK``) so that
    concurrent unpacks from a driver-side threadpool cannot interleave their
    ATTACH+INSERT statements on the same destination DB.

    Idempotent: each source is identified by its SHA-256; a `_merged_sources`
    table in the destination records successful merges, so re-invoking with
    the same source is a no-op (rows are not doubled). If the same run is
    merged twice from different source files (same content, fresh path), the
    content hash still dedups.
    """
    src_hash = _hash_file(src)
    with _MERGE_LOCK:
        dst_conn = sqlite3.connect(str(dst), timeout=30.0, isolation_level=None)
        try:
            # Tolerate short contention under WAL mode. Even with the
            # process lock, cross-process writers (if ever introduced) or
            # readers holding WAL may briefly block the writer.
            dst_conn.execute("PRAGMA busy_timeout = 30000")
            dst_conn.execute("PRAGMA journal_mode=WAL")
            dst_conn.execute(
                "CREATE TABLE IF NOT EXISTS _merged_sources("
                "hash TEXT PRIMARY KEY, merged_at REAL)"
            )
            already = dst_conn.execute(
                "SELECT 1 FROM _merged_sources WHERE hash = ?", (src_hash,)
            ).fetchone()
            if already:
                # Already merged — drop the duplicate source and return.
                src.unlink()
                return
            dst_conn.execute("ATTACH DATABASE ? AS src", (str(src),))
            try:
                dst_conn.execute("BEGIN IMMEDIATE")
                try:
                    for table in ("scalars", "episodes", "events"):
                        dst_conn.execute(
                            f"INSERT INTO {table} SELECT * FROM src.{table}"
                        )
                    dst_conn.execute(
                        "INSERT OR IGNORE INTO _merged_sources"
                        "(hash, merged_at) VALUES (?, ?)",
                        (src_hash, time.time()),
                    )
                    dst_conn.execute("COMMIT")
                except Exception:
                    dst_conn.execute("ROLLBACK")
                    raise
            finally:
                dst_conn.execute("DETACH DATABASE src")
        finally:
            dst_conn.close()
    src.unlink()
