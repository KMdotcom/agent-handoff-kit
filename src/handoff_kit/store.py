"""SQLite storage for checkpoints, run completion, and idempotent tool results.

Keep SQL here so a later Postgres/Redis backend only has to replace this module.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from handoff_kit.models import Checkpoint, HandoffStatus, dumps_safe, loads_field


class CheckpointStore:
    """Persist and query handoff checkpoints, run completion, and tool results.

    Interface is intentionally small so alternative backends can mirror it.
    """

    def __init__(self, db_path: str | Path = "handoff_kit.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _write_error(self, operation: str, exc: sqlite3.Error) -> sqlite3.Error:
        return type(exc)(f"SQLite {operation} failed for db_path={self.db_path!r}: {exc}")

    def _init_db(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        from_agent TEXT NOT NULL,
                        to_agent TEXT NOT NULL,
                        context TEXT NOT NULL,
                        required_keys TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        status TEXT NOT NULL,
                        error TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id
                    ON checkpoints (run_id, created_at)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_meta (
                        run_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_invocations (
                        run_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY (run_id, idempotency_key)
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise self._write_error("init_db", exc) from exc

    def save(self, checkpoint: Checkpoint) -> None:
        """Insert or replace a checkpoint row."""
        row = checkpoint.to_row()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (
                        checkpoint_id, run_id, from_agent, to_agent,
                        context, required_keys, created_at, status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise self._write_error("save checkpoint", exc) from exc

    def update_status(
        self,
        checkpoint_id: str,
        status: HandoffStatus,
        error: str | None = None,
    ) -> None:
        """Update status (and optional error) for an existing checkpoint."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE checkpoints
                    SET status = ?, error = ?
                    WHERE checkpoint_id = ?
                    """,
                    (status.value, error, checkpoint_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise self._write_error("update_status", exc) from exc

    def last_good_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Most recent VERIFIED checkpoint for ``run_id`` — the safe rollback point."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT checkpoint_id, run_id, from_agent, to_agent,
                       context, required_keys, created_at, status, error
                FROM checkpoints
                WHERE run_id = ? AND status = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_id, HandoffStatus.VERIFIED.value),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Checkpoint.from_row(tuple(row))

    def history(self, run_id: str) -> list[Checkpoint]:
        """Full ordered checkpoint history for a run (oldest first)."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT checkpoint_id, run_id, from_agent, to_agent,
                       context, required_keys, created_at, status, error
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return [Checkpoint.from_row(tuple(r)) for r in rows]

    def set_run_status(self, run_id: str, status: str) -> None:
        """Upsert run lifecycle status (``open`` | ``completed``)."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO run_meta (run_id, status, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status = excluded.status,
                        updated_at = excluded.updated_at
                    """,
                    (run_id, status, time.time()),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise self._write_error("set_run_status", exc) from exc

    def get_run_status(self, run_id: str) -> str | None:
        """Return run status or None if never recorded."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT status FROM run_meta WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        return None if row is None else str(row["status"])

    def get_tool_result(self, run_id: str, idempotency_key: str) -> Any | None:
        """Return a previously stored tool result, or None."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT result_json FROM tool_invocations
                WHERE run_id = ? AND idempotency_key = ?
                """,
                (run_id, idempotency_key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return loads_field(row["result_json"], "result_json")

    def save_tool_result(
        self,
        run_id: str,
        idempotency_key: str,
        tool_name: str,
        result: Any,
    ) -> None:
        """Persist a tool result for idempotent replay (insert-or-ignore)."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO tool_invocations (
                        run_id, idempotency_key, tool_name, result_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        idempotency_key,
                        tool_name,
                        dumps_safe(result),
                        time.time(),
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise self._write_error("save_tool_result", exc) from exc
