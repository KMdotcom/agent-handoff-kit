"""SQLite-backed checkpoint, run-meta, and tool-invocation persistence.

All SQL lives in this module so a future Postgres/Redis backend only requires
swapping this file (or implementing the same method surface).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Union

from agent_relay.models import Checkpoint, HandoffStatus

Row = tuple  # matches Checkpoint.to_row()


class CheckpointStore:
    """Persist and query handoff checkpoints, run completion, and tool results.

    Interface is intentionally small so alternative backends can mirror it.
    """

    def __init__(self, db_path: Union[str, Path] = "agent_relay.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
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

    def save(self, checkpoint: Checkpoint) -> None:
        """Insert or replace a checkpoint row."""
        row = checkpoint.to_row()
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

    def update_status(
        self,
        checkpoint_id: str,
        status: HandoffStatus,
        error: Optional[str] = None,
    ) -> None:
        """Update status (and optional error) for an existing checkpoint."""
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

    def last_good_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
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

    # --- run completion meta ---

    def set_run_status(self, run_id: str, status: str) -> None:
        """Upsert run lifecycle status (``open`` | ``completed``)."""
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

    def get_run_status(self, run_id: str) -> Optional[str]:
        """Return run status or None if never recorded."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT status FROM run_meta WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        return None if row is None else str(row["status"])

    # --- tool idempotency ---

    def get_tool_result(self, run_id: str, idempotency_key: str) -> Optional[Any]:
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
        return json.loads(row["result_json"])

    def save_tool_result(
        self,
        run_id: str,
        idempotency_key: str,
        tool_name: str,
        result: Any,
    ) -> None:
        """Persist a tool result for idempotent replay (insert-or-ignore)."""
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
                    json.dumps(result),
                    time.time(),
                ),
            )
            conn.commit()
