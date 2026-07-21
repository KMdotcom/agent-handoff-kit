"""Checkpoint data model and status enum."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HandoffStatus(str, Enum):
    """Lifecycle of a handoff checkpoint.

    PENDING   — saved, not yet verified
    VERIFIED  — receiving agent has all required keys (safe rollback point)
    FAILED    — pre-flight verification failed (missing required keys)
    RECOVERED — reserved for callers that mark a restored checkpoint as used
    """

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    RECOVERED = "RECOVERED"


@dataclass
class Checkpoint:
    """A point-in-time snapshot of state at an agent-to-agent handoff boundary."""

    run_id: str
    from_agent: str
    to_agent: str
    context: dict[str, Any]
    required_keys: list[str]
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    status: HandoffStatus = HandoffStatus.PENDING
    error: Optional[str] = None

    def to_row(self) -> tuple:
        """Serialize to a SQLite-friendly tuple (JSON-encode dict/list fields)."""
        return (
            self.checkpoint_id,
            self.run_id,
            self.from_agent,
            self.to_agent,
            json.dumps(self.context),
            json.dumps(self.required_keys),
            self.created_at,
            self.status.value,
            self.error,
        )

    @classmethod
    def from_row(cls, row: tuple) -> Checkpoint:
        """Rebuild a Checkpoint from a DB row produced by ``to_row``."""
        (
            checkpoint_id,
            run_id,
            from_agent,
            to_agent,
            context_json,
            required_keys_json,
            created_at,
            status,
            error,
        ) = row
        return cls(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            from_agent=from_agent,
            to_agent=to_agent,
            context=json.loads(context_json),
            required_keys=json.loads(required_keys_json),
            created_at=created_at,
            status=HandoffStatus(status),
            error=error,
        )
