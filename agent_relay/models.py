"""Checkpoint data model, status enum, and handoff envelope helpers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union


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


def is_handoff_envelope(payload: dict[str, Any]) -> bool:
    """True when ``payload`` uses the MVP envelope shape (state/messages/meta)."""
    return (
        isinstance(payload, dict)
        and "state" in payload
        and "meta" in payload
        and isinstance(payload.get("state"), dict)
        and isinstance(payload.get("meta"), dict)
    )


def make_handoff_envelope(
    state: dict[str, Any],
    *,
    from_agent: str,
    to_agent: str,
    messages: Optional[list[Any]] = None,
    sdk: str = "openai-agents",
) -> dict[str, Any]:
    """Build the structured checkpoint payload stored in ``Checkpoint.context``."""
    return {
        "state": dict(state),
        "messages": list(messages or []),
        "meta": {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "sdk": sdk,
        },
    }


def checkpoint_state(checkpoint: Union[Checkpoint, dict[str, Any]]) -> dict[str, Any]:
    """App state dict from a checkpoint (envelope ``state`` or legacy flat context)."""
    payload = checkpoint.context if isinstance(checkpoint, Checkpoint) else checkpoint
    if is_handoff_envelope(payload):
        return dict(payload["state"])
    return dict(payload)


def checkpoint_messages(checkpoint: Union[Checkpoint, dict[str, Any]]) -> list[Any]:
    """Message list from an envelope checkpoint; empty for legacy flat payloads."""
    payload = checkpoint.context if isinstance(checkpoint, Checkpoint) else checkpoint
    if is_handoff_envelope(payload):
        msgs = payload.get("messages") or []
        return list(msgs) if isinstance(msgs, list) else []
    return []
