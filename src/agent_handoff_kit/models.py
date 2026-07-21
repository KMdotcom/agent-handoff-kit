"""Checkpoint shapes and the handoff envelope helpers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HandoffStatus(str, Enum):
    """Where a handoff checkpoint sits in its lifecycle.

    PENDING   — written, not verified yet
    VERIFIED  — required keys present (safe rollback point)
    FAILED    — pre-flight failed (missing keys)
    RECOVERED — marked after a successful resume
    """

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    RECOVERED = "RECOVERED"


def json_sanitize(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures.

    Handles nested dicts/lists, Pydantic ``model_dump`` / ``.dict()``, and
    falls back to ``str(...)`` for leaves that still fail ``json.dumps``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return json_sanitize(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return json_sanitize(value.dict())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def dumps_safe(value: Any) -> str:
    """``json.dumps`` after ``json_sanitize`` so callers need not pre-serialize."""
    return json.dumps(json_sanitize(value))


def loads_field(raw: str, field_name: str) -> Any:
    """``json.loads`` with a clear ``ValueError`` naming the corrupt field."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON in checkpoint field {field_name!r}: {exc}") from exc


@dataclass
class Checkpoint:
    """Snapshot of state at an agent-to-agent handoff boundary."""

    run_id: str
    from_agent: str
    to_agent: str
    context: dict[str, Any]
    required_keys: list[str]
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    status: HandoffStatus = HandoffStatus.PENDING
    error: str | None = None

    def to_row(self) -> tuple[Any, ...]:
        """SQLite row (JSON-encode dict/list fields via safe serialize)."""
        return (
            self.checkpoint_id,
            self.run_id,
            self.from_agent,
            self.to_agent,
            dumps_safe(self.context),
            dumps_safe(self.required_keys),
            self.created_at,
            self.status.value,
            self.error,
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Checkpoint:
        """Rebuild from ``to_row`` output."""
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
            context=loads_field(context_json, "context"),
            required_keys=loads_field(required_keys_json, "required_keys"),
            created_at=created_at,
            status=HandoffStatus(status),
            error=error,
        )


def is_handoff_envelope(payload: dict[str, Any]) -> bool:
    """True if payload looks like ``{state, messages, meta}``."""
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
    messages: list[Any] | None = None,
    sdk: str = "openai-agents",
) -> dict[str, Any]:
    """Build the structured blob we store in ``Checkpoint.context``."""
    return {
        "state": dict(state),
        "messages": list(messages or []),
        "meta": {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "sdk": sdk,
        },
    }


def checkpoint_state(checkpoint: Checkpoint | dict[str, Any]) -> dict[str, Any]:
    """App state from an envelope, or the flat dict for older checkpoints."""
    payload = checkpoint.context if isinstance(checkpoint, Checkpoint) else checkpoint
    if is_handoff_envelope(payload):
        return dict(payload["state"])
    return dict(payload)


def checkpoint_messages(checkpoint: Checkpoint | dict[str, Any]) -> list[Any]:
    """Messages from an envelope; empty list for flat legacy payloads."""
    payload = checkpoint.context if isinstance(checkpoint, Checkpoint) else checkpoint
    if is_handoff_envelope(payload):
        msgs = payload.get("messages") or []
        return list(msgs) if isinstance(msgs, list) else []
    return []
