"""agent-relay: lightweight recovery for multi-agent handoffs."""

from agent_relay.core import HandoffVerificationError, Relay
from agent_relay.models import (
    Checkpoint,
    HandoffStatus,
    checkpoint_messages,
    checkpoint_state,
    make_handoff_envelope,
)
from agent_relay.store import CheckpointStore

__version__ = "0.1.0"

__all__ = [
    "Relay",
    "HandoffVerificationError",
    "Checkpoint",
    "HandoffStatus",
    "CheckpointStore",
    "checkpoint_state",
    "checkpoint_messages",
    "make_handoff_envelope",
    "__version__",
]
