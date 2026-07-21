"""agent-handoff-kit — recover multi-agent handoffs without a heavy workflow engine."""

from agent_handoff_kit.core import HandoffVerificationError, Relay
from agent_handoff_kit.models import (
    Checkpoint,
    HandoffStatus,
    checkpoint_messages,
    checkpoint_state,
    make_handoff_envelope,
)
from agent_handoff_kit.store import CheckpointStore

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
