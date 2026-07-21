"""Core Relay checkpoint / verify / recover tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_handoff_kit import (
    HandoffStatus,
    HandoffVerificationError,
    Relay,
    checkpoint_state,
    make_handoff_envelope,
)
from agent_handoff_kit.models import Checkpoint, dumps_safe, json_sanitize, loads_field


class _FakeModel:
    """Pydantic-like object with model_dump (not JSON-serializable itself)."""

    def __init__(self, **data: object) -> None:
        self._data = data

    def model_dump(self) -> dict:
        return dict(self._data)


class CheckpointCoreTests(unittest.TestCase):
    def test_verify_then_recover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            env = make_handoff_envelope(
                {"ticket_id": "T-1", "summary": "ok"},
                from_agent="a",
                to_agent="b",
            )
            cp = relay.checkpoint("r1", "a", "b", env, ["ticket_id", "summary"])
            relay.verify(cp)
            recovered = relay.recover("r1")
            assert recovered is not None
            self.assertEqual(recovered.status, HandoffStatus.VERIFIED)
            self.assertEqual(checkpoint_state(recovered)["ticket_id"], "T-1")

    def test_missing_keys_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            env = make_handoff_envelope(
                {"ticket_id": "T-1"},
                from_agent="a",
                to_agent="b",
            )
            cp = relay.checkpoint("r2", "a", "b", env, ["ticket_id", "summary"])
            with self.assertRaises(HandoffVerificationError):
                relay.verify(cp)
            self.assertIsNone(relay.recover("r2"))
            hist = relay.store.history("r2")
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0].status, HandoffStatus.FAILED)

    def test_guarded_handoff_keeps_verified_on_body_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")

            @relay.guarded_handoff(
                run_id="r3",
                from_agent="a",
                to_agent="b",
                required_keys=["ticket_id"],
            )
            def boom(context: dict) -> dict:
                raise RuntimeError("billing API down")

            with self.assertRaises(RuntimeError):
                boom({"ticket_id": "T-1", "summary": "x"})

            recovered = relay.recover("r3")
            assert recovered is not None
            self.assertEqual(recovered.status, HandoffStatus.VERIFIED)
            self.assertEqual(checkpoint_state(recovered)["ticket_id"], "T-1")

    def test_nested_model_dump_does_not_blow_up_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            state = {
                "ticket_id": "T-1",
                "payload": _FakeModel(amount=42, note="refund"),
            }
            env = make_handoff_envelope(state, from_agent="a", to_agent="b")
            cp = relay.checkpoint("r4", "a", "b", env, ["ticket_id"])
            relay.verify(cp)
            recovered = relay.recover("r4")
            assert recovered is not None
            nested = checkpoint_state(recovered)["payload"]
            self.assertEqual(nested, {"amount": 42, "note": "refund"})

    def test_json_sanitize_and_loads_field(self) -> None:
        sanitized = json_sanitize({"m": _FakeModel(x=1), "raw": object()})
        self.assertEqual(sanitized["m"], {"x": 1})
        self.assertIsInstance(sanitized["raw"], str)
        dumps_safe({"ok": True})
        with self.assertRaises(ValueError) as ctx:
            loads_field("{not-json", "context")
        self.assertIn("context", str(ctx.exception))

    def test_corrupt_checkpoint_context_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Checkpoint.from_row(
                (
                    "id",
                    "run",
                    "a",
                    "b",
                    "{bad",
                    '["ticket_id"]',
                    1.0,
                    "PENDING",
                    None,
                )
            )
        self.assertIn("context", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
