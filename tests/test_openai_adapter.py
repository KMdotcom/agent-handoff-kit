"""Tests for the OpenAI adapter seam (no live network)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_relay import HandoffStatus, Relay
from agent_relay.openai_adapter import (
    context_from_run_context,
    resume_run,
    wrap_on_handoff,
)


class OpenAIAdapterTests(unittest.TestCase):
    def test_context_from_run_context_merges_dict_and_input(self) -> None:
        ctx = SimpleNamespace(context={"ticket_id": "T-1", "summary": "old"})
        out = context_from_run_context(ctx, {"summary": "new", "category": "billing"})
        self.assertEqual(
            out,
            {"ticket_id": "T-1", "summary": "new", "category": "billing"},
        )

    def test_wrap_on_handoff_verifies_before_user_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            relay = Relay(db)
            order: list[str] = []

            def on_handoff(ctx):  # noqa: ANN001
                order.append("user")
                # Checkpoint must already be VERIFIED when user code runs.
                cp = relay.recover("run-1")
                assert cp is not None
                self.assertEqual(cp.status, HandoffStatus.VERIFIED)
                order.append("verified_seen")

            wrapped = wrap_on_handoff(
                relay,
                run_id="run-1",
                from_agent="triage",
                to_agent="resolver",
                required_keys=["ticket_id", "summary"],
                on_handoff=on_handoff,
            )
            ctx = SimpleNamespace(
                context={"ticket_id": "T-1", "summary": "Double charge"}
            )
            asyncio.run(wrapped(ctx))
            self.assertEqual(order, ["user", "verified_seen"])
            self.assertIsNotNone(resume_run(relay, "run-1"))

    def test_wrap_on_handoff_fails_preflight_on_missing_keys(self) -> None:
        from agent_relay import HandoffVerificationError

        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            wrapped = wrap_on_handoff(
                relay,
                run_id="run-2",
                from_agent="triage",
                to_agent="resolver",
                required_keys=["ticket_id", "summary"],
            )
            ctx = SimpleNamespace(context={"ticket_id": "T-1"})  # missing summary
            with self.assertRaises(HandoffVerificationError):
                asyncio.run(wrapped(ctx))
            self.assertIsNone(relay.recover("run-2"))
            hist = relay.store.history("run-2")
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0].status, HandoffStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
