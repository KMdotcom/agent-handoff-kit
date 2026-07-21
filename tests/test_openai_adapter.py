"""Tests for envelope helpers, OpenAI adapter, DurableRunner, idempotency."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from handoff_kit import (
    HandoffStatus,
    HandoffVerificationError,
    Relay,
    checkpoint_messages,
    checkpoint_state,
    make_handoff_envelope,
)
from handoff_kit.idempotency import idempotent_tool
from handoff_kit.openai_adapter import (
    DurableRunner,
    RunAlreadyCompleted,
    build_envelope_from_run_context,
    context_from_run_context,
    resume_run,
    wrap_on_handoff,
)


class EnvelopeTests(unittest.TestCase):
    def test_checkpoint_state_and_messages(self) -> None:
        env = make_handoff_envelope(
            {"ticket_id": "T-1", "summary": "x"},
            from_agent="triage",
            to_agent="resolver",
            messages=[{"type": "message", "text": "hi"}],
        )
        self.assertEqual(checkpoint_state(env)["ticket_id"], "T-1")
        self.assertEqual(len(checkpoint_messages(env)), 1)

    def test_legacy_flat_context(self) -> None:
        flat = {"ticket_id": "T-1"}
        self.assertEqual(checkpoint_state(flat), flat)
        self.assertEqual(checkpoint_messages(flat), [])

    def test_verify_uses_envelope_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            env = make_handoff_envelope(
                {"ticket_id": "T-1", "summary": "ok"},
                from_agent="a",
                to_agent="b",
            )
            cp = relay.checkpoint("r", "a", "b", env, ["ticket_id", "summary"])
            relay.verify(cp)
            self.assertEqual(cp.status, HandoffStatus.VERIFIED)


class OpenAIAdapterTests(unittest.TestCase):
    def test_context_from_run_context_merges_dict_and_input(self) -> None:
        ctx = SimpleNamespace(context={"ticket_id": "T-1", "summary": "old"})
        out = context_from_run_context(ctx, {"summary": "new", "category": "billing"})
        self.assertEqual(
            out,
            {"ticket_id": "T-1", "summary": "new", "category": "billing"},
        )

    def test_build_envelope_includes_turn_input(self) -> None:
        ctx = SimpleNamespace(
            context={"ticket_id": "T-1", "summary": "s"},
            turn_input=[{"type": "message", "content": "hello"}],
        )
        env = build_envelope_from_run_context(
            ctx, from_agent="triage", to_agent="resolver"
        )
        self.assertIn("state", env)
        self.assertEqual(env["state"]["ticket_id"], "T-1")
        self.assertEqual(len(env["messages"]), 1)

    def test_wrap_on_handoff_verifies_before_user_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            order: list[str] = []

            def on_handoff(ctx):  # noqa: ANN001
                order.append("user")
                cp = relay.recover("run-1")
                assert cp is not None
                self.assertEqual(cp.status, HandoffStatus.VERIFIED)
                self.assertEqual(checkpoint_state(cp)["ticket_id"], "T-1")
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
                context={"ticket_id": "T-1", "summary": "Double charge"},
                turn_input=[],
            )
            asyncio.run(wrapped(ctx))
            self.assertEqual(order, ["user", "verified_seen"])
            payload = resume_run(relay, "run-1")
            assert payload is not None
            self.assertEqual(checkpoint_state(payload)["summary"], "Double charge")

    def test_wrap_on_handoff_fails_preflight_on_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            wrapped = wrap_on_handoff(
                relay,
                run_id="run-2",
                from_agent="triage",
                to_agent="resolver",
                required_keys=["ticket_id", "summary"],
            )
            ctx = SimpleNamespace(context={"ticket_id": "T-1"}, turn_input=[])
            with self.assertRaises(HandoffVerificationError):
                asyncio.run(wrapped(ctx))
            self.assertIsNone(relay.recover("run-2"))
            hist = relay.store.history("run-2")
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0].status, HandoffStatus.FAILED)


class IdempotencyTests(unittest.TestCase):
    def test_idempotent_tool_caches_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            calls = {"n": 0}

            @idempotent_tool(relay, "run-x", key_fn=lambda ticket_id: f"r:{ticket_id}")
            def refund(ticket_id: str) -> str:
                calls["n"] += 1
                return f"ok:{ticket_id}"

            self.assertEqual(refund("T-1"), "ok:T-1")
            self.assertEqual(refund("T-1"), "ok:T-1")
            self.assertEqual(calls["n"], 1)


class DurableRunnerTests(unittest.TestCase):
    def test_completed_run_refuses_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            relay.store.set_run_status("done-1", "completed")

            async def _go() -> None:
                await DurableRunner.run(
                    relay,
                    "done-1",
                    starting_agent=object(),
                    user_input="x",
                    agents={},
                )

            with self.assertRaises(RunAlreadyCompleted):
                asyncio.run(_go())

    def test_cold_resume_uses_destination_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            relay = Relay(Path(tmp) / "t.db")
            env = make_handoff_envelope(
                {"ticket_id": "T-1", "summary": "s"},
                from_agent="triage",
                to_agent="resolver",
            )
            cp = relay.checkpoint(
                "cold-1", "triage", "resolver", env, ["ticket_id", "summary"]
            )
            relay.verify(cp)
            relay.store.set_run_status("cold-1", "open")

            calls: list[str] = []

            class FakeAgent:
                name = "Resolver"

            async def fake_runner_run(agent, user_input, **kwargs):  # noqa: ANN001
                calls.append(getattr(agent, "name", "?"))
                return SimpleNamespace(final_output="resumed")

            import handoff_kit.openai_adapter as oa

            real_resume = oa.resume_with_agent

            async def fake_resume(relay, run_id, agent, **kwargs):  # noqa: ANN001
                calls.append(f"resume:{agent.name}")
                relay.store.update_status(cp.checkpoint_id, HandoffStatus.RECOVERED)
                return SimpleNamespace(final_output="resumed")

            oa.resume_with_agent = fake_resume  # type: ignore[assignment]
            try:
                result = asyncio.run(
                    DurableRunner.run(
                        relay,
                        "cold-1",
                        starting_agent=FakeAgent(),
                        user_input="hi",
                        agents={"resolver": FakeAgent()},
                    )
                )
            finally:
                oa.resume_with_agent = real_resume  # type: ignore[assignment]

            self.assertEqual(result.final_output, "resumed")
            self.assertIn("resume:Resolver", calls)
            self.assertEqual(relay.store.get_run_status("cold-1"), "completed")


if __name__ == "__main__":
    unittest.main()
