#!/usr/bin/env python3
"""BYO-style demo: native OpenAI Agents ``handoff`` via ``relayed_handoff``.

Shows the copy-paste MVP path:
  1. ``relayed_handoff(...)`` checkpoints at the SDK transfer boundary
  2. Receiving agent crashes mid-run (simulated ConnectionError)
  3. ``resume_run`` / ``resume_with_agent`` continue from VERIFIED context

Offline by default (scripted Model). Live::

    python3 demo/demo_openai_adapter.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(_ROOT / ".env")

try:
    from agents import Agent, ModelResponse, Runner, Usage, function_tool
    from agents.models.interface import Model, ModelTracing
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
except ImportError:
    print(
        "Missing dependency: openai-agents\n"
        "  pip install 'openai-agents'\n"
        "Then re-run this demo."
    )
    raise SystemExit(2) from None

from agent_relay import Relay
from agent_relay.openai_adapter import relayed_handoff, resume_run, resume_with_agent

DB_PATH = _ROOT / "demo" / "demo_openai_adapter.db"
_resolver_crashes_left = 1


def _text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class ScriptedTriageModel(Model):
    """Emit a native handoff tool call to transfer_to_resolver."""

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        call = ResponseFunctionToolCall(
            type="function_call",
            call_id=f"call_{uuid.uuid4().hex[:8]}",
            name="transfer_to_resolver",
            arguments="{}",
        )
        return ModelResponse(
            output=[call],
            usage=Usage(),
            response_id=f"resp_{uuid.uuid4().hex[:8]}",
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            raise NotImplementedError
            yield None  # pragma: no cover

        return _gen()


class ScriptedResolverModel(Model):
    """Crash once (simulates worker death), then return a final answer."""

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        global _resolver_crashes_left
        if _resolver_crashes_left > 0:
            _resolver_crashes_left -= 1
            raise ConnectionError("simulated worker crash mid-resolver")
        return ModelResponse(
            output=[
                _text_message(
                    "Refund of $42.00 approved for the double charge on invoice #8821."
                )
            ],
            usage=Usage(),
            response_id=f"resp_{uuid.uuid4().hex[:8]}",
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            raise NotImplementedError
            yield None  # pragma: no cover

        return _gen()


@function_tool
def note_resolution(notes: str) -> str:
    """Record a short resolution note (live-mode helper tool)."""
    return f"Noted: {notes}"


def build_agents(*, live: bool, relay: Relay, run_id: str) -> tuple[Agent, Agent]:
    if live:
        resolver = Agent(
            name="Resolver",
            instructions=(
                "Resolve the billing ticket using the context you were given. "
                "Reply with one short paragraph confirming the refund."
            ),
            tools=[note_resolution],
            model="gpt-4.1-mini",
        )
        triage = Agent(
            name="Triage",
            instructions=(
                "You are triage. Always hand off to the Resolver agent immediately "
                "using the transfer tool. Do not solve the ticket yourself."
            ),
            handoffs=[
                relayed_handoff(
                    relay,
                    run_id=run_id,
                    from_agent="triage",
                    to_agent="resolver",
                    agent=resolver,
                    required_keys=["ticket_id", "category", "summary"],
                )
            ],
            model="gpt-4.1-mini",
        )
        return triage, resolver

    resolver = Agent(
        name="Resolver",
        instructions="Resolve the ticket.",
        model=ScriptedResolverModel(),
    )
    triage = Agent(
        name="Triage",
        instructions="Hand off to resolver.",
        handoffs=[
            relayed_handoff(
                relay,
                run_id=run_id,
                from_agent="triage",
                to_agent="resolver",
                agent=resolver,
                required_keys=["ticket_id", "category", "summary"],
            )
        ],
        model=ScriptedTriageModel(),
    )
    return triage, resolver


async def run_pipeline(*, live: bool) -> int:
    print("=" * 60)
    mode = "LIVE (OpenAI API)" if live else "OFFLINE (scripted Model)"
    print(f"DEMO: relayed_handoff drop-in [{mode}]")
    print("=" * 60)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[setup] Removed existing {DB_PATH.name}")

    global _resolver_crashes_left
    _resolver_crashes_left = 0 if live else 1

    relay = Relay(DB_PATH)
    run_id = "adapter-demo-T-2001"
    triage, resolver = build_agents(live=live, relay=relay, run_id=run_id)

    # App-owned state dict — available as ctx.context inside on_handoff.
    context: dict[str, Any] = {
        "ticket_id": "T-2001",
        "category": "billing",
        "summary": "Double charge on invoice #8821",
        "customer": "Ada Lovelace",
    }
    print(f"[app] Starting context: {context}")

    print("\n→ Runner.run(triage) with relayed_handoff → resolver…")
    try:
        result = await Runner.run(
            triage,
            "I was double-charged on invoice #8821.",
            context=context,
        )
        if live:
            # Live models may succeed without our crash injection; still show
            # that a VERIFIED handoff checkpoint was written.
            print(f"[live] Run finished: {result.final_output!s}")
            history = relay.store.history(run_id)
            if not history:
                print("ERROR: expected a handoff checkpoint; none found.")
                return 1
            print("\nCheckpoints:")
            for cp in history:
                print(
                    f"  - {cp.from_agent} → {cp.to_agent}: {cp.status.value} "
                    f"({cp.checkpoint_id[:8]}…)"
                )
            print(
                "\nTIP: offline mode injects a mid-resolver crash to exercise "
                "resume_with_agent. Re-run without --live to see recovery."
            )
            return 0
        print("ERROR: expected resolver to crash offline; it did not.")
        return 1
    except ConnectionError as exc:
        print("\n" + "!" * 60)
        print("CRASH: resolver worker failed after handoff.")
        print(f"  Exception: {exc.__class__.__name__}: {exc}")

        recovered = resume_run(relay, run_id)
        if recovered is None:
            print("ERROR: resume_run() returned None — handoff was not checkpointed.")
            return 1

        print("\nRECOVERY: resume_run() returned last VERIFIED context.")
        print(f"  context: {recovered}")
        print("  Calling resume_with_agent(resolver) only (not full triage)…")
        print("!" * 60 + "\n")

        result = await resume_with_agent(
            relay,
            run_id,
            resolver,
            prompt=lambda ctx: (
                "Finish resolving this ticket.\n"
                f"Context JSON: {json.dumps(ctx)}"
            ),
            mark_recovered=True,
        )
        print(f"[resolver] Resumed output: {result.final_output}")

    print("\n" + "=" * 60)
    print("SUCCESS: relayed_handoff + resume_with_agent completed.")
    for cp in relay.store.history(run_id):
        print(
            f"  - {cp.from_agent} → {cp.to_agent}: {cp.status.value} "
            f"({cp.checkpoint_id[:8]}…)"
        )
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real OpenAI models (requires OPENAI_API_KEY)",
    )
    args = parser.parse_args()
    if args.live and not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: --live requires OPENAI_API_KEY in the environment or .env"
        )
        return 2
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    return asyncio.run(run_pipeline(live=args.live))


if __name__ == "__main__":
    raise SystemExit(main())
