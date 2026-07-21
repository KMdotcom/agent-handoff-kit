#!/usr/bin/env python3
"""Real OpenAI Agents SDK demo with agent-relay recovery.

Pipeline: triage agent → (guarded handoff) → resolver agent (flaky billing tool)
→ closer agent.

By default this uses a *scripted* Model so you can verify recovery without an
API key while still exercising real ``Agent`` / ``Runner`` / ``function_tool``
types from ``openai-agents``.

Live mode (real OpenAI models). Put your key in a repo-root ``.env``
(see ``.env.example``), or export it in the shell::

    # .env at repo root:
    # OPENAI_API_KEY=sk-...
    python3 demo/demo_openai_agents.py --live

Offline / CI mode (default)::

    python3 demo/demo_openai_agents.py
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
    """Minimal .env loader (stdlib-only). Does not override existing env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
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

DB_PATH = _ROOT / "demo" / "demo_openai_agents.db"

# Deterministic flaky tool flag (not randomness).
_billing_failed_once = False
# The Agents SDK catches tool exceptions and feeds them back as
# function_call_output strings — Runner.run does NOT raise. We record the
# hard failure here and re-raise in the handoff body so agent-relay recovery
# is exercised the way a production wrapper would treat a fatal tool outage.
_last_tool_error: Exception | None = None


@function_tool
def billing_lookup(ticket_id: str) -> str:
    """Look up billing details for a support ticket. May time out transiently."""
    global _billing_failed_once, _last_tool_error
    print(f"[tool:billing_lookup] ticket_id={ticket_id}")
    if not _billing_failed_once:
        _billing_failed_once = True
        _last_tool_error = TimeoutError("billing API timed out after 30s")
        raise _last_tool_error
    _last_tool_error = None
    return (
        f"Ticket {ticket_id}: double charge confirmed on invoice #8821; "
        "refund of $42.00 approved."
    )


def _text_message(text: str, msg_id: str | None = None) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=msg_id or f"msg_{uuid.uuid4().hex[:8]}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _input_has_tool_result(input_items: str | list[Any]) -> bool:
    if isinstance(input_items, str):
        return False
    for item in input_items:
        kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if kind in {"function_call_output", "tool_result"}:
            return True
    return False


class ScriptedSupportModel(Model):
    """Minimal Model that drives triage / resolver / closer without a network.

    Resolver: first call → invoke ``billing_lookup``; after tool result → final text.
    """

    def __init__(self, role: str) -> None:
        self.role = role

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        if self.role == "triage":
            payload = {
                "ticket_id": "T-2001",
                "customer": "Ada Lovelace",
                "category": "billing",
                "priority": "high",
                "summary": "Double charge on invoice #8821",
            }
            return ModelResponse(
                output=[_text_message(json.dumps(payload))],
                usage=Usage(),
                response_id=f"resp_{uuid.uuid4().hex[:8]}",
            )

        if self.role == "resolver":
            if not _input_has_tool_result(input):
                # Extract ticket_id from the handoff context string if present.
                ticket_id = "T-2001"
                raw = input if isinstance(input, str) else str(input)
                if "T-2001" in raw:
                    ticket_id = "T-2001"
                call = ResponseFunctionToolCall(
                    type="function_call",
                    call_id=f"call_{uuid.uuid4().hex[:8]}",
                    name="billing_lookup",
                    arguments=json.dumps({"ticket_id": ticket_id}),
                )
                return ModelResponse(
                    output=[call],
                    usage=Usage(),
                    response_id=f"resp_{uuid.uuid4().hex[:8]}",
                )
            return ModelResponse(
                output=[
                    _text_message(
                        "Refund queued for $42.00 after billing_lookup confirmed "
                        "the double charge."
                    )
                ],
                usage=Usage(),
                response_id=f"resp_{uuid.uuid4().hex[:8]}",
            )

        # closer
        return ModelResponse(
            output=[
                _text_message(
                    "Ticket closed. Customer notified that refund of $42.00 is on the way."
                )
            ],
            usage=Usage(),
            response_id=f"resp_{uuid.uuid4().hex[:8]}",
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            if False:  # pragma: no cover — satisfy async generator typing
                yield None
            raise NotImplementedError("ScriptedSupportModel does not stream")

        return _gen()


def _parse_triage_output(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Live models may wrap JSON in prose — pull the first {...} block.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError(f"Could not parse triage JSON from: {text!r}")


def build_agents(*, live: bool) -> tuple[Agent, Agent, Agent]:
    if live:
        triage = Agent(
            name="Triage",
            instructions=(
                "You are a support triage agent. Classify the user ticket and "
                "reply with ONLY a JSON object with keys: ticket_id, customer, "
                "category, priority, summary. Invent a ticket_id like T-2001."
            ),
            model="gpt-4.1-mini",
        )
        resolver = Agent(
            name="Resolver",
            instructions=(
                "You resolve billing tickets. Always call billing_lookup with the "
                "ticket_id from the context, then summarize the resolution in one "
                "short paragraph."
            ),
            tools=[billing_lookup],
            model="gpt-4.1-mini",
        )
        closer = Agent(
            name="Closer",
            instructions=(
                "You close support tickets. Confirm the resolution in one sentence "
                "and state that the ticket is closed."
            ),
            model="gpt-4.1-mini",
        )
        return triage, resolver, closer

    triage = Agent(
        name="Triage",
        instructions="Classify tickets.",
        model=ScriptedSupportModel("triage"),
    )
    resolver = Agent(
        name="Resolver",
        instructions="Resolve via billing_lookup.",
        tools=[billing_lookup],
        model=ScriptedSupportModel("resolver"),
    )
    closer = Agent(
        name="Closer",
        instructions="Close tickets.",
        model=ScriptedSupportModel("closer"),
    )
    return triage, resolver, closer


async def run_pipeline(*, live: bool) -> int:
    print("=" * 60)
    mode = "LIVE (OpenAI API)" if live else "OFFLINE (scripted Model)"
    print(f"DEMO: OpenAI Agents SDK + agent-relay [{mode}]")
    print("=" * 60)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[setup] Removed existing {DB_PATH.name}")

    global _billing_failed_once
    _billing_failed_once = False
    global _last_tool_error
    _last_tool_error = None

    relay = Relay(DB_PATH)
    run_id = "openai-demo-T-2001"
    triage, resolver, closer = build_agents(live=live)

    print("\n[triage] Running triage agent…")
    triage_result = await Runner.run(
        triage, "I was double-charged on invoice #8821. Please help."
    )
    triage_text = str(triage_result.final_output)
    context = _parse_triage_output(triage_text)
    # Ensure keys the resolver handoff requires.
    context.setdefault("ticket_id", "T-2001")
    context.setdefault("category", "billing")
    context.setdefault("summary", triage_text)
    print(f"[triage] Context: {context}")

    @relay.guarded_handoff(
        run_id=run_id,
        from_agent="triage",
        to_agent="resolver",
        required_keys=["ticket_id", "category", "summary"],
    )
    async def handoff_to_resolver(ctx: dict) -> dict:
        print("\n[resolver] Running resolver agent (may hit flaky billing tool)…")
        prompt = (
            "Resolve this support ticket using billing_lookup.\n"
            f"Context JSON: {json.dumps(ctx)}"
        )
        global _last_tool_error
        _last_tool_error = None
        result = await Runner.run(resolver, prompt)
        # Promote fatal tool outages out of the SDK's soft-error path so the
        # guarded handoff can recover from a VERIFIED checkpoint.
        if _last_tool_error is not None:
            err = _last_tool_error
            _last_tool_error = None
            raise err
        out = dict(ctx)
        out["resolution"] = str(result.final_output)
        return out

    @relay.guarded_handoff(
        run_id=run_id,
        from_agent="resolver",
        to_agent="closer",
        required_keys=["ticket_id", "resolution"],
    )
    async def handoff_to_closer(ctx: dict) -> dict:
        print("\n[closer] Running closer agent…")
        prompt = (
            "Close this ticket.\n"
            f"Context JSON: {json.dumps(ctx)}"
        )
        result = await Runner.run(closer, prompt)
        out = dict(ctx)
        out["status"] = "closed"
        out["close_note"] = str(result.final_output)
        return out

    print("\n→ Handing off triage → resolver (checkpointed)…")
    try:
        context = await handoff_to_resolver(context)
    except Exception as exc:
        print("\n" + "!" * 60)
        print("CRASH: resolver / billing tool failed (expected once).")
        print(f"  Exception: {exc.__class__.__name__}: {exc}")

        recovered = relay.recover(run_id)
        if recovered is None:
            print("ERROR: recover() returned None — verification timing is wrong.")
            return 1

        print("\nRECOVERY: last VERIFIED checkpoint found.")
        print(f"  checkpoint_id: {recovered.checkpoint_id}")
        print(f"  from → to:     {recovered.from_agent} → {recovered.to_agent}")
        print(f"  status:        {recovered.status.value}")
        print(f"  context:       {recovered.context}")
        print("  Retrying ONLY the resolver handoff…")
        print("!" * 60)

        context = await handoff_to_resolver(recovered.context)

    print("\n→ Handing off resolver → closer (checkpointed)…")
    context = await handoff_to_closer(context)

    print("\n" + "=" * 60)
    print("SUCCESS: real Agents SDK pipeline completed with agent-relay.")
    print(f"  Final context: {context}")
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
            "ERROR: --live requires OPENAI_API_KEY.\n"
            "  Add it to a .env file at the repo root (see .env.example),\n"
            "  or: export OPENAI_API_KEY=sk-..."
        )
        return 2

    # Quiet noisy SDK tracing in demos unless the user opted in.
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

    return asyncio.run(run_pipeline(live=args.live))


if __name__ == "__main__":
    raise SystemExit(main())
