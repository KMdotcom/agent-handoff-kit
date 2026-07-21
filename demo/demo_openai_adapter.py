#!/usr/bin/env python3
"""DurableRunner demo: full-context handoff, auto-resume, idempotent refund tool.

Offline by default (scripted Model). Live::

    python3 demo/demo_openai_adapter.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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
    from agents import Agent, ModelResponse, Usage
    from agents.models.interface import Model
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
except ImportError as exc:
    print(
        "Missing dependency: openai-agents (or an import it needs failed).\n"
        f"  python: {sys.executable}\n"
        f"  error:  {exc}\n"
        "  Fix with the SAME interpreter:\n"
        f"    {sys.executable} -m pip install 'openai-agents'\n"
        "  Then re-run:\n"
        f"    {sys.executable} demo/demo_openai_adapter.py"
    )
    raise SystemExit(2) from None

from agent_handoff_kit import Relay
from agent_handoff_kit.idempotency import make_idempotent_function_tool
from agent_handoff_kit.models import checkpoint_messages, checkpoint_state
from agent_handoff_kit.openai_adapter import (
    DurableRunner,
    RunAlreadyCompleted,
    relayed_handoff,
)

DB_PATH = _ROOT / "demo" / "demo_openai_adapter.db"
_resolver_crashes_left = 1
_refund_calls = 0


def _text_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class ScriptedTriageModel(Model):
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
    """Crash once, then call issue_refund and finish."""

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        **kwargs: Any,
    ) -> ModelResponse:
        global _resolver_crashes_left
        if _resolver_crashes_left > 0:
            _resolver_crashes_left -= 1
            raise ConnectionError("simulated worker crash mid-resolver")

        # After resume: invoke refund tool once, then (on next turn) finalize.
        has_tool_result = False
        if not isinstance(input, str):
            for item in input:
                kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if kind in {"function_call_output", "tool_result"}:
                    has_tool_result = True
                    break
        if not has_tool_result:
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id=f"call_{uuid.uuid4().hex[:8]}",
                        name="issue_refund",
                        arguments='{"ticket_id":"T-2001","amount":42.0}',
                    )
                ],
                usage=Usage(),
                response_id=f"resp_{uuid.uuid4().hex[:8]}",
            )
        return ModelResponse(
            output=[
                _text_message("Refund of $42.00 approved for the double charge on invoice #8821.")
            ],
            usage=Usage(),
            response_id=f"resp_{uuid.uuid4().hex[:8]}",
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            raise NotImplementedError
            yield None  # pragma: no cover

        return _gen()


def build_agents(*, live: bool, relay: Relay, run_id: str) -> tuple[Agent, Agent, Any]:
    global _refund_calls

    def issue_refund(ticket_id: str, amount: float = 42.0) -> str:
        global _refund_calls
        _refund_calls += 1
        print(f"[tool:issue_refund] call=#{_refund_calls} ticket={ticket_id} amount={amount}")
        return f"Refund of ${amount:.2f} issued for {ticket_id}."

    refund_tool = make_idempotent_function_tool(
        relay,
        run_id,
        issue_refund,
        key_fn=lambda ticket_id, amount=42.0: f"refund:{ticket_id}",
    )

    if live:
        resolver = Agent(
            name="Resolver",
            instructions=(
                "Resolve billing tickets. Call issue_refund with the ticket_id, "
                "then confirm in one short paragraph."
            ),
            tools=[refund_tool],
            model="gpt-4.1-mini",
        )
        triage = Agent(
            name="Triage",
            instructions=(
                "Always hand off to the Resolver agent immediately using the "
                "transfer tool. Do not solve the ticket yourself."
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
        return triage, resolver, refund_tool

    resolver = Agent(
        name="Resolver",
        instructions="Resolve via issue_refund.",
        tools=[refund_tool],
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
    return triage, resolver, refund_tool


async def run_pipeline(*, live: bool) -> int:
    print("=" * 60)
    mode = "LIVE (OpenAI API)" if live else "OFFLINE (scripted Model)"
    print(f"DEMO: DurableRunner + full context + idempotency [{mode}]")
    print("=" * 60)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[setup] Removed existing {DB_PATH.name}")

    global _resolver_crashes_left, _refund_calls
    _resolver_crashes_left = 0 if live else 1
    _refund_calls = 0

    relay = Relay(DB_PATH)
    run_id = "adapter-demo-T-2001"
    triage, resolver, _refund_tool = build_agents(live=live, relay=relay, run_id=run_id)
    agents = {"triage": triage, "resolver": resolver}

    context: dict[str, Any] = {
        "ticket_id": "T-2001",
        "category": "billing",
        "summary": "Double charge on invoice #8821",
        "customer": "Ada Lovelace",
    }
    print(f"[app] Starting context: {context}")

    print("\n→ DurableRunner.run(triage) …")
    result = await DurableRunner.run(
        relay,
        run_id,
        triage,
        "I was double-charged on invoice #8821.",
        context=context,
        agents=agents,
    )
    print(f"[done] Final output: {result.final_output}")

    history = relay.store.history(run_id)
    print("\nCheckpoints:")
    for cp in history:
        state = checkpoint_state(cp)
        msgs = checkpoint_messages(cp)
        print(
            f"  - {cp.from_agent} → {cp.to_agent}: {cp.status.value} "
            f"({cp.checkpoint_id[:8]}…) state_keys={sorted(state)} "
            f"messages={len(msgs)}"
        )

    print(f"\n[idempotency] issue_refund raw executions: {_refund_calls}")
    if not live and _refund_calls != 1:
        print("ERROR: expected exactly one refund execution after auto-resume.")
        return 1

    # Re-run same run_id should refuse (completed marker).
    try:
        await DurableRunner.run(
            relay,
            run_id,
            triage,
            "retry?",
            context=context,
            agents=agents,
        )
        print("ERROR: expected RunAlreadyCompleted on second DurableRunner.run")
        return 1
    except RunAlreadyCompleted as exc:
        print(f"[guard] Second run blocked: {exc}")

    print("\n" + "=" * 60)
    print("SUCCESS: DurableRunner auto-recovered; run marked completed.")
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
        print("ERROR: --live requires OPENAI_API_KEY in the environment or .env")
        return 2
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    return asyncio.run(run_pipeline(live=args.live))


if __name__ == "__main__":
    raise SystemExit(main())
