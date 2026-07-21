"""OpenAI Agents SDK adapter for handoff-kit.

When the Agents SDK API shifts, change this file — not ``handoff_kit.core``.

Quick use::

    from handoff_kit import Relay
    from handoff_kit.openai_adapter import DurableRunner, relayed_handoff

    relay = Relay("support.db")
    to_resolver = relayed_handoff(
        relay, "ticket-42", "triage", "resolver", resolver,
        required_keys=["ticket_id", "summary"],
    )
    triage = Agent(name="Triage", handoffs=[to_resolver], ...)
    await DurableRunner.run(
        relay, "ticket-42", triage, user_msg,
        context={"ticket_id": "T-1", "summary": "..."},
        agents={"triage": triage, "resolver": resolver},
    )
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Optional, Union

from handoff_kit.core import Relay
from handoff_kit.models import (
    HandoffStatus,
    Checkpoint,
    checkpoint_messages,
    checkpoint_state,
    make_handoff_envelope,
)

logger = logging.getLogger(__name__)

ExtractContext = Callable[..., dict[str, Any]]
OnHandoff = Callable[..., Any]


class RunAlreadyCompleted(RuntimeError):
    """Raised when DurableRunner is asked to run a completed run_id again."""


def _json_sanitize(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_sanitize(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _json_sanitize(value.dict())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def context_from_run_context(ctx: Any, input_data: Any = None) -> dict[str, Any]:
    """Extract the app **state** dict from ``RunContextWrapper`` (+ handoff input).

    Does not build the full envelope — see ``build_envelope_from_run_context``.
    """
    out: dict[str, Any] = {}
    inner = getattr(ctx, "context", None)
    if isinstance(inner, dict):
        # If caller already stored an envelope in context, use its state.
        if "state" in inner and "meta" in inner and isinstance(inner.get("state"), dict):
            out.update(inner["state"])
        else:
            out.update(inner)
    elif inner is not None and hasattr(inner, "model_dump"):
        dumped = inner.model_dump()
        if isinstance(dumped, dict):
            out.update(dumped)

    if isinstance(input_data, dict):
        out.update(input_data)
    elif input_data is not None and hasattr(input_data, "model_dump"):
        dumped = input_data.model_dump()
        if isinstance(dumped, dict):
            out.update(dumped)
    return out


def messages_from_run_context(ctx: Any) -> list[Any]:
    """Best-effort message/turn capture from the SDK run context."""
    turn_input = getattr(ctx, "turn_input", None) or []
    if not isinstance(turn_input, list):
        return []
    return [_json_sanitize(item) for item in turn_input]


def build_envelope_from_run_context(
    ctx: Any,
    *,
    from_agent: str,
    to_agent: str,
    input_data: Any = None,
    extract_context: Optional[ExtractContext] = None,
) -> dict[str, Any]:
    """Build ``{state, messages, meta}`` for a handoff checkpoint."""
    extract = extract_context or context_from_run_context
    if extract is context_from_run_context:
        state = context_from_run_context(ctx, input_data)
    else:
        try:
            state = extract(ctx, input_data)
        except TypeError:
            state = extract(ctx)
    if not isinstance(state, dict):
        raise TypeError("extract_context must return a dict")
    return make_handoff_envelope(
        state,
        from_agent=from_agent,
        to_agent=to_agent,
        messages=messages_from_run_context(ctx),
    )


def wrap_on_handoff(
    relay: Relay,
    run_id: str,
    from_agent: str,
    to_agent: str,
    required_keys: list[str],
    on_handoff: Optional[OnHandoff] = None,
    extract_context: Optional[ExtractContext] = None,
) -> Callable[..., Any]:
    """Wrap an ``on_handoff`` callback with envelope checkpoint + pre-flight verify."""

    async def _checkpoint_then_user(ctx: Any, input_data: Any = None) -> Any:
        envelope = build_envelope_from_run_context(
            ctx,
            from_agent=from_agent,
            to_agent=to_agent,
            input_data=input_data,
            extract_context=extract_context,
        )
        cp = relay.checkpoint(
            run_id=run_id,
            from_agent=from_agent,
            to_agent=to_agent,
            context=envelope,
            required_keys=required_keys,
        )
        # VERIFIED before the receiving agent runs — same invariant as core.
        relay.verify(cp)
        relay.store.set_run_status(run_id, "open")

        if on_handoff is None:
            return None

        sig = inspect.signature(on_handoff)
        if len(sig.parameters) >= 2:
            result = on_handoff(ctx, input_data)
        else:
            result = on_handoff(ctx)
        if inspect.isawaitable(result):
            return await result
        return result

    return _checkpoint_then_user


def relayed_handoff(
    relay: Relay,
    run_id: str,
    from_agent: str,
    to_agent: str,
    agent: Any,
    required_keys: list[str],
    *,
    on_handoff: Optional[OnHandoff] = None,
    extract_context: Optional[ExtractContext] = None,
    input_type: Any = None,
    **handoff_kwargs: Any,
) -> Any:
    """Return an ``agents.handoff(...)`` that checkpoints at the transfer boundary."""
    try:
        from agents import handoff as openai_handoff
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "relayed_handoff requires openai-agents. "
            "Install with: pip install 'handoff-kit[openai]'"
        ) from exc

    wrapped = wrap_on_handoff(
        relay,
        run_id=run_id,
        from_agent=from_agent,
        to_agent=to_agent,
        required_keys=required_keys,
        on_handoff=on_handoff,
        extract_context=extract_context,
    )

    if input_type is not None:

        async def _on_with_input(ctx: Any, input_data: Any) -> Any:
            return await wrapped(ctx, input_data)

        return openai_handoff(
            agent,
            on_handoff=_on_with_input,
            input_type=input_type,
            **handoff_kwargs,
        )

    async def _on_without_input(ctx: Any) -> Any:
        return await wrapped(ctx, None)

    return openai_handoff(
        agent,
        on_handoff=_on_without_input,
        **handoff_kwargs,
    )


def wrap_handoff(
    relay: Relay,
    run_id: str,
    from_agent: str,
    to_agent: str,
    on_handoff: OnHandoff,
    required_keys: list[str],
    extract_context: Optional[ExtractContext] = None,
) -> Callable[..., Any]:
    """Backward-compatible helper: wrap a user ``on_handoff`` for ``handoff()``."""
    return wrap_on_handoff(
        relay,
        run_id=run_id,
        from_agent=from_agent,
        to_agent=to_agent,
        required_keys=required_keys,
        on_handoff=on_handoff,
        extract_context=extract_context,
    )


def resume_run(
    relay: Relay,
    run_id: str,
    *,
    mark_recovered: bool = False,
) -> Optional[dict[str, Any]]:
    """Return the last VERIFIED checkpoint payload (envelope or legacy flat dict)."""
    cp = relay.recover(run_id)
    if cp is None:
        return None
    if mark_recovered:
        relay.store.update_status(cp.checkpoint_id, HandoffStatus.RECOVERED)
        cp.status = HandoffStatus.RECOVERED
    return dict(cp.context)


def _default_resume_prompt(checkpoint: Checkpoint) -> str:
    state = checkpoint_state(checkpoint)
    messages = checkpoint_messages(checkpoint)
    parts = [
        "Continue from the recovered handoff checkpoint.",
        f"Context JSON: {json.dumps(state)}",
    ]
    if messages:
        # Keep prompt bounded — full list is also in the checkpoint store.
        summary = json.dumps(messages[:20])
        parts.append(f"Recent messages (truncated): {summary}")
    return "\n".join(parts)


async def resume_with_agent(
    relay: Relay,
    run_id: str,
    agent: Any,
    *,
    prompt: Optional[Union[str, Callable[[dict[str, Any]], str]]] = None,
    mark_recovered: bool = True,
    **runner_kwargs: Any,
) -> Any:
    """Recover the last VERIFIED checkpoint and run ``agent`` only."""
    try:
        from agents import Runner
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "resume_with_agent requires openai-agents. "
            "Install with: pip install 'handoff-kit[openai]'"
        ) from exc

    cp = relay.recover(run_id)
    if cp is None:
        raise RuntimeError(
            f"No VERIFIED checkpoint for run_id={run_id!r}; cannot resume."
        )
    state = checkpoint_state(cp)

    if prompt is None:
        user_input = _default_resume_prompt(cp)
    elif callable(prompt):
        # Callables receive the app state dict (not the raw envelope).
        user_input = prompt(state)
    else:
        user_input = prompt

    logger.info(
        "Resuming agent %r for run_id=%s with state keys=%s",
        getattr(agent, "name", agent),
        run_id,
        sorted(state.keys()),
    )
    result = await Runner.run(agent, user_input, context=state, **runner_kwargs)
    if mark_recovered:
        relay.store.update_status(cp.checkpoint_id, HandoffStatus.RECOVERED)
    return result


def _resolve_agent(
    agents: dict[str, Any],
    to_agent: str,
) -> Optional[Any]:
    if to_agent in agents:
        return agents[to_agent]
    lowered = {k.lower(): v for k, v in agents.items()}
    return lowered.get(to_agent.lower())


class DurableRunner:
    """Automatic recovery wrapper around ``Runner.run``.

    - If ``run_id`` is already ``completed``, raises ``RunAlreadyCompleted``.
    - If a VERIFIED checkpoint exists for an incomplete run, resumes the
      destination agent directly (process restart / cold start).
    - On in-process failure after a VERIFIED handoff, resumes the destination
      agent once, then re-raises if that also fails.
    - Marks the run ``completed`` on success.
    """

    @staticmethod
    async def run(
        relay: Relay,
        run_id: str,
        starting_agent: Any,
        user_input: Any,
        *,
        context: Optional[dict[str, Any]] = None,
        agents: Optional[dict[str, Any]] = None,
        max_resume_attempts: int = 1,
        **runner_kwargs: Any,
    ) -> Any:
        try:
            from agents import Runner
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DurableRunner requires openai-agents. "
                "Install with: pip install 'handoff-kit[openai]'"
            ) from exc

        agents = agents or {}
        status = relay.store.get_run_status(run_id)
        if status == "completed":
            raise RunAlreadyCompleted(
                f"run_id={run_id!r} is already completed; refusing to re-run."
            )

        cp = relay.recover(run_id)
        if cp is not None:
            dest = _resolve_agent(agents, cp.to_agent)
            if dest is not None:
                logger.info(
                    "DurableRunner cold-resume run_id=%s -> %s",
                    run_id,
                    cp.to_agent,
                )
                result = await resume_with_agent(
                    relay, run_id, dest, mark_recovered=True, **runner_kwargs
                )
                relay.store.set_run_status(run_id, "completed")
                return result

        relay.store.set_run_status(run_id, "open")
        try:
            result = await Runner.run(
                starting_agent,
                user_input,
                context=context,
                **runner_kwargs,
            )
        except Exception as exc:
            cp = relay.recover(run_id)
            if cp is None or max_resume_attempts < 1:
                raise
            dest = _resolve_agent(agents, cp.to_agent)
            if dest is None:
                raise
            logger.warning(
                "DurableRunner caught failure after VERIFIED handoff "
                "run_id=%s to_agent=%s (%s: %s); auto-resuming once",
                run_id,
                cp.to_agent,
                type(exc).__name__,
                exc,
            )
            try:
                result = await resume_with_agent(
                    relay, run_id, dest, mark_recovered=True, **runner_kwargs
                )
            except Exception:
                raise
            relay.store.set_run_status(run_id, "completed")
            return result

        relay.store.set_run_status(run_id, "completed")
        return result
