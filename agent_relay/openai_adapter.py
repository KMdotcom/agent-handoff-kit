"""OpenAI Agents SDK integration seam.

IMPORTANT: The OpenAI Agents SDK API surface moves quickly. This module is the
*only* place that should be adjusted when pinning a new SDK version.
``agent_relay.core.Relay`` must stay framework-agnostic and should never need
to change because of an Agents SDK release.

Drop-in usage (native ``handoff``)::

    from agent_relay import Relay
    from agent_relay.openai_adapter import relayed_handoff, resume_run, resume_with_agent

    relay = Relay("support.db")
    run_id = "ticket-42"

    # Pass a dict as Runner.run(..., context=context). On transfer, we
    # checkpoint + verify that dict (pre-flight) before the target agent runs.
    to_resolver = relayed_handoff(
        relay,
        run_id=run_id,
        from_agent="triage",
        to_agent="resolver",
        agent=resolver_agent,
        required_keys=["ticket_id", "summary"],
    )
    triage = Agent(name="Triage", handoffs=[to_resolver], ...)

    await Runner.run(triage, user_msg, context={"ticket_id": "T-1", "summary": "..."})

After a crash::

    ctx = resume_run(relay, run_id)          # last VERIFIED context dict
    await resume_with_agent(resolver_agent, ctx)  # continue receiving agent only
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional, Union

from agent_relay.core import Relay
from agent_relay.models import HandoffStatus

logger = logging.getLogger(__name__)

ExtractContext = Callable[..., dict[str, Any]]
OnHandoff = Callable[..., Any]


def context_from_run_context(ctx: Any, input_data: Any = None) -> dict[str, Any]:
    """Default extractor: prefer ``ctx.context`` when it is a mapping.

    Also merges ``input_data`` (parsed handoff tool args) when it is a mapping
    or has ``model_dump()``. Override via ``extract_context=`` when your app
    keeps state elsewhere.
    """
    out: dict[str, Any] = {}
    inner = getattr(ctx, "context", None)
    if isinstance(inner, dict):
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


def wrap_on_handoff(
    relay: Relay,
    run_id: str,
    from_agent: str,
    to_agent: str,
    required_keys: list[str],
    on_handoff: Optional[OnHandoff] = None,
    extract_context: Optional[ExtractContext] = None,
) -> Callable[..., Any]:
    """Wrap an ``on_handoff`` callback with checkpoint + pre-flight verify.

    The returned callable matches the SDK contract:
      - ``(ctx)`` when there is no ``input_type``
      - ``(ctx, input_data)`` when ``input_type`` is set

    Return values of ``on_handoff`` are ignored by the SDK (side effects only);
    we still run your callback after a successful verify.
    """
    extract = extract_context or context_from_run_context

    async def _checkpoint_then_user(ctx: Any, input_data: Any = None) -> Any:
        # Build the dict we persist. Keep this mapping in the adapter only.
        if extract is context_from_run_context:
            context = context_from_run_context(ctx, input_data)
        else:
            # Support both (ctx) and (ctx, input_data) extractors.
            try:
                context = extract(ctx, input_data)
            except TypeError:
                context = extract(ctx)
        if not isinstance(context, dict):
            raise TypeError("extract_context must return a dict")

        cp = relay.checkpoint(
            run_id=run_id,
            from_agent=from_agent,
            to_agent=to_agent,
            context=context,
            required_keys=required_keys,
        )
        # VERIFIED before the receiving agent runs — same invariant as core.
        relay.verify(cp)

        if on_handoff is None:
            return None

        # SDK allows sync or async on_handoff; mirror that here.
        sig = inspect.signature(on_handoff)
        if len(sig.parameters) >= 2:
            result = on_handoff(ctx, input_data)
        else:
            result = on_handoff(ctx)
        if inspect.isawaitable(result):
            return await result
        return result

    # Expose a sync-looking callable whose signature length the SDK inspects
    # when building handoff(). We provide both arities via a thin shim when
    # the user later passes input_type=... — see relayed_handoff().
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
    """Return an ``agents.handoff(...)`` that checkpoints at the transfer boundary.

    This is the copy-paste integration point. Requires ``openai-agents`` installed.
    Extra ``handoff_kwargs`` are forwarded to ``agents.handoff`` (e.g.
    ``tool_name_override``, ``input_filter``, ``is_enabled``).
    """
    try:
        from agents import handoff as openai_handoff
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "relayed_handoff requires the openai-agents package. "
            "Install with: pip install 'openai-agents'"
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

    # agents.handoff validates on_handoff arity against input_type.
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
    """Backward-compatible helper: wrap a user ``on_handoff`` for ``handoff()``.

    Prefer ``relayed_handoff`` for new code — it returns a ready-made Handoff.
    """
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
    """Return context from the last VERIFIED checkpoint, or None.

    By default does **not** change status, so you can call ``resume_with_agent``
    (which also reads the VERIFIED checkpoint) afterward. Pass
    ``mark_recovered=True`` once resume has succeeded if you want an audit mark.
    """
    cp = relay.recover(run_id)
    if cp is None:
        return None
    if mark_recovered:
        relay.store.update_status(cp.checkpoint_id, HandoffStatus.RECOVERED)
        cp.status = HandoffStatus.RECOVERED
    return dict(cp.context)


async def resume_with_agent(
    relay: Relay,
    run_id: str,
    agent: Any,
    *,
    prompt: Optional[Union[str, Callable[[dict[str, Any]], str]]] = None,
    mark_recovered: bool = True,
    **runner_kwargs: Any,
) -> Any:
    """Recover the last VERIFIED context and run ``agent`` only (not the whole graph).

    Default prompt embeds the recovered context JSON. Override with a string or
    ``prompt(context) -> str``. Forwards extra kwargs to ``Runner.run``
    (e.g. ``max_turns``).
    """
    try:
        from agents import Runner
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "resume_with_agent requires the openai-agents package. "
            "Install with: pip install 'openai-agents'"
        ) from exc

    import json

    cp = relay.recover(run_id)
    if cp is None:
        raise RuntimeError(
            f"No VERIFIED checkpoint for run_id={run_id!r}; cannot resume."
        )
    context = dict(cp.context)

    if prompt is None:
        user_input = (
            "Continue from the recovered handoff checkpoint. "
            f"Context JSON: {json.dumps(context)}"
        )
    elif callable(prompt):
        user_input = prompt(context)
    else:
        user_input = prompt

    logger.info(
        "Resuming agent %r for run_id=%s with recovered context keys=%s",
        getattr(agent, "name", agent),
        run_id,
        sorted(context.keys()),
    )
    result = await Runner.run(agent, user_input, context=context, **runner_kwargs)
    if mark_recovered:
        relay.store.update_status(cp.checkpoint_id, HandoffStatus.RECOVERED)
    return result
