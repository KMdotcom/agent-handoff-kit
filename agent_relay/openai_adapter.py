"""OpenAI Agents SDK integration seam.

IMPORTANT: The OpenAI Agents SDK API surface moves quickly. This module is the
*only* place that should be adjusted when pinning a new SDK version.
``agent_relay.core.Relay`` must stay framework-agnostic and should never need
to change because of an Agents SDK release.

Typical usage with ``handoff(..., on_handoff=...)``::

    from agent_relay import Relay
    from agent_relay.openai_adapter import wrap_handoff, resume_run

    relay = Relay("support.db")
    run_id = "ticket-42"

    def on_handoff(ctx, input_data=None):
        # Build / return the context dict you want checkpointed.
        return {"ticket_id": "...", "summary": "...", ...}

    guarded = wrap_handoff(
        relay,
        run_id=run_id,
        from_agent="triage",
        to_agent="resolver",
        on_handoff=on_handoff,
        required_keys=["ticket_id", "summary"],
    )
    # Pass ``guarded`` as the ``on_handoff`` callback to the SDK's handoff().

After a crash::

    context = resume_run(relay, run_id)
    # Re-invoke the receiving agent with ``context`` instead of restarting.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent_relay.core import Relay


def wrap_handoff(
    relay: Relay,
    run_id: str,
    from_agent: str,
    to_agent: str,
    on_handoff: Callable[..., Any],
    required_keys: list[str],
) -> Callable[..., Any]:
    """Wrap an OpenAI Agents ``on_handoff`` callback with checkpoint + verify.

    Uses ``Relay.guarded_handoff`` under the hood. The wrapped callback is
    expected to accept a context-like first argument (or ``context=``) that is
    a ``dict``. Adapt SDK-specific ``RunContextWrapper`` objects at this seam
    if needed for your pinned SDK version — keep that mapping here, not in core.
    """
    return relay.guarded_handoff(
        run_id=run_id,
        from_agent=from_agent,
        to_agent=to_agent,
        required_keys=required_keys,
    )(on_handoff)


def resume_run(relay: Relay, run_id: str) -> Optional[dict[str, Any]]:
    """Return context from the last VERIFIED checkpoint, or None if none exists."""
    cp = relay.recover(run_id)
    if cp is None:
        return None
    return dict(cp.context)
