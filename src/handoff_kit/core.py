"""Checkpoint / verify / recover for agent-to-agent handoffs.

Framework-agnostic on purpose. OpenAI wiring lives in ``openai_adapter``.
"""

from __future__ import annotations

import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

from handoff_kit.models import Checkpoint, HandoffStatus, checkpoint_state
from handoff_kit.store import CheckpointStore

logger = logging.getLogger(__name__)


class HandoffVerificationError(Exception):
    """Raised when the handoff is missing keys the next agent needs."""


class Relay:
    """Save a clean handoff snapshot, then resume from it if something blows up.

    Flow:
      1. ``checkpoint`` — write PENDING
      2. ``verify`` — mark VERIFIED *before* the risky step runs
      3. run the receiving agent / wrapped function
      4. on crash, ``recover`` returns that last VERIFIED snapshot

    Why verify before the function runs
    -----------------------------------
    If we only marked VERIFIED after a successful handoff body, the first
    crash inside ``to_agent`` would leave nothing to roll back to. Pre-flight
    verify means: "the receiver was given what it needs." That stays true even
    if its own work later throws — so we never downgrade VERIFIED on a
    downstream failure.
    """

    def __init__(
        self,
        store: Optional[Union[CheckpointStore, str, Path]] = None,
    ) -> None:
        if store is None:
            self.store = CheckpointStore()
        elif isinstance(store, CheckpointStore):
            self.store = store
        else:
            self.store = CheckpointStore(store)

    def checkpoint(
        self,
        run_id: str,
        from_agent: str,
        to_agent: str,
        context: dict[str, Any],
        required_keys: list[str],
    ) -> Checkpoint:
        """Persist a PENDING checkpoint for this handoff."""
        cp = Checkpoint(
            run_id=run_id,
            from_agent=from_agent,
            to_agent=to_agent,
            context=dict(context),
            required_keys=list(required_keys),
            status=HandoffStatus.PENDING,
        )
        self.store.save(cp)
        return cp

    def verify(self, checkpoint: Checkpoint) -> Checkpoint:
        """Pre-flight: required keys must exist on app state before the risky step.

        Envelope checkpoints check ``context["state"]``. Flat dicts are checked as-is.
        """
        state = checkpoint_state(checkpoint)
        missing = [k for k in checkpoint.required_keys if k not in state]
        if missing:
            msg = (
                f"Handoff from {checkpoint.from_agent!r} to {checkpoint.to_agent!r} "
                f"missing required keys: {missing}"
            )
            checkpoint.status = HandoffStatus.FAILED
            checkpoint.error = msg
            self.store.update_status(
                checkpoint.checkpoint_id, HandoffStatus.FAILED, error=msg
            )
            raise HandoffVerificationError(msg)

        checkpoint.status = HandoffStatus.VERIFIED
        checkpoint.error = None
        self.store.update_status(
            checkpoint.checkpoint_id, HandoffStatus.VERIFIED, error=None
        )
        return checkpoint

    def recover(self, run_id: str) -> Optional[Checkpoint]:
        """Last VERIFIED checkpoint for ``run_id``, or None."""
        return self.store.last_good_checkpoint(run_id)

    def guarded_handoff(
        self,
        run_id: str,
        from_agent: str,
        to_agent: str,
        required_keys: list[str],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: checkpoint → verify (pre-flight) → call ``fn(context)``.

        On failure after verify: log and re-raise. Do not touch VERIFIED status —
        that checkpoint is still the rollback point.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def _extract_context(
                args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
                kwargs = dict(kwargs)
                if args:
                    context = args[0]
                    rest_args = args[1:]
                elif "context" in kwargs:
                    context = kwargs.pop("context")
                    rest_args = ()
                else:
                    raise TypeError(
                        f"{fn.__name__} must be called with a context dict "
                        "as the first argument or context= keyword"
                    )
                if not isinstance(context, dict):
                    raise TypeError("context must be a dict")
                return context, rest_args, kwargs

            def _checkpoint_and_verify(context: dict[str, Any]) -> Checkpoint:
                cp = self.checkpoint(
                    run_id=run_id,
                    from_agent=from_agent,
                    to_agent=to_agent,
                    context=context,
                    required_keys=required_keys,
                )
                # VERIFIED before fn runs — a crash inside fn is still recoverable.
                self.verify(cp)
                return cp

            def _log_body_failure(cp: Checkpoint) -> None:
                logger.exception(
                    "Handoff body failed after VERIFIED checkpoint %s "
                    "(run_id=%s, %s -> %s). Status stays VERIFIED; "
                    "call recover(%r) to resume.",
                    cp.checkpoint_id,
                    run_id,
                    from_agent,
                    to_agent,
                    run_id,
                )

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    context, rest_args, kwargs = _extract_context(args, kwargs)
                    cp = _checkpoint_and_verify(context)
                    try:
                        return await fn(context, *rest_args, **kwargs)
                    except Exception:
                        _log_body_failure(cp)
                        raise

                return async_wrapper

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                context, rest_args, kwargs = _extract_context(args, kwargs)
                cp = _checkpoint_and_verify(context)
                try:
                    return fn(context, *rest_args, **kwargs)
                except Exception:
                    _log_body_failure(cp)
                    raise

            return wrapper

        return decorator
