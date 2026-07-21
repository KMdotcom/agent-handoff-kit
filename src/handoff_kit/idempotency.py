"""Cache tool results so a resumed agent does not double-apply side effects."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from handoff_kit.core import Relay
from handoff_kit.store import CheckpointStore

logger = logging.getLogger(__name__)

KeyFn = Callable[..., str]


def _resolve_store(relay_or_store: Relay | CheckpointStore) -> CheckpointStore:
    if isinstance(relay_or_store, CheckpointStore):
        return relay_or_store
    return relay_or_store.store


def idempotent_tool(
    relay_or_store: Relay | CheckpointStore,
    run_id: str,
    key_fn: KeyFn,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: cache tool results by ``(run_id, key_fn(...))`` in SQLite.

    On a later resume with the same key, returns the cached result without
    re-executing the wrapped function.
    """
    store = _resolve_store(relay_or_store)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = getattr(fn, "__name__", "tool")

        def _key_and_lookup(*args: Any, **kwargs: Any) -> tuple[str, Any | None]:
            key = key_fn(*args, **kwargs)
            if not isinstance(key, str) or not key:
                raise TypeError("key_fn must return a non-empty str")
            return key, store.get_tool_result(run_id, key)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                key, cached = _key_and_lookup(*args, **kwargs)
                if cached is not None:
                    logger.info("Idempotent hit for %s key=%s run_id=%s", tool_name, key, run_id)
                    return cached
                result = await fn(*args, **kwargs)
                store.save_tool_result(run_id, key, tool_name, result)
                return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key, cached = _key_and_lookup(*args, **kwargs)
            if cached is not None:
                logger.info("Idempotent hit for %s key=%s run_id=%s", tool_name, key, run_id)
                return cached
            result = fn(*args, **kwargs)
            store.save_tool_result(run_id, key, tool_name, result)
            return result

        return wrapper

    return decorator


def make_idempotent_function_tool(
    relay_or_store: Relay | CheckpointStore,
    run_id: str,
    fn: Callable[..., Any],
    key_fn: KeyFn,
    **function_tool_kwargs: Any,
) -> Any:
    """Wrap ``fn`` with idempotency, then ``agents.function_tool``.

    Requires ``openai-agents``. Extra kwargs are forwarded to ``function_tool``.
    """
    try:
        from agents import function_tool
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "make_idempotent_function_tool requires openai-agents. "
            "Install with: pip install 'handoff-kit[openai]'"
        ) from exc

    wrapped = idempotent_tool(relay_or_store, run_id, key_fn)(fn)
    return function_tool(wrapped, **function_tool_kwargs)
