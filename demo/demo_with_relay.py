#!/usr/bin/env python3
"""Demo: same 3-agent support pipeline WITH agent-relay.

The resolver fails exactly once (deterministic flag), then succeeds on retry.
After the crash we recover the last VERIFIED checkpoint and retry ONLY the
resolver step — triage does not re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python demo/demo_with_relay.py` with zero install steps.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_relay import Relay

DB_PATH = _ROOT / "demo" / "demo_with_relay.db"


def triage_agent(ticket: str) -> dict:
    print("[triage] Classifying ticket…")
    context = {
        "ticket_id": "T-1001",
        "customer": "Ada Lovelace",
        "category": "billing",
        "priority": "high",
        "summary": "Double charge on invoice #8821",
        "raw_ticket": ticket,
    }
    print(f"[triage] Done. Context: {context}")
    return context


def resolver_agent(context: dict) -> dict:
    """Fails on the first call only (function attribute flag — not random)."""
    print("[resolver] Attempting to resolve ticket…")
    if not getattr(resolver_agent, "_has_failed_once", False):
        resolver_agent._has_failed_once = True
        raise TimeoutError("billing API timed out after 30s")

    context = dict(context)
    context["resolution_notes"] = "Verified double charge; refund queued"
    context["refund_amount"] = 42.00
    print(f"[resolver] Done. Context: {context}")
    return context


def closer_agent(context: dict) -> dict:
    print("[closer] Closing ticket…")
    context = dict(context)
    context["status"] = "closed"
    context["resolution"] = "Refund issued"
    print(f"[closer] Done. Final context: {context}")
    return context


def main() -> int:
    print("=" * 60)
    print("DEMO: support pipeline WITH agent-relay")
    print("=" * 60)

    # Repeatable: wipe any previous demo DB.
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[setup] Removed existing {DB_PATH.name}")

    relay = Relay(DB_PATH)
    run_id = "demo-ticket-T-1001"
    ticket = "I was double-charged on invoice #8821"

    # Reset failure flag for a clean run.
    resolver_agent._has_failed_once = False

    # Guard the triage → resolver handoff. required_keys are what resolver needs.
    guarded_resolve = relay.guarded_handoff(
        run_id=run_id,
        from_agent="triage",
        to_agent="resolver",
        required_keys=["ticket_id", "customer", "category", "summary"],
    )(resolver_agent)

    # Guard the resolver → closer handoff.
    guarded_close = relay.guarded_handoff(
        run_id=run_id,
        from_agent="resolver",
        to_agent="closer",
        required_keys=["ticket_id", "resolution_notes", "refund_amount"],
    )(closer_agent)

    context = triage_agent(ticket)
    print("\n→ Handing off triage → resolver (checkpointed)…\n")

    try:
        context = guarded_resolve(context)
    except TimeoutError as exc:
        print("\n" + "!" * 60)
        print("CRASH: resolver failed (expected once).")
        print(f"  Exception: {exc.__class__.__name__}: {exc}")

        recovered = relay.recover(run_id)
        if recovered is None:
            print("ERROR: recover() returned None — verification timing is wrong.")
            print("  VERIFIED must be set BEFORE the wrapped function runs.")
            return 1

        print("\nRECOVERY: last VERIFIED checkpoint found.")
        print(f"  checkpoint_id: {recovered.checkpoint_id}")
        print(f"  from → to:     {recovered.from_agent} → {recovered.to_agent}")
        print(f"  status:        {recovered.status.value}")
        print(f"  context:       {recovered.context}")
        print("  Retrying ONLY the resolver step (not the whole pipeline)…")
        print("!" * 60 + "\n")

        # Retry resolver only, with the recovered context.
        context = guarded_resolve(recovered.context)

    print("\n→ Handing off resolver → closer (checkpointed)…\n")
    context = guarded_close(context)

    print("\n" + "=" * 60)
    print("SUCCESS: pipeline completed. Nothing was lost.")
    print(f"  Final context: {context}")
    history = relay.store.history(run_id)
    print(f"  Checkpoints recorded: {len(history)}")
    for cp in history:
        print(
            f"    - {cp.from_agent} → {cp.to_agent}: {cp.status.value} "
            f"({cp.checkpoint_id[:8]}…)"
        )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
