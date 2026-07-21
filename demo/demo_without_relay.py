#!/usr/bin/env python3
"""Demo: 3-agent support pipeline WITHOUT handoff-kit.

Shows what happens when the resolver crashes mid-handoff: triage's work is
gone, and the only option is restarting from scratch.
"""

from __future__ import annotations

import sys
import traceback


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
    print("[resolver] Attempting to resolve ticket…")
    # Simulated tool timeout — always fails in this unprotected demo.
    raise TimeoutError("billing API timed out after 30s")


def closer_agent(context: dict) -> dict:
    print("[closer] Closing ticket…")
    context = dict(context)
    context["status"] = "closed"
    context["resolution"] = "Refund issued"
    print(f"[closer] Done. Final context: {context}")
    return context


def main() -> int:
    print("=" * 60)
    print("DEMO: support pipeline WITHOUT handoff-kit")
    print("=" * 60)

    ticket = "I was double-charged on invoice #8821"
    context = None

    try:
        context = triage_agent(ticket)
        print("\n→ Handing off triage → resolver (no checkpoint)…\n")
        context = resolver_agent(context)
        context = closer_agent(context)
    except Exception as exc:
        print("\n" + "!" * 60)
        print("CRASH: resolver failed.")
        print(f"  Exception: {exc.__class__.__name__}: {exc}")
        print()
        print("RESULT: Triage's work is completely lost.")
        print(f"  In-memory context after crash: {context!r}")
        print("  There is no checkpoint to recover from.")
        print("  The only option is restarting the entire pipeline from scratch.")
        print("!" * 60)
        traceback.print_exc(file=sys.stdout)
        return 1

    print("\nPipeline completed (unexpected for this demo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
