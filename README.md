# agent-relay

A lightweight, framework-agnostic recovery layer for multi-agent handoffs.

When one AI agent hands work to another mid-workflow, that handoff can fail silently (context dropped) or catastrophically (a crash loses all prior work). **agent-relay** checkpoints state at the handoff boundary, verifies the receiving agent got what it needs *before* the risky step runs, and lets you roll back to the last clean checkpoint instead of restarting the whole run.

## Why this exists (and why not LangGraph)

**LangGraph already has strong native checkpointing.** This project deliberately does **not** compete there. If you are on LangGraph, use its checkpointers.

agent-relay targets stacks whose own docs still push you toward bolting on a heavy general-purpose system (e.g. Temporal) for handoff durability:

- **OpenAI Agents SDK** — first-class `handoff`, but no crash recovery at handoff boundaries
- **CrewAI / PydanticAI** — adapters planned; core is framework-agnostic today

This is the thin, purpose-built alternative: checkpoint → verify → recover around handoffs, not a full workflow engine.

## Install

MVP is **stdlib-only** (SQLite via `sqlite3`).

```bash
# from the repo root
pip install -e .
# or just put the repo on PYTHONPATH / run demos as shown below
```

## Quickstart

```python
from agent_relay import Relay

relay = Relay("my_run.db")
run_id = "ticket-42"

@relay.guarded_handoff(
    run_id=run_id,
    from_agent="triage",
    to_agent="resolver",
    required_keys=["ticket_id", "summary"],
)
def resolve(context: dict) -> dict:
    # your receiving-agent logic
    context = dict(context)
    context["resolution_notes"] = "…"
    return context

context = {"ticket_id": "T-1", "summary": "Double charge"}
try:
    context = resolve(context)
except Exception:
    recovered = relay.recover(run_id)
    if recovered:
        context = resolve(recovered.context)  # retry this step only
```

### Pre-flight verification (the subtle bit)

`verify()` runs **before** the wrapped function. A checkpoint is marked `VERIFIED` once required keys are present — even if `to_agent` later throws. Downstream failures do **not** downgrade that status, so `recover(run_id)` still has a safe rollback point.

### OpenAI Agents SDK adapter

```python
from agent_relay import Relay
from agent_relay.openai_adapter import wrap_handoff, resume_run

relay = Relay("support.db")
guarded = wrap_handoff(
    relay,
    run_id="ticket-42",
    from_agent="triage",
    to_agent="resolver",
    on_handoff=my_on_handoff,
    required_keys=["ticket_id", "summary"],
)
# pass ``guarded`` as on_handoff to the SDK
# after a crash: resume_run(relay, "ticket-42")
```

The adapter file is the version-sensitive seam; core stays stable when the Agents SDK moves.

## Demos

No third-party deps. From the repo root:

```bash
python demo/demo_without_relay.py   # crash; triage work lost
python demo/demo_with_relay.py      # crash once, recover, finish
```

## Public API

| Symbol | Role |
|--------|------|
| `Relay` | checkpoint / verify / recover / `guarded_handoff` |
| `Checkpoint` / `HandoffStatus` | data model |
| `CheckpointStore` | SQLite persistence (swap later for Postgres/Redis) |
| `HandoffVerificationError` | missing `required_keys` |

## Next steps before pitching

- [ ] **CrewAI adapter** — wrap delegation / task boundaries with the same Relay
- [ ] **PydanticAI adapter** — wrap programmatic handoff loops (`message_history` handoffs)
- [ ] **Real (non-toy) demo** — a live multi-agent workflow on OpenAI Agents SDK, not plain functions
- [ ] **Get 5–10 developers** from LangGraph / CrewAI / OpenAI Agents communities to try the MVP and give feedback on the API

## License

MIT
