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

### OpenAI Agents SDK adapter (drop-in)

Pass a dict as `Runner.run(..., context=...)`. `relayed_handoff` checkpoints and
verifies that dict at the native SDK transfer boundary, then you resume only
the receiving agent after a crash:

```python
from agents import Agent, Runner
from agent_relay import Relay
from agent_relay.openai_adapter import relayed_handoff, resume_run, resume_with_agent

relay = Relay("support.db")
run_id = "ticket-42"
resolver = Agent(name="Resolver", instructions="Resolve the ticket.")
triage = Agent(
    name="Triage",
    instructions="Hand off to Resolver when you have ticket_id + summary.",
    handoffs=[
        relayed_handoff(
            relay,
            run_id=run_id,
            from_agent="triage",
            to_agent="resolver",
            agent=resolver,
            required_keys=["ticket_id", "summary"],
        )
    ],
)

context = {"ticket_id": "T-1", "summary": "Double charge"}
try:
    await Runner.run(triage, "I was double-charged", context=context)
except Exception:
    recovered = resume_run(relay, run_id)          # last VERIFIED context
    await resume_with_agent(relay, run_id, resolver)  # resolver only
```

`agent_relay.openai_adapter` is the version-sensitive seam; `core` stays stable when the Agents SDK moves. Requires `pip install 'openai-agents'`.

## Demos

Plain demos (stdlib only) from the repo root:

```bash
python3 demo/demo_without_relay.py   # crash; triage work lost
python3 demo/demo_with_relay.py      # crash once, recover, finish
```

### Real OpenAI Agents SDK demos

```bash
pip install -r requirements.txt
# optional for --live: copy .env.example → .env and set OPENAI_API_KEY

python3 demo/demo_openai_agents.py     # programmatic guarded Runner.run
python3 demo/demo_openai_adapter.py    # native handoff via relayed_handoff + resume
python3 demo/demo_openai_adapter.py --live
```

`guarded_handoff` supports both sync and async callables (needed for `Runner.run`).

## Public API

| Symbol | Role |
|--------|------|
| `Relay` | checkpoint / verify / recover / `guarded_handoff` |
| `Checkpoint` / `HandoffStatus` | data model |
| `CheckpointStore` | SQLite persistence (swap later for Postgres/Redis) |
| `HandoffVerificationError` | missing `required_keys` |

## Next steps before pitching

- [x] **OpenAI adapter drop-in** — `relayed_handoff` / `resume_with_agent` + `demo/demo_openai_adapter.py`
- [ ] **CrewAI adapter** — wrap delegation / task boundaries with the same Relay
- [ ] **PydanticAI adapter** — wrap programmatic handoff loops (`message_history` handoffs)
- [x] **Real (non-toy) demo** — `demo/demo_openai_agents.py` (scripted Model offline + `--live` for real OpenAI)
- [ ] **PyPI publish** — `pip install agent-relay`
- [ ] **Get 5–10 developers** from LangGraph / CrewAI / OpenAI Agents communities to try the MVP and give feedback on the API

## License

MIT
