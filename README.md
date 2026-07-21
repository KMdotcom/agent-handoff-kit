# agent-relay

A lightweight, framework-agnostic recovery layer for multi-agent handoffs.

When one AI agent hands work to another mid-workflow, that handoff can fail silently (context dropped) or catastrophically (a crash loses all prior work). **agent-relay** checkpoints state at the handoff boundary, verifies the receiving agent got what it needs *before* the risky step runs, and lets you roll back to the last clean checkpoint instead of restarting the whole run.

## Why this exists (and why not LangGraph)

**LangGraph already has strong native checkpointing.** This project deliberately does **not** compete there. If you are on LangGraph, use its checkpointers.

agent-relay targets stacks whose own docs still push you toward bolting on a heavy general-purpose system (e.g. Temporal) for handoff durability:

- **OpenAI Agents SDK** — first-class `handoff`, but no crash recovery at handoff boundaries
- **CrewAI / PydanticAI** — adapters planned; core is framework-agnostic today

## Python version support

Supports **Python 3.10–3.13**. There is nothing special about 3.11.

If installs seem missing, you almost certainly mixed interpreters (e.g. Homebrew `python3` → 3.14 while packages landed in 3.11). Always install and run with the **same** binary:

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[openai]"
python demo/demo_openai_adapter.py
```

Python **3.14** may not work until `openai-agents` supports it — prefer 3.12 for demos.

## Install

Core library is **stdlib-only** (SQLite). OpenAI integration is an optional extra.

```bash
pip install agent-relay              # core
pip install "agent-relay[openai]"    # + openai-agents
# from this repo:
pip install -e ".[openai]"
```

### Publish to PyPI (maintainers)

```bash
python -m pip install build twine
python -m build
twine upload dist/*
```

## Quickstart (OpenAI Agents — DurableRunner)

```python
from agents import Agent, Runner
from agent_relay import Relay
from agent_relay.idempotency import make_idempotent_function_tool
from agent_relay.openai_adapter import DurableRunner, relayed_handoff

relay = Relay("support.db")
run_id = "ticket-42"

def _refund(ticket_id: str) -> str:
    return f"refunded {ticket_id}"

refund = make_idempotent_function_tool(
    relay, run_id, _refund, key_fn=lambda ticket_id: f"refund:{ticket_id}"
)

resolver = Agent(name="Resolver", instructions="Resolve it.", tools=[refund])
triage = Agent(
    name="Triage",
    instructions="Hand off to Resolver.",
    handoffs=[
        relayed_handoff(
            relay, run_id, "triage", "resolver", resolver,
            required_keys=["ticket_id", "summary"],
        )
    ],
)

await DurableRunner.run(
    relay,
    run_id,
    triage,
    "I was double-charged",
    context={"ticket_id": "T-1", "summary": "Double charge"},
    agents={"triage": triage, "resolver": resolver},
)
# On crash after handoff: DurableRunner auto-resumes resolver once.
# Re-running the same run_id after success raises RunAlreadyCompleted.
```

Checkpoints store a **full-context envelope**: `{state, messages, meta}`. `required_keys` apply to `state`. Helpers: `checkpoint_state`, `checkpoint_messages`.

### Framework-agnostic core

```python
from agent_relay import Relay

relay = Relay("my_run.db")

@relay.guarded_handoff(
    run_id="ticket-42",
    from_agent="triage",
    to_agent="resolver",
    required_keys=["ticket_id", "summary"],
)
def resolve(context: dict) -> dict:
    return {**context, "resolution_notes": "…"}
```

## Demos

```bash
python demo/demo_without_relay.py      # crash; work lost
python demo/demo_with_relay.py         # plain guarded_handoff recovery
python demo/demo_openai_agents.py      # programmatic Runner.run + flaky tool
python demo/demo_openai_adapter.py     # DurableRunner + idempotent refund (offline)
python demo/demo_openai_adapter.py --live
```

## Public API

| Symbol | Role |
|--------|------|
| `Relay` | checkpoint / verify / recover / `guarded_handoff` |
| `Checkpoint` / `HandoffStatus` | data model |
| `checkpoint_state` / `checkpoint_messages` | envelope helpers |
| `CheckpointStore` | SQLite persistence (+ run_meta, tool_invocations) |
| `idempotent_tool` / `make_idempotent_function_tool` | side-effect dedup |
| `relayed_handoff` / `DurableRunner` | OpenAI Agents adapter |

## Next steps before pitching

- [x] OpenAI adapter drop-in (`relayed_handoff`, `DurableRunner`)
- [x] Full-context envelope + automatic recovery + tool idempotency
- [x] PyPI-ready packaging metadata
- [ ] CrewAI adapter
- [ ] PydanticAI adapter
- [ ] Actually publish `agent-relay` to PyPI
- [ ] Get 5–10 developers to try it

## License

MIT
