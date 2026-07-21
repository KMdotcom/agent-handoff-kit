# handoff-kit

Checkpoint multi-agent handoffs. Verify the receiver got what it needs. Resume after a crash — without Temporal.

LangGraph already has strong native checkpointing; this is not that. handoff-kit is for OpenAI Agents (and soon CrewAI / PydanticAI): a thin recovery layer at the handoff boundary.

## Install

```bash
pip install handoff-kit
pip install "handoff-kit[openai]"   # OpenAI Agents SDK adapter
```

Python **3.10–3.13**. Always install with the same interpreter you run (`python -m pip …`).

## Quickstart

```python
from agents import Agent
from handoff_kit import Relay
from handoff_kit.openai_adapter import DurableRunner, relayed_handoff

relay = Relay("support.db")
run_id = "ticket-42"

resolver = Agent(name="Resolver", instructions="Resolve the ticket.")
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
```

On failure after a verified handoff, `DurableRunner` resumes the destination agent once. Completed `run_id`s refuse a second run.

Framework-agnostic core (no OpenAI)::

```python
from handoff_kit import Relay

relay = Relay("run.db")

@relay.guarded_handoff(
    run_id="t1", from_agent="a", to_agent="b", required_keys=["ticket_id"]
)
def handle(context: dict) -> dict:
    return {**context, "done": True}
```

## Demos

```bash
pip install -e ".[openai]"
python demo/demo_with_relay.py
python demo/demo_openai_adapter.py          # offline DurableRunner
python demo/demo_openai_adapter.py --live   # needs OPENAI_API_KEY / .env
```

## License

MIT
