# agent-handoff-kit

[![PyPI version](https://img.shields.io/pypi/v/agent-handoff-kit)](https://pypi.org/project/agent-handoff-kit/)
[![CI](https://github.com/KMdotcom/agent-handoff-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/KMdotcom/agent-handoff-kit/actions/workflows/ci.yml)

When the receiving agent crashes after a handoff, you shouldn’t rerun triage or lose
context—**agent-handoff-kit** saves a verified snapshot at the boundary and lets you
resume from there.

```bash
pip install agent-handoff-kit
pip install "agent-handoff-kit[openai]"   # OpenAI Agents SDK adapter
```

## Without vs with

Same three-agent billing pipeline; resolver times out once.

```bash
python demo/demo_without_relay.py
python demo/demo_with_relay.py
```

### Without (`demo_without_relay.py`)

```text
[triage] Done. Context: {'ticket_id': 'T-1001', …}

CRASH: resolver failed (expected once).
RESULT: Triage's work is completely lost.
  There is no checkpoint to recover from.
  The only option is restarting the entire pipeline from scratch.
```

### With (`demo_with_relay.py`)

```text
[triage] Done. Context: {'ticket_id': 'T-1001', …}

→ Handing off triage → resolver (checkpointed)…

CRASH: resolver failed (expected once).

RECOVERY: last VERIFIED checkpoint found.
  Retrying ONLY the resolver step (not the whole pipeline)…

SUCCESS: pipeline completed. Nothing was lost.
```

OpenAI + live crash proof: `python demo/demo_openai_adapter.py --live --force-crash`
(see [Demos](#demos) below).

---

**Alpha (0.1.0).** PyPI: `agent-handoff-kit` · import: `agent_handoff_kit` ·
[GitHub](https://github.com/KMdotcom/agent-handoff-kit). OpenAI Agents first; pin the
version in production experiments.

LangGraph already has strong native checkpointing; this is **not** that.
agent-handoff-kit is a thin recovery layer at the handoff boundary for OpenAI Agents
(and plain Python callables). Prefer LangGraph or Temporal when you need a full workflow
engine, durable timers, or graph-native state.

## When to use / not

**Use** when agents hand off with a shared context dict, you need a verified rollback
point before the receiver runs, and you want SQLite-backed resume after a process
crash — without adopting a workflow platform.

**Do not use** as a LangGraph replacement, a Temporal substitute for long-running
workflows, or a general agent framework. It does not schedule, retry with backoff
policies, or manage tool sandboxes.

## Install

Python **3.10–3.13**. Use the **same interpreter** for install and run
(`python -m pip …` then that same `python`). Mixing Homebrew / system Pythons is a
common footgun (`ModuleNotFoundError: agent_handoff_kit`).

Editable install for development:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,openai]"
```

## Concepts

1. **Verify-before-run** — `verify` marks a checkpoint `VERIFIED` *before* the
   receiving agent or wrapped function runs. If the body crashes, status stays
   `VERIFIED` so `recover` still has a safe rollback point.
2. **Envelope** — preferred payload shape stored in `Checkpoint.context`:

   ```python
   {"state": {...}, "messages": [...], "meta": {"from_agent": ..., "to_agent": ..., "sdk": ...}}
   ```

   Flat dicts still work for legacy / core-only use.
3. **`required_keys`** — checked against **app state** (`context["state"]` for
   envelopes, or the flat dict itself). Missing keys → `FAILED` +
   `HandoffVerificationError`; nothing to recover.

## Core quickstart

```python
from agent_handoff_kit import Relay

relay = Relay("run.db")

@relay.guarded_handoff(
    run_id="t1", from_agent="a", to_agent="b", required_keys=["ticket_id"]
)
def handle(context: dict) -> dict:
    return {**context, "done": True}

try:
    handle({"ticket_id": "T-1"})
except Exception:
    cp = relay.recover("t1")  # last VERIFIED snapshot
```

Manual flow: `checkpoint` → `verify` → run receiver → on crash `recover(run_id)`.

## OpenAI Agents quickstart

```python
from agents import Agent
from agent_handoff_kit import Relay
from agent_handoff_kit.openai_adapter import DurableRunner, relayed_handoff

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

On failure after a verified handoff, `DurableRunner` resumes the destination agent
once. Completed `run_id`s refuse a second run (`RunAlreadyCompleted`).

## Idempotency

Side-effecting tools should not double-apply on resume:

```python
from agent_handoff_kit.idempotency import idempotent_tool

@idempotent_tool(relay, run_id, key_fn=lambda ticket_id: f"refund:{ticket_id}")
def issue_refund(ticket_id: str) -> str:
    ...
```

Results are cached in SQLite under `(run_id, idempotency_key)`. For Agents SDK tools,
see `make_idempotent_function_tool`.

## API map

| Piece | Role |
| --- | --- |
| `Relay` | `checkpoint` / `verify` / `recover` / `guarded_handoff` |
| `Checkpoint` / `HandoffStatus` | Snapshot + lifecycle |
| `CheckpointStore` | SQLite backend (`timeout=30`) |
| `make_handoff_envelope` / `checkpoint_state` / `checkpoint_messages` | Envelope helpers |
| `relayed_handoff` / `DurableRunner` / `resume_with_agent` | OpenAI adapter |
| `idempotent_tool` | Tool result cache |

**Limits (Alpha):** single SQLite file; no distributed locks beyond SQLite busy
timeout; non-JSON payloads are sanitized (`model_dump` / `.dict()` / `str`); not a
full workflow engine.

**Proving crash recovery:** `DurableRunner` auto-resumes only when an exception
escapes `Runner.run` (typically a model/worker failure after a VERIFIED handoff).
Tool errors are often returned to the model as soft outputs and may not trigger
auto-resume — see `demo/demo_openai_agents.py` for manual `guarded_handoff` recovery
with a flaky tool.

| Demo | What it proves |
| --- | --- |
| `demo/demo_without_relay.py` | Unprotected pipeline (contrast with hero above) |
| `demo/demo_with_relay.py` | Core `Relay` + recover (no OpenAI) |
| `demo/demo_openai_adapter.py` | Offline: scripted model crash + DurableRunner + idempotency |
| `demo/demo_openai_adapter.py --live` | Live smoke (happy path) |
| `demo/demo_openai_adapter.py --live --force-crash` | **Live proof:** real API + model crash + DurableRunner auto-resume |
| `demo/demo_openai_agents.py --live` | Live `guarded_handoff` + flaky billing tool |

## Demos

```bash
python -m pip install -e ".[openai]"
python demo/demo_without_relay.py
python demo/demo_with_relay.py
python demo/demo_openai_adapter.py
python demo/demo_openai_adapter.py --live
python demo/demo_openai_adapter.py --live --force-crash   # run before announcing
```

## Development

```bash
python -m pip install -e ".[dev,openai]"
ruff check src tests demo && ruff format --check src tests demo
pytest -q
```

## Publishing (TestPyPI first)

PyPI versions are **immutable**. A bad `0.1.0` forces `0.1.1`. Always dry-run on
TestPyPI before real upload.

```bash
rm -rf build/ dist/ *.egg-info src/*.egg-info
python -m build
python -m twine check dist/*

# TestPyPI
python -m twine upload --repository testpypi dist/*

python -m venv /tmp/hk-test && source /tmp/hk-test/bin/activate
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  agent-handoff-kit
python -c "from agent_handoff_kit import Relay; print(Relay)"
```

(`--extra-index-url` keeps optional deps like `openai-agents` resolvable from real
PyPI.)

Then production:

```bash
python -m twine upload dist/*
```

### Launch checklist

- [ ] `.gitignore` covers `.env`, `*.db`, `dist/`, `build/`, egg-info
- [ ] SQLite `timeout=30.0`
- [ ] Safe JSON serialize for non-JSON payloads
- [ ] Clean build; sdist lists all `agent_handoff_kit` modules
- [ ] `twine check dist/*` passes
- [ ] Local wheel: `from agent_handoff_kit import Relay`
- [ ] TestPyPI upload + fresh-venv install + import
- [ ] Real PyPI: `twine upload dist/*`

Verify package contents after a clean build:

```bash
tar -tzf dist/agent_handoff_kit-*.tar.gz | grep 'agent_handoff_kit/.*\.py'
# must list __init__.py, core.py, models.py, store.py, idempotency.py, openai_adapter.py
```

## License

MIT
