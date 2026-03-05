# AI Context — SwarmSignal Repository

This file is long-term memory for AI agents working in this repo.
Read this before writing any code.

---

## System Goals

- Build autonomous AI intelligence infrastructure — not a model lab, an AI refinery
- SwarmSignal is the foundation layer: AI trends, signal extraction, edge inference
- Every model earns its slot. No bloat. Signal first.
- Primary goal is **stability**, not refactoring

## Coding Style

- Python 3.12+, type hints everywhere
- Minimal changes only — patch-based, never rewrite large files
- No unnecessary abstractions. Three similar lines > a premature helper
- All code must pass the test gate: `pytest`, `ruff`, `mypy`
- Maximum modification scope: 200 lines per change

## Agent Responsibilities

All agents inherit from `BaseAgent` and must implement `run()`.
All agents must be registered in the `AGENTS` dictionary:

```python
# swarm/agents/__init__.py
AGENTS = {}
```

No ad-hoc agent creation. New agents require a registry entry.

## Dataset Schema

All pipeline outputs must conform to `contracts/pair.schema.json`:

```json
{
  "instruction": "string",
  "response": "string",
  "score": "float",
  "domain": "string",
  "metadata": "object"
}
```

Required fields: `instruction`, `response`, `score`, `domain`.
Optional: `metadata` (trajectory step, specialty, format, source).

## Repository Structure

```
src/            SOURCE   — training scripts, cooks, assemblers
core/           READ-ONLY — runtime, config, registry
contracts/      READ-ONLY — JSON schema definitions
tests/          READ-ONLY — validation layer
agents/         EDITABLE  — AI agent implementations
pipelines/      EDITABLE  — training and dataset pipelines
scripts/        EDITABLE  — shell scripts for training, eval, builds
supervisor/     GUARD     — repo_guard.py enforces protected paths
```

## Protected Paths

```python
PROTECTED_PATHS = ["core/", "contracts/", "tests/"]
```

Patches touching these paths will be rejected by `supervisor/repo_guard.py`.

## Key Files

| File | Purpose |
|------|---------|
| `SWARM_ENGINEERING_CONTRACT.md` | Rules all agents must follow |
| `ARCHITECTURE.md` | System structure reference |
| `AI_CONTEXT.md` | This file — long-term repo memory |
| `core/runtime.py` | SwarmRuntime, RunContext, agent execution |
| `core/config.py` | SwarmConfig — paths, model defaults, thresholds |
| `core/registry.py` | AgentRegistry — register, get, list, validate |
| `agents/__init__.py` | AGENTS dict — the single source of truth |
| `contracts/pair.schema.json` | Canonical dataset schema |
| `supervisor/repo_guard.py` | Protected path enforcement |
| `scripts/run_tests.sh` | Mandatory test gate (pytest + ruff + mypy) |

## Workflow

1. Read `SWARM_ENGINEERING_CONTRACT.md` and `ARCHITECTURE.md`
2. Propose a plan
3. Generate a patch (agents/ or pipelines/ only)
4. Run `scripts/run_tests.sh`
5. If tests fail, change is rejected
6. Submit change
