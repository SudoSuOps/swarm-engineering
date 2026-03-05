# SwarmSignal Architecture

The SwarmSignal system is structured as follows:

```
src/        – training scripts, data pipelines, signal collection
core/       – runtime and registry (immutable)
agents/     – AI agents (registry pattern)
pipelines/  – dataset and eval pipelines
contracts/  – schema definitions
tests/      – validation layer
```

Agents may only modify code inside `agents/` and `pipelines/`.
