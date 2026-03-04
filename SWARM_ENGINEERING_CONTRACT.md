# Swarm Engineering Contract

## System Rules

### 1. Repository Architecture is Immutable

The following directories are **READ-ONLY**:

- `core/`
- `contracts/`
- `tests/`

Never modify files in these directories unless explicitly instructed.

### 2. Editable Zones

Editable directories:

- `agents/`
- `pipelines/`
- `scripts/`

### 3. Architecture Cannot Be Changed

You may **NOT**:

- Rename modules
- Restructure folders
- Remove interfaces
- Refactor system architecture

You may **ONLY** implement functionality inside existing modules.

### 4. Validation Required

Before submitting code you must ensure:

- Tests pass
- Code compiles
- Schema contracts remain valid
- Existing functions remain backward compatible

### 5. Modification Scope

- Maximum modification scope: **200 lines**
- If more changes are required, propose a plan first

### 6. Schema Contracts

Dataset outputs must match the schema in:

```
contracts/pair.schema.json
```

### 7. Tests Required

Tests go in: `tests/`

Always include tests when adding new functionality.

### 8. Minimal Changes

Fix the task with the smallest safe modification. Never refactor unrelated code.

---

## Workflow

1. Analyze the repository structure
2. Identify the minimal file to modify
3. Explain the change in a short plan
4. Implement the change

## Output Format

1. Plan
2. Files modified
3. Code changes
4. Tests added
5. Confirmation tests pass
