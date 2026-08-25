# CLAUDE.md

## vLLM source checkout

A full vLLM checkout is available locally as a **sibling of the main repo root**
(not the worktree): `../vllm` relative to
`vllm-simulated-model/`, i.e. absolute
`/Users/villardl/Projects/github.com/vllm-project/vllm`.

Use it read-only to look up vLLM internals, CLI flags, and benchmark result
schemas (e.g. `vllm/benchmarks/serve.py`). **Do not modify the sibling project.**

Note for worktrees: this file may be checked out under
`vllm-simulated-model/.claude/worktrees/<name>/`, so `../vllm` from the worktree
does **not** point at the checkout — resolve it from the main repo root instead.

## Running tests

The test suite requires a virtual environment. Each worktree (and the root
repo) has its own `.venv` at its top level.

### Create and activate the venv (once per worktree)

```bash
uv venv --python 3.12
source .venv/bin/activate
# Install the package in editable mode plus test extras.
# vLLM must already be installed from source.
uv pip install -e ".[test]"
```

### Run tests

```bash
source .venv/bin/activate
pytest tests/
```

The venv's `pytest` is isolated from system-wide packages (e.g. `pytest-ansible`)
that can crash pytest startup, so always use the venv binary, never the system one.

## Documentation style

When writing markdown documentation, use GitHub-flavored alerts for callouts:

```markdown
> [!NOTE]
> Informational context or helpful background.

> [!TIP]
> Optional suggestions to improve workflow or understanding.

> [!IMPORTANT]
> Critical information users should not miss.

> [!WARNING]
> Potential issues or non-obvious behaviors that could cause problems.

> [!CAUTION]
> Actions that can cause data loss, security issues, or system damage.
```

Use alerts sparingly — reserve them for content that truly needs visual emphasis.
Prefer alerts over bold text for platform limitations, security notes, and
destructive operations.
