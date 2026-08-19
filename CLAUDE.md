# CLAUDE.md

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
