#!/usr/bin/env python
import sys
import os

# Add parent vllm venv to path
parent_venv = "/Users/villardl/Projects/github.com/vllm-project/vllm/.venv/lib/python3.12/site-packages"
if parent_venv not in sys.path:
    sys.path.insert(0, parent_venv)

# Now import and run pytest
import pytest

sys.exit(pytest.main([
    "tests/test_model_reload.py",
    "tests/test_tuner_api.py",
    "tests/test_tune.py",
    "-v",
    "--tb=short"
]))
