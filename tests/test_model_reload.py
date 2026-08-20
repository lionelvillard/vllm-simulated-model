import json
import types
from pathlib import Path

import pytest

from vllm_simulated.model import _bootstrap_config_file, _load_latency_from_file


def _make_hf_config(beta):
    """Minimal hf_config stand-in with a latency block."""
    return types.SimpleNamespace(
        latency={
            "type": "physics",
            "hardware": {
                "peak_tflops": 989.0,
                "hbm_gbps": 3350.0,
                "weight_dtype": "bfloat16",
            },
            "beta": beta,
            "tp": 1,
            "deterministic_length": True,
        },
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        vocab_size=256,
    )


@pytest.fixture()
def config_path(tmp_path):
    return str(tmp_path / "sim-config.json")


def test_load_latency_from_file_returns_model_with_correct_beta(config_path):
    hf = _make_hf_config([0.5, 0.8, 10.0])
    cfg = {"latency": dict(hf.latency)}
    Path(config_path).write_text(json.dumps(cfg))

    model = _load_latency_from_file(config_path, hf)

    # PhysicsLatencyModel stores beta on its config attribute
    assert tuple(model.config.beta) == (0.5, 0.8, 10.0)


def test_bootstrap_config_file_creates_file_if_absent(config_path):
    hf = _make_hf_config([1.0, 1.0, 0.0])
    _bootstrap_config_file(config_path, hf)

    data = json.loads(Path(config_path).read_text())
    assert data["latency"]["beta"] == [1.0, 1.0, 0.0]


def test_bootstrap_config_file_does_not_overwrite_existing(config_path):
    Path(config_path).write_text(json.dumps({"latency": {"beta": [9, 9, 9]}}))
    hf = _make_hf_config([1.0, 1.0, 0.0])
    _bootstrap_config_file(config_path, hf)

    data = json.loads(Path(config_path).read_text())
    assert data["latency"]["beta"] == [9, 9, 9]  # unchanged


def test_load_latency_from_file_picks_up_updated_beta(config_path):
    hf = _make_hf_config([1.0, 1.0, 0.0])
    Path(config_path).write_text(json.dumps({"latency": dict(hf.latency)}))

    # Update the file with a new beta
    new_cfg = {"latency": {**dict(hf.latency), "beta": [2.0, 0.5, 7.0]}}
    Path(config_path).write_text(json.dumps(new_cfg))

    model = _load_latency_from_file(config_path, hf)
    assert tuple(model.config.beta) == (2.0, 0.5, 7.0)
