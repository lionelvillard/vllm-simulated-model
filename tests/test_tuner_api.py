import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

INITIAL_CONFIG = {
    "latency": {
        "type": "physics",
        "hardware": {
            "peak_tflops": 989.0,
            "hbm_gbps": 3350.0,
            "weight_dtype": "bfloat16",
        },
        "beta": [1.0, 1.0, 0.0],
        "tp": 1,
        "deterministic_length": True,
    }
}


@pytest.fixture()
def config_file(tmp_path):
    p = tmp_path / "sim-config.json"
    p.write_text(json.dumps(INITIAL_CONFIG))
    return p


@pytest.fixture()
def client(config_file, monkeypatch):
    monkeypatch.setenv("VLLM_SIM_TUNER", "1")
    monkeypatch.setenv("VLLM_SIM_CONFIG_PATH", str(config_file))
    # Import after env vars are set so attach_router sees them
    from vllm_simulated.tuner_api import SimTunerEndpointPlugin
    app = FastAPI()
    plugin = SimTunerEndpointPlugin()
    plugin.attach_router(app)
    return TestClient(app)


def test_post_sim_config_updates_beta(client, config_file):
    resp = client.post("/sim/config", json={"beta": [0.15, 0.9, 5.0]})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "beta": [0.15, 0.9, 5.0]}
    data = json.loads(config_file.read_text())
    assert data["latency"]["beta"] == [0.15, 0.9, 5.0]


def test_post_sim_config_rejects_wrong_length(client):
    resp = client.post("/sim/config", json={"beta": [1.0, 2.0]})
    assert resp.status_code == 422


def test_post_sim_config_rejects_negative_values(client):
    resp = client.post("/sim/config", json={"beta": [1.0, -0.1, 0.0]})
    assert resp.status_code == 422


def test_no_route_when_tuner_disabled(config_file, monkeypatch):
    monkeypatch.delenv("VLLM_SIM_TUNER", raising=False)
    monkeypatch.setenv("VLLM_SIM_CONFIG_PATH", str(config_file))
    from vllm_simulated.tuner_api import SimTunerEndpointPlugin
    app = FastAPI()
    plugin = SimTunerEndpointPlugin()
    plugin.attach_router(app)
    c = TestClient(app)
    resp = c.post("/sim/config", json={"beta": [1.0, 1.0, 0.0]})
    assert resp.status_code == 404


def test_atomic_write_does_not_corrupt_file(client, config_file):
    resp = client.post("/sim/config", json={"beta": [0.3, 0.7, 12.5]})
    assert resp.status_code == 200
    # File must be valid JSON after the write
    data = json.loads(config_file.read_text())
    assert isinstance(data["latency"]["beta"], list)
