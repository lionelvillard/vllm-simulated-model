import json
from pathlib import Path

from evaluation.gen_sim_config import build_sim_config, render_configmap, H100_SXM5

FIX = Path(__file__).parent / "fixtures" / "eval"


def _real():
    with open(FIX / "qwen3-32b-config.json") as f:
        return json.load(f)


def test_build_preserves_architecture_and_injects_physics():
    sim = build_sim_config(_real())
    # architecture copied verbatim
    assert sim["hidden_size"] == 5120
    assert sim["num_hidden_layers"] == 64
    assert sim["num_key_value_heads"] == 8
    assert sim["intermediate_size"] == 25600
    # simulated architecture registered
    assert sim["architectures"] == ["SimulatedForCausalLM"]
    # physics latency injected with H100 SXM5 defaults
    lat = sim["latency"]
    assert lat["type"] == "physics"
    assert lat["hardware"] == H100_SXM5
    assert lat["tp"] == 1
    assert lat["beta"] == [1.0, 1.0, 0.0]
    assert lat["deterministic_length"] is True


def test_build_drops_stale_latency():
    real = _real()
    real["latency"] = {"type": "linear", "base_ms": 999.0}
    sim = build_sim_config(real)
    assert sim["latency"]["type"] == "physics"


def test_render_configmap_is_valid_yaml_with_embedded_json():
    import yaml
    sim = build_sim_config(_real())
    text = render_configmap(sim)
    doc = yaml.safe_load(text)
    assert doc["kind"] == "ConfigMap"
    assert doc["metadata"]["name"] == "vllm-sim-model-config"
    embedded = json.loads(doc["data"]["config.json"])
    assert embedded["architectures"] == ["SimulatedForCausalLM"]
    assert embedded["latency"]["type"] == "physics"
