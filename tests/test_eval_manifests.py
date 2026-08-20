# tests/test_eval_manifests.py
from pathlib import Path

import yaml

DEPLOY = (
    Path(__file__).parent.parent
    / "models"
    / "qwen3-32b"
    / "deployments"
    / "h100-sxm5-tp1"
    / "k8s"
    / "eval"
)


def _load(name):
    with open(DEPLOY / name) as f:
        return list(yaml.safe_load_all(f))[0]


def test_all_manifests_parse():
    for f in DEPLOY.glob("*.yaml"):
        with open(f) as fh:
            docs = list(yaml.safe_load_all(fh))
        assert docs and all(d.get("kind") for d in docs), f


def test_real_deployment_requests_gpu_and_correct_model():
    d = _load("real-deployment.yaml")
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    cmd = " ".join(c["command"] + c.get("args", []))
    assert "Qwen/Qwen3-32B" in cmd
    assert "--served-model-name qwen3-32b" in cmd
    assert "--tensor-parallel-size 1" in cmd


def test_sim_and_real_share_served_model_name():
    real = _load("real-deployment.yaml")
    sim = _load("sim-deployment.yaml")
    rc = " ".join(real["spec"]["template"]["spec"]["containers"][0]["command"]
                  + real["spec"]["template"]["spec"]["containers"][0].get("args", []))
    sc = " ".join(sim["spec"]["template"]["spec"]["containers"][0]["command"]
                  + sim["spec"]["template"]["spec"]["containers"][0].get("args", []))
    assert "--served-model-name qwen3-32b" in rc
    assert "--served-model-name qwen3-32b" in sc


def test_services_target_their_apps():
    assert _load("real-service.yaml")["spec"]["selector"]["app"] == "vllm-real"
    assert _load("sim-service.yaml")["spec"]["selector"]["app"] == "vllm-sim"
