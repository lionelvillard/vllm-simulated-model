# tests/test_eval_manifests.py
from pathlib import Path

import yaml

STANDALONE = (
    Path(__file__).parent.parent
    / "models"
    / "qwen3-32b"
    / "deployments"
    / "h100-sxm5"
    / "standalone"
    / "latency"
)

VARIANTS = {
    "flat": ("dcd098", STANDALONE / "flat"),
    "physics": ("eae748", STANDALONE / "physics"),
    "physics-beta-1.0": ("72d33f", STANDALONE / "physics-beta-1.0"),
}


def _load(path):
    with open(path) as f:
        return list(yaml.safe_load_all(f))[0]


def test_all_manifests_parse():
    for variant, (_, base) in VARIANTS.items():
        for f in base.glob("**/*.yaml"):
            with open(f) as fh:
                docs = list(yaml.safe_load_all(fh))
            assert docs and all(d.get("kind") for d in docs), f


def test_real_deployment_requests_gpu_and_correct_model():
    for variant, (h, base) in VARIANTS.items():
        d = _load(base / "eval" / "real-deployment.yaml")
        c = d["spec"]["template"]["spec"]["containers"][0]
        assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
        cmd = " ".join(c["command"] + c.get("args", []))
        assert "Qwen/Qwen3-32B" in cmd
        assert "--served-model-name qwen3-32b" in cmd
        assert "--tensor-parallel-size 1" in cmd


def test_sim_and_real_share_served_model_name():
    for variant, (h, base) in VARIANTS.items():
        real = _load(base / "eval" / "real-deployment.yaml")
        sim = _load(base / "eval" / "sim-deployment.yaml")
        rc = " ".join(real["spec"]["template"]["spec"]["containers"][0]["command"]
                      + real["spec"]["template"]["spec"]["containers"][0].get("args", []))
        sc = " ".join(sim["spec"]["template"]["spec"]["containers"][0]["command"]
                      + sim["spec"]["template"]["spec"]["containers"][0].get("args", []))
        assert "--served-model-name qwen3-32b" in rc
        assert "--served-model-name qwen3-32b" in sc


def test_services_target_their_apps():
    for variant, (h, base) in VARIANTS.items():
        real_svc = _load(base / "eval" / "real-service.yaml")
        sim_svc = _load(base / "eval" / "sim-service.yaml")
        assert real_svc["spec"]["selector"]["app"] == f"vllm-qwen3-32b-standalone-{h}-real"
        assert sim_svc["spec"]["selector"]["app"] == f"vllm-qwen3-32b-standalone-{h}-sim"


def test_benchmark_job_urls_match_services():
    for variant, (h, base) in VARIANTS.items():
        job = _load(base / "eval" / "benchmark-job.yaml")
        cmd = job["spec"]["template"]["spec"]["containers"][0]["command"]
        script = " ".join(cmd)
        assert f"vllm-qwen3-32b-standalone-{h}-real:8000" in script
        assert f"vllm-qwen3-32b-standalone-{h}-sim:8000" in script


def test_deployment_names_match_services():
    for variant, (h, base) in VARIANTS.items():
        dep = _load(base / "deployment.yaml")
        svc = _load(base / "service.yaml")
        expected = f"vllm-qwen3-32b-standalone-{h}"
        assert dep["metadata"]["name"] == expected
        assert svc["spec"]["selector"]["app"] == expected


def test_configmap_name_matches_deployment_volume():
    for variant, (h, base) in VARIANTS.items():
        cm = _load(base / "configmap.yaml")
        dep = _load(base / "deployment.yaml")
        expected_cm = f"vllm-qwen3-32b-standalone-{h}-config"
        assert cm["metadata"]["name"] == expected_cm
        volumes = dep["spec"]["template"]["spec"]["volumes"]
        cfg_vol = next(v for v in volumes if v["name"] == "model-config")
        assert cfg_vol["configMap"]["name"] == expected_cm
