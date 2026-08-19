import argparse
import copy
import json

import yaml

H100_SXM5 = {"peak_tflops": 989.0, "hbm_gbps": 3350.0, "weight_dtype": "bfloat16"}


def build_sim_config(
    real_config: dict,
    *,
    hardware: dict = H100_SXM5,
    tp: int = 1,
    beta=(1.0, 1.0, 0.0),
    deterministic_length: bool = True,
) -> dict:
    sim = copy.deepcopy(real_config)
    sim.pop("latency", None)
    sim["architectures"] = ["SimulatedForCausalLM"]
    sim["latency"] = {
        "type": "physics",
        "hardware": dict(hardware),
        "beta": [float(b) for b in beta],
        "tp": int(tp),
        "deterministic_length": bool(deterministic_length),
    }
    return sim


def render_configmap(sim_config: dict, name: str = "vllm-sim-model-config") -> str:
    doc = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name},
        "data": {"config.json": json.dumps(sim_config, indent=2)},
    }
    return yaml.safe_dump(doc, sort_keys=False)


def _load_real_config(model: str) -> dict:
    """CLI-only: fetch config.json from HF or read a local path."""
    import os

    if os.path.isfile(model):
        with open(model) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model, filename="config.json")
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate sim ConfigMap from a real model config."
    )
    ap.add_argument("--model", required=True, help="HF model id or path to config.json")
    ap.add_argument("--out", required=True, help="output ConfigMap YAML path")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument(
        "--peak-tflops", type=float, default=H100_SXM5["peak_tflops"]
    )
    ap.add_argument("--hbm-gbps", type=float, default=H100_SXM5["hbm_gbps"])
    ap.add_argument(
        "--weight-dtype", default=H100_SXM5["weight_dtype"]
    )
    args = ap.parse_args(argv)

    real = _load_real_config(args.model)
    hardware = {
        "peak_tflops": args.peak_tflops,
        "hbm_gbps": args.hbm_gbps,
        "weight_dtype": args.weight_dtype,
    }
    sim = build_sim_config(real, hardware=hardware, tp=args.tp)
    with open(args.out, "w") as f:
        f.write(render_configmap(sim))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
