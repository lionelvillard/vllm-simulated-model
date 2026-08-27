import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

# Resolve vllm from the same venv as this interpreter, not from PATH.
_VLLM_BIN = str(Path(sys.executable).parent / "vllm")

from evaluation.compare import (
    PointResult, aggregate, compare_point, load_result,
    render_json, render_markdown,
)


@dataclass(frozen=True)
class SweepPoint:
    isl: int
    osl: int
    concurrency: int
    num_prompts: int

    @property
    def label(self) -> str:
        return f"ISL={self.isl} OSL={self.osl} c={self.concurrency}"


def load_sweep(path) -> list[SweepPoint]:
    with open(Path(path)) as f:
        doc = yaml.safe_load(f)
    return [
        SweepPoint(
            isl=p["isl"], osl=p["osl"],
            concurrency=p["concurrency"], num_prompts=p["num_prompts"],
        )
        for p in doc["points"]
    ]


def bench_argv(
    point: SweepPoint,
    *,
    base_url: str,
    model: str,
    tokenizer: str,
    result_dir: str,
    result_filename: str,
    seed: int = 0,
    backend: str = "openai",
    endpoint: str = "/v1/completions",
) -> list[str]:
    return [
        _VLLM_BIN, "bench", "serve",
        "--backend", backend,
        "--base-url", base_url,
        "--endpoint", endpoint,
        "--model", model,
        "--tokenizer", tokenizer,
        "--dataset-name", "random",
        "--random-input-len", str(point.isl),
        "--random-output-len", str(point.osl),
        "--num-prompts", str(point.num_prompts),
        "--max-concurrency", str(point.concurrency),
        "--ignore-eos",
        "--seed", str(seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "90,99",
        "--save-result",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
    ]


def _detect_model(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["id"]


def _fetch_version(base_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url}/version", timeout=5) as resp:
            return json.loads(resp.read()).get("version", "unknown")
    except Exception:
        return "unknown"


def _load_latency_config(path: str) -> dict | None:
    """Return the latency section from a ConfigMap YAML or plain sim-config JSON."""
    with open(path) as f:
        text = f.read()
    doc = yaml.safe_load(text)
    if isinstance(doc, dict) and doc.get("kind") == "ConfigMap":
        cfg = json.loads(doc["data"]["config.json"])
    else:
        cfg = doc if isinstance(doc, dict) else json.loads(text)
    return cfg.get("latency")


def _run_bench(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def _warmup_argv(point, *, base_url, model, tokenizer, result_dir, seed):
    warm = SweepPoint(
        isl=point.isl, osl=point.osl, concurrency=point.concurrency,
        num_prompts=max(1, point.concurrency)
    )
    return bench_argv(
        warm, base_url=base_url, model=model, tokenizer=tokenizer,
        result_dir=result_dir, result_filename="warmup.json", seed=seed,
    )


def run(
    *, real_url, sim_url, tokenizer, sweep_path, out_dir,
    seed=0, warmup=True, model_config=None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = load_sweep(sweep_path)
    print(f"sweep: {sweep_path} ({len(points)} points)")
    endpoints = [
        ("real", real_url, _detect_model(real_url)),
        ("sim",  sim_url,  _detect_model(sim_url)),
    ]
    for label, url, model in endpoints:
        print(f"{label}: {url}  model={model}")
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": endpoints[0][2],
        "tokenizer": tokenizer,
        "seed": seed,
        "real_url": real_url,
        "sim_url": sim_url,
        "vllm_version": {
            "real": _fetch_version(real_url),
            "sim": _fetch_version(sim_url),
        },
        "latency_config": _load_latency_config(model_config) if model_config else None,
    }
    results: list[PointResult] = []
    for pt in points:
        for label, url, model in endpoints:
            if warmup:
                print(f"\n[{pt.label}] warmup → {label} ({url})")
                _run_bench(_warmup_argv(
                    pt, base_url=url, model=model, tokenizer=tokenizer,
                    result_dir=str(out), seed=seed))
            fname = (
                f"c{pt.concurrency}-isl{pt.isl}-osl{pt.osl}-{label}.json"
            )
            print(f"\n[{pt.label}] bench → {label} ({url})")
            _run_bench(bench_argv(
                pt, base_url=url, model=model, tokenizer=tokenizer,
                result_dir=str(out), result_filename=fname, seed=seed))
        real = load_result(
            out / f"c{pt.concurrency}-isl{pt.isl}-osl{pt.osl}-real.json"
        )
        sim = load_result(
            out / f"c{pt.concurrency}-isl{pt.isl}-osl{pt.osl}-sim.json"
        )
        results.append(PointResult(
            label=pt.label,
            params={"isl": pt.isl, "osl": pt.osl, "concurrency": pt.concurrency},
            comparisons=compare_point(real, sim)))
    agg = aggregate(results)
    (out / "report.md").write_text(render_markdown(results, agg, meta=meta))
    (out / "report.json").write_text(
        json.dumps(render_json(results, agg, meta=meta), indent=2)
    )
    print(f"wrote {out / 'report.md'} and {out / 'report.json'}")
    print(f"overall median MAPE: {agg['overall'] * 100:.1f}%")


def _replace_beta(text: str, beta: list[float]) -> str:
    """Replace the "beta": [...] value in JSON text without reformatting other keys."""
    return re.sub(r'"beta":\s*\[[\s\S]*?\]', f'"beta": {json.dumps(beta)}', text)


def promote(*, deployment_dir: str, latency_model: str = "physics") -> None:
    """Promote a tuned beta into the k8s sim-config.json and configmap.yaml."""
    dep = Path(deployment_dir)
    tuned_path = dep / "evaluation" / "results" / latency_model / "tuned-sim-config.json"
    if not tuned_path.exists():
        raise FileNotFoundError(f"tuned config not found: {tuned_path}")

    with open(tuned_path) as f:
        tuned = json.load(f)
    beta = tuned["latency"]["beta"]
    print(f"promoting beta from {tuned_path}: {beta}")

    k8s = dep / "k8s"
    sim_cfg = k8s / "sim-config.json"
    sim_cfg.write_text(_replace_beta(sim_cfg.read_text(), beta))
    print(f"updated {sim_cfg}")

    cm_path = k8s / "configmap.yaml"
    cm_path.write_text(_replace_beta(cm_path.read_text(), beta))
    print(f"updated {cm_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sim-vs-real evaluation tools.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("eval", help="Run sim-vs-real benchmark sweep.")
    ep.add_argument("--real-url", required=True)
    ep.add_argument("--sim-url", required=True)
    ep.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    ep.add_argument(
        "--sweep", default=str(Path(__file__).parent / "sweep.yaml")
    )
    ep.add_argument("--out", default="eval-out")
    ep.add_argument("--seed", type=int, default=0)
    ep.add_argument("--no-warmup", action="store_true")
    ep.add_argument(
        "--model-config", default=None,
        help="Path to sim ConfigMap YAML or sim-config.json; included in the report.",
    )

    pp = sub.add_parser("promote", help="Promote tuned beta into k8s config files.")
    pp.add_argument(
        "--deployment-dir", required=True,
        help="Deployment directory (parent of k8s/ and evaluation/).",
    )
    pp.add_argument(
        "--latency-model", default="physics",
        help="Latency model variant whose tuned results to promote (default: physics).",
    )

    args = ap.parse_args(argv)
    if args.cmd == "eval":
        run(real_url=args.real_url, sim_url=args.sim_url,
            tokenizer=args.tokenizer, sweep_path=args.sweep, out_dir=args.out,
            seed=args.seed, warmup=not args.no_warmup,
            model_config=args.model_config)
    elif args.cmd == "promote":
        promote(deployment_dir=args.deployment_dir, latency_model=args.latency_model)


if __name__ == "__main__":
    main()
