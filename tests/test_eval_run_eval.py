# tests/test_eval_run_eval.py
from pathlib import Path

from evaluation.run_eval import SweepPoint, load_sweep, bench_argv


def test_load_sweep(tmp_path):
    p = tmp_path / "sweep.yaml"
    p.write_text(
        "points:\n"
        "  - {isl: 1024, osl: 128, concurrency: 16, num_prompts: 256}\n"
        "  - {isl: 256, osl: 256, concurrency: 32, num_prompts: 256}\n"
    )
    pts = load_sweep(p)
    assert len(pts) == 2
    assert pts[0] == SweepPoint(isl=1024, osl=128, concurrency=16, num_prompts=256)
    assert pts[0].label == "ISL=1024 OSL=128 c=16"


def test_bench_argv_enforces_fairness_flags():
    pt = SweepPoint(isl=1024, osl=128, concurrency=16, num_prompts=256)
    argv = bench_argv(
        pt, base_url="http://localhost:9001", model="qwen3-32b",
        tokenizer="Qwen/Qwen3-32B", result_dir="/tmp/out",
        result_filename="c16-sim.json",
    )
    joined = " ".join(argv)
    assert argv[:3] == ["vllm", "bench", "serve"]
    assert "--ignore-eos" in argv
    assert "--seed 0" in joined
    assert "--random-input-len 1024" in joined
    assert "--random-output-len 128" in joined
    assert "--max-concurrency 16" in joined
    assert "--num-prompts 256" in joined
    assert "--model qwen3-32b" in joined
    assert "--tokenizer Qwen/Qwen3-32B" in joined
    assert "--percentile-metrics ttft,tpot,itl,e2el" in joined
    assert "--metric-percentiles 90,99" in joined
    assert "--save-result" in argv
    assert "--result-filename c16-sim.json" in joined
    assert "--base-url http://localhost:9001" in joined
