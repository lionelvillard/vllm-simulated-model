import argparse
import json
import time
import urllib.request
from pathlib import Path

from scipy.optimize import minimize_scalar

from evaluation.compare import (
    MetricComparison,
    aggregate,
    compare_point,
    load_result,
    render_json,
    render_markdown,
    PointResult,
)
from evaluation.run_eval import (
    SweepPoint,
    _detect_model,
    _run_bench,
    bench_argv,
)

_TUNING_POINT = SweepPoint(
    isl=1024, osl=128, concurrency=1, num_prompts=32
)

_PF_BOUNDS = (0.05, 5.0)
_DC_BOUNDS = (0.05, 5.0)
_BASE_BOUNDS = (0.0, 200.0)


def _ttft_mape(comparisons: list[MetricComparison]) -> float:
    return next(c.ape for c in comparisons if c.name == "TTFT mean")


def _itl_mape(comparisons: list[MetricComparison]) -> float:
    return next(c.ape for c in comparisons if c.name == "ITL mean")


def _post_beta(sim_url: str, beta: list[float]) -> None:
    body = json.dumps({"beta": beta}).encode()
    req = urllib.request.Request(
        f"{sim_url}/sim/config",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"POST /sim/config returned {resp.status}: {resp.read()}"
            )


def _make_bench_sim(
    *,
    sim_url: str,
    sim_model: str,
    tokenizer: str,
    real_result: dict,
    out_dir: Path,
    seed: int,
):
    counter = [0]

    def bench_sim(beta: list[float]) -> list[MetricComparison]:
        counter[0] += 1
        n = counter[0]
        _post_beta(sim_url, beta)
        time.sleep(0.5)  # barrier: let any in-flight requests drain
        fname = f"sim-tune-{n}.json"
        argv = bench_argv(
            _TUNING_POINT,
            base_url=sim_url,
            model=sim_model,
            tokenizer=tokenizer,
            result_dir=str(out_dir),
            result_filename=fname,
            seed=seed,
        )
        _run_bench(argv)
        sim_result = load_result(out_dir / fname)
        return compare_point(real_result, sim_result)

    return bench_sim


def _coordinate_search(
    *,
    bench_sim,
    bench_sim_overall=None,  # override for testing
) -> dict:
    history = {"phase1": [], "phase2": [], "phase3": []}

    # Phase 1: tune beta_pf — minimize TTFT mean APE
    def f_pf(b):
        comps = bench_sim([b, 1.0, 0.0])
        mape = _ttft_mape(comps)
        history["phase1"].append({"beta": [b, 1.0, 0.0], "mape": mape})
        return mape

    res_pf = minimize_scalar(f_pf, bounds=_PF_BOUNDS, method="bounded")
    best_pf = res_pf.x

    # Phase 2: tune beta_dc — minimize ITL mean APE
    def f_dc(b):
        comps = bench_sim([best_pf, b, 0.0])
        mape = _itl_mape(comps)
        history["phase2"].append({"beta": [best_pf, b, 0.0], "mape": mape})
        return mape

    res_dc = minimize_scalar(f_dc, bounds=_DC_BOUNDS, method="bounded")
    best_dc = res_dc.x

    # Phase 3: tune beta_base — minimize overall median MAPE
    def f_base(b):
        if bench_sim_overall is not None:
            mape = bench_sim_overall([best_pf, best_dc, b])
        else:
            comps = bench_sim([best_pf, best_dc, b])
            pt = PointResult(
                label="tune",
                params={},
                comparisons=comps,
            )
            mape = aggregate([pt])["overall"]
        history["phase3"].append(
            {"beta": [best_pf, best_dc, b], "mape": mape}
        )
        return mape

    res_base = minimize_scalar(f_base, bounds=_BASE_BOUNDS, method="bounded")
    best_base = res_base.x

    return {
        "beta": [best_pf, best_dc, best_base],
        "history": history,
    }


def _write_tuned_config(
    *, model_config: str, beta: list[float], out_dir: str
) -> None:
    with open(model_config) as f:
        cfg = json.load(f)
    cfg["latency"]["beta"] = beta
    out = Path(out_dir) / "tuned-sim-config.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")


def _write_tuning_report(
    *,
    history: dict,
    best_beta: list[float],
    real_result: dict,
    final_sim_result: dict,
    out_dir: Path,
) -> None:
    comps = compare_point(real_result, final_sim_result)
    pt = PointResult(
        label=f"ISL=1024 OSL=128 c=1 (tuned beta={best_beta})",
        params={"isl": 1024, "osl": 128, "concurrency": 1},
        comparisons=comps,
    )
    agg = aggregate([pt])

    lines = ["# Auto-Tuning Report", ""]
    lines.append(f"**Tuned beta:** `{best_beta}`")
    lines.append("")
    for phase, name in [("phase1", "beta_pf"), ("phase2", "beta_dc"), ("phase3", "beta_base")]:
        lines.append(f"## Phase: {name}")
        lines.append("")
        lines.append("| beta | MAPE |")
        lines.append("|---:|---:|")
        for step in history[phase]:
            lines.append(
                f"| {step['beta']} | {step['mape'] * 100:.2f}% |"
            )
        lines.append("")

    lines.append("## Final comparison at tuned beta")
    lines.append("")
    lines += render_markdown([pt], agg).splitlines()

    (out_dir / "tuning-report.md").write_text("\n".join(lines))
    report_json = {
        "best_beta": best_beta,
        "history": history,
        "final_comparison": render_json([pt], agg),
    }
    (out_dir / "tuning-report.json").write_text(
        json.dumps(report_json, indent=2)
    )
    print(f"wrote {out_dir / 'tuning-report.md'} and {out_dir / 'tuning-report.json'}")


def tune(
    *,
    real_url: str,
    sim_url: str,
    model_config: str,
    tokenizer: str,
    out_dir: str,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    real_model = _detect_model(real_url)
    sim_model = _detect_model(sim_url)
    print(f"real: {real_url}  model={real_model}")
    print(f"sim:  {sim_url}  model={sim_model}")

    # Run real benchmark once
    real_fname = "real.json"
    _run_bench(
        bench_argv(
            _TUNING_POINT,
            base_url=real_url,
            model=real_model,
            tokenizer=tokenizer,
            result_dir=str(out),
            result_filename=real_fname,
            seed=seed,
        )
    )
    real_result = load_result(out / real_fname)
    print("real benchmark done")

    bench_sim = _make_bench_sim(
        sim_url=sim_url,
        sim_model=sim_model,
        tokenizer=tokenizer,
        real_result=real_result,
        out_dir=out,
        seed=seed,
    )

    result = _coordinate_search(bench_sim=bench_sim)
    best_beta = [round(b, 6) for b in result["beta"]]
    print(f"tuned beta: {best_beta}")

    # Run one final benchmark at the tuned beta to record the final comparison
    _post_beta(sim_url, best_beta)
    time.sleep(0.5)
    final_fname = "sim-final.json"
    _run_bench(
        bench_argv(
            _TUNING_POINT,
            base_url=sim_url,
            model=sim_model,
            tokenizer=tokenizer,
            result_dir=str(out),
            result_filename=final_fname,
            seed=seed,
        )
    )
    final_sim_result = load_result(out / final_fname)

    _write_tuned_config(model_config=model_config, beta=best_beta, out_dir=str(out))
    _write_tuning_report(
        history=result["history"],
        best_beta=best_beta,
        real_result=real_result,
        final_sim_result=final_sim_result,
        out_dir=out,
    )
    return {"beta": best_beta}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Auto-tune physics beta parameters via coordinate search."
    )
    ap.add_argument("--real-url", required=True)
    ap.add_argument("--sim-url", required=True)
    ap.add_argument(
        "--model-config", required=True,
        help="Path to the sim-config.json whose beta will be tuned.",
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", default="tune-out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tune(
        real_url=args.real_url,
        sim_url=args.sim_url,
        model_config=args.model_config,
        tokenizer=args.tokenizer,
        out_dir=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
