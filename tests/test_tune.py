import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluation.compare import MetricComparison, PointResult


def _make_comparison(ttft_ape, itl_ape):
    """Build a minimal list of MetricComparison objects for mocking."""
    return [
        MetricComparison(
            name="TTFT mean", unit="ms", real=100.0,
            sim=100.0 * (1 + ttft_ape), ape=abs(ttft_ape),
            signed_pct=ttft_ape * 100, lower_is_better=True,
        ),
        MetricComparison(
            name="ITL mean", unit="ms", real=20.0,
            sim=20.0 * (1 + itl_ape), ape=abs(itl_ape),
            signed_pct=itl_ape * 100, lower_is_better=True,
        ),
    ]


def test_extract_ttft_mape():
    from evaluation.tune import _ttft_mape, _itl_mape

    comps = _make_comparison(ttft_ape=0.12, itl_ape=0.05)
    assert _ttft_mape(comps) == pytest.approx(0.12)
    assert _itl_mape(comps) == pytest.approx(0.05)


def test_coordinate_search_finds_minimum(tmp_path):
    """Coordinate search must find beta values that minimize each phase's objective."""
    from evaluation.tune import _coordinate_search

    calls = []

    def fake_bench_sim(beta):
        calls.append(list(beta))
        b_pf, b_dc, b_base = beta
        # True minimum: beta_pf=0.3, beta_dc=0.7, beta_base=8.0
        ttft_ape = abs(b_pf - 0.3)
        itl_ape = abs(b_dc - 0.7)
        overall = (abs(b_pf - 0.3) + abs(b_dc - 0.7) + abs(b_base - 8.0)) / 3
        return _make_comparison(ttft_ape, itl_ape)

    def fake_overall(beta):
        b_pf, b_dc, b_base = beta
        return abs(b_pf - 0.3) + abs(b_dc - 0.7) + abs(b_base - 8.0)

    result = _coordinate_search(
        bench_sim=fake_bench_sim,
        bench_sim_overall=fake_overall,
    )

    assert result["beta"][0] == pytest.approx(0.3, abs=0.02)
    assert result["beta"][1] == pytest.approx(0.7, abs=0.02)
    assert result["beta"][2] == pytest.approx(8.0, abs=0.5)


def test_write_tuned_config(tmp_path):
    from evaluation.tune import _write_tuned_config

    original_cfg = {
        "architectures": ["SimulatedForCausalLM"],
        "latency": {
            "type": "physics",
            "hardware": {"peak_tflops": 989.0, "hbm_gbps": 3350.0, "weight_dtype": "bfloat16"},
            "beta": [1.0, 1.0, 0.0],
            "tp": 1,
            "deterministic_length": True,
        },
    }
    model_config_path = tmp_path / "sim-config.json"
    model_config_path.write_text(json.dumps(original_cfg))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _write_tuned_config(
        model_config=str(model_config_path),
        beta=[0.15, 0.92, 6.3],
        out_dir=str(out_dir),
    )

    result = json.loads((out_dir / "tuned-sim-config.json").read_text())
    assert result["latency"]["beta"] == [0.15, 0.92, 6.3]
    # Architecture block must be preserved
    assert result["architectures"] == ["SimulatedForCausalLM"]
    assert result["latency"]["tp"] == 1
