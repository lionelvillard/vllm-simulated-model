from pathlib import Path
import pytest

from evaluation.metrics import METRICS, Metric, MetricComparison
from evaluation.compare import load_result, compare_point


def test_metrics_registry_covers_spec():
    keys = {m.key for m in METRICS}
    # spec §7 report rows
    assert {
        "mean_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
        "mean_itl_ms", "mean_tpot_ms",
        "mean_e2el_ms", "p99_e2el_ms",
        "output_throughput", "request_throughput",
    } <= keys
    # every metric is well-formed
    for m in METRICS:
        assert isinstance(m, Metric)
        assert m.unit in {"ms", "tok/s", "req/s"}


def test_metric_comparison_is_frozen():
    c = MetricComparison(
        name="TTFT mean", unit="ms", real=100.0, sim=110.0,
        ape=0.1, signed_pct=10.0, lower_is_better=True,
    )
    assert c.real == 100.0


FIX = Path(__file__).parent / "fixtures" / "eval"


def test_compare_point_computes_ape_and_sign():
    real = load_result(FIX / "real-c16.json")
    sim = load_result(FIX / "sim-c16.json")
    comps = {c.name: c for c in compare_point(real, sim)}
    # TTFT mean: |110-100|/100 = 0.10 ; signed +10%
    assert comps["TTFT mean"].ape == pytest.approx(0.10)
    assert comps["TTFT mean"].signed_pct == pytest.approx(10.0)
    # ITL mean: |19-20|/20 = 0.05 ; signed -5%
    assert comps["ITL mean"].ape == pytest.approx(0.05)
    assert comps["ITL mean"].signed_pct == pytest.approx(-5.0)
    # throughput present
    assert comps["Output throughput"].ape == pytest.approx(0.10)


def test_compare_point_missing_key_raises():
    real = load_result(FIX / "real-c16.json")
    sim = load_result(FIX / "sim-c16.json")
    del sim["mean_ttft_ms"]
    with pytest.raises(KeyError):
        compare_point(real, sim)
