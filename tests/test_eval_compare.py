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


from evaluation.compare import PointResult, aggregate, render_markdown, render_json


def _point(label):
    real = load_result(FIX / "real-c16.json")
    sim = load_result(FIX / "sim-c16.json")
    return PointResult(label=label, params={"isl": 1024, "osl": 128, "c": 16},
                       comparisons=compare_point(real, sim))


def test_aggregate_medians():
    pts = [_point("a"), _point("b")]
    agg = aggregate(pts)
    # identical points -> median equals the single-point APE
    assert agg["TTFT mean"] == pytest.approx(0.10)
    assert "overall" in agg
    assert agg["overall"] >= 0.0


def test_render_markdown_has_tables_and_signed_hint():
    pts = [_point("ISL=1024 OSL=128 c=16")]
    md = render_markdown(pts, aggregate(pts))
    assert "ISL=1024 OSL=128 c=16" in md
    assert "| TTFT mean" in md
    assert "%" in md
    assert "Aggregate" in md
    # signed hint mentions direction for TTFT/ITL
    assert "sim slower" in md or "sim faster" in md


def test_render_json_roundtrips():
    pts = [_point("p")]
    out = render_json(pts, aggregate(pts))
    assert out["aggregate"]["overall"] >= 0.0
    assert out["points"][0]["label"] == "p"
    assert out["points"][0]["metrics"][0]["ape"] >= 0.0
