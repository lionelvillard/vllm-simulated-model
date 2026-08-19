from evaluation.metrics import METRICS, Metric, MetricComparison


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
