from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str          # human label, e.g. "TTFT mean"
    key: str           # vllm bench serve JSON key, e.g. "mean_ttft_ms"
    unit: str          # "ms" | "tok/s" | "req/s"
    lower_is_better: bool


@dataclass(frozen=True)
class MetricComparison:
    name: str
    unit: str
    real: float
    sim: float
    ape: float         # |sim - real| / real, as a fraction (0.1 == 10%)
    signed_pct: float  # (sim - real) / real * 100, sign preserved
    lower_is_better: bool


METRICS: list[Metric] = [
    Metric("TTFT mean", "mean_ttft_ms", "ms", True),
    Metric("TTFT median", "median_ttft_ms", "ms", True),
    Metric("TTFT p90", "p90_ttft_ms", "ms", True),
    Metric("TTFT p99", "p99_ttft_ms", "ms", True),
    Metric("TPOT mean", "mean_tpot_ms", "ms", True),
    Metric("ITL mean", "mean_itl_ms", "ms", True),
    Metric("ITL median", "median_itl_ms", "ms", True),
    Metric("ITL p99", "p99_itl_ms", "ms", True),
    Metric("E2E mean", "mean_e2el_ms", "ms", True),
    Metric("E2E p99", "p99_e2el_ms", "ms", True),
    Metric("Output throughput", "output_throughput", "tok/s", False),
    Metric("Request throughput", "request_throughput", "req/s", False),
]
