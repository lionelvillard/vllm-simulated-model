import json
from pathlib import Path

from evaluation.metrics import METRICS, MetricComparison


def load_result(path):
    with open(Path(path)) as f:
        return json.load(f)


def compare_point(real: dict, sim: dict) -> list[MetricComparison]:
    comps: list[MetricComparison] = []
    for m in METRICS:
        if m.key not in real:
            raise KeyError(f"metric {m.key!r} missing from real result")
        if m.key not in sim:
            raise KeyError(f"metric {m.key!r} missing from sim result")
        r = float(real[m.key])
        s = float(sim[m.key])
        ape = abs(s - r) / r if r != 0 else float("inf")
        signed = (s - r) / r * 100.0 if r != 0 else float("inf")
        comps.append(
            MetricComparison(
                name=m.name, unit=m.unit, real=r, sim=s,
                ape=ape, signed_pct=signed, lower_is_better=m.lower_is_better,
            )
        )
    return comps
