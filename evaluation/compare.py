import json
from pathlib import Path
from dataclasses import dataclass
from statistics import median

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


@dataclass(frozen=True)
class PointResult:
    label: str
    params: dict
    comparisons: list  # list[MetricComparison]


def aggregate(points) -> dict:
    per_metric: dict[str, list[float]] = {}
    for p in points:
        for c in p.comparisons:
            per_metric.setdefault(c.name, []).append(c.ape)
    agg = {name: median(vals) for name, vals in per_metric.items()}
    agg["overall"] = median(agg.values()) if agg else 0.0
    return agg


def _signed_hint(c) -> str:
    if c.unit != "ms":
        direction = "sim higher" if c.signed_pct > 0 else "sim lower"
        return f"{direction}"
    if c.signed_pct > 0:
        return "sim slower"
    return "sim faster"


def _render_meta_markdown(meta: dict) -> list[str]:
    lines: list[str] = []
    lines += ["## Environment", ""]
    lines += ["| | |", "|---|---|"]
    lines.append(f"| Date | {meta['date']} |")
    lines.append(f"| Model | {meta['model']} |")
    lines.append(f"| Tokenizer | {meta['tokenizer']} |")
    lines.append(f"| Seed | {meta['seed']} |")
    ver = meta.get("vllm_version", {})
    lines.append(f"| vLLM (real) | {ver.get('real', 'unknown')} |")
    lines.append(f"| vLLM (sim) | {ver.get('sim', 'unknown')} |")
    lines.append("")
    lc = meta.get("latency_config")
    if lc:
        lines += ["## Latency Config", ""]
        lines += ["| Parameter | Value |", "|---|---|"]
        lines.append(f"| Type | {lc.get('type', '—')} |")
        lines.append(f"| TP | {lc.get('tp', 1)} |")
        if "beta" in lc:
            beta = lc["beta"]
            lines.append(f"| β\\_pf | {beta[0]} |")
            lines.append(f"| β\\_dc | {beta[1]} |")
            lines.append(f"| β\\_base | {beta[2]} |")
        hw = lc.get("hardware", {})
        if hw:
            lines.append(f"| Peak TFLOPs | {hw.get('peak_tflops', '—')} |")
            lines.append(f"| HBM bandwidth | {hw.get('hbm_gbps', '—')} GB/s |")
            lines.append(f"| Weight dtype | {hw.get('weight_dtype', '—')} |")
        lines.append("")
    lines += ["---", ""]
    return lines


def render_markdown(points, agg, *, meta=None) -> str:
    lines: list[str] = ["# Sim-vs-Real Evaluation Report", ""]
    if meta:
        lines += _render_meta_markdown(meta)
    for p in points:
        lines.append(f"### {p.label}")
        lines.append("")
        lines.append("| Metric | Real | Sim | APE | Signed |")
        lines.append("|---|--:|--:|--:|:--|")
        for c in p.comparisons:
            lines.append(
                f"| {c.name} ({c.unit}) | {c.real:.2f} | {c.sim:.2f} "
                f"| {c.ape * 100:.1f}% | {c.signed_pct:+.1f}% {_signed_hint(c)} |"
            )
        lines.append("")
    lines.append("## Aggregate (median APE across points)")
    lines.append("")
    lines.append("| Metric | Median APE |")
    lines.append("|---|--:|")
    for name, val in agg.items():
        if name == "overall":
            continue
        lines.append(f"| {name} | {val * 100:.1f}% |")
    lines.append("")
    lines.append(f"**Overall median MAPE: {agg['overall'] * 100:.1f}%**")
    lines.append("")
    return "\n".join(lines)


def render_json(points, agg, *, meta=None) -> dict:
    out = {}
    if meta:
        out["meta"] = meta
    out["aggregate"] = agg
    out["points"] = [
        {
            "label": p.label,
            "params": p.params,
            "metrics": [
                {
                    "name": c.name, "unit": c.unit, "real": c.real,
                    "sim": c.sim, "ape": c.ape, "signed_pct": c.signed_pct,
                }
                for c in p.comparisons
            ],
        }
        for p in points
    ]
    return out
