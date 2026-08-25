# Sim-vs-Real Evaluation Report

## Environment

| | |
|---|---|
| Date | 2026-08-24 18:45:09 |
| Model | qwen3-32b |
| Tokenizer | Qwen/Qwen3-32B |
| Seed | 0 |
| vLLM (real) | 0.27.1 |
| vLLM (sim) | 0.27.1 |

## Latency Config

| Parameter | Value |
|---|---|
| Type | physics |
| TP | 1 |
| β\_pf | 0.152128 |
| β\_dc | 0.0 |
| β\_base | 126.024825 |
| Peak TFLOPs | 989.0 |
| HBM bandwidth | 3350.0 GB/s |
| Weight dtype | bfloat16 |

---

### ISL=1024 OSL=128 c=1

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 224.57 | 458.88 | 104.3% | +104.3% sim slower |
| TTFT median (ms) | 210.86 | 263.76 | 25.1% | +25.1% sim slower |
| TTFT p90 (ms) | 257.68 | 769.28 | 198.5% | +198.5% sim slower |
| TTFT p99 (ms) | 414.88 | 2063.67 | 397.4% | +397.4% sim slower |
| TPOT mean (ms) | 23.84 | 128.99 | 441.2% | +441.2% sim slower |
| ITL mean (ms) | 23.84 | 128.99 | 441.2% | +441.2% sim slower |
| ITL median (ms) | 23.80 | 129.01 | 442.0% | +442.0% sim slower |
| ITL p99 (ms) | 111.84 | 379.74 | 239.5% | +239.5% sim slower |
| E2E mean (ms) | 3251.67 | 16841.01 | 417.9% | +417.9% sim slower |
| E2E p99 (ms) | 3348.40 | 18442.91 | 450.8% | +450.8% sim slower |
| Output throughput (tok/s) | 39.35 | 7.60 | 80.7% | -80.7% sim lower |
| Request throughput (req/s) | 0.31 | 0.06 | 80.7% | -80.7% sim lower |

## Aggregate (median APE across points)

| Metric | Median APE |
|---|--:|
| TTFT mean | 104.3% |
| TTFT median | 25.1% |
| TTFT p90 | 198.5% |
| TTFT p99 | 397.4% |
| TPOT mean | 441.2% |
| ITL mean | 441.2% |
| ITL median | 442.0% |
| ITL p99 | 239.5% |
| E2E mean | 417.9% |
| E2E p99 | 450.8% |
| Output throughput | 80.7% |
| Request throughput | 80.7% |

**Overall median MAPE: 318.5%**
