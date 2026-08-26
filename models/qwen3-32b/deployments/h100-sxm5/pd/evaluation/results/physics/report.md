# Sim-vs-Real Evaluation Report

## Environment

| | |
|---|---|
| Date | 2026-08-26 10:56:18 |
| Model | qwen3-32b |
| Tokenizer | Qwen/Qwen3-32B |
| Seed | 0 |
| vLLM (real) | 0.28.0 |
| vLLM (sim) | 0.27.0 |

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
| TTFT mean (ms) | 116.37 | 163.42 | 40.4% | +40.4% sim slower |
| TTFT median (ms) | 118.33 | 156.45 | 32.2% | +32.2% sim slower |
| TTFT p90 (ms) | 121.36 | 161.55 | 33.1% | +33.1% sim slower |
| TTFT p99 (ms) | 126.16 | 302.56 | 139.8% | +139.8% sim slower |
| TPOT mean (ms) | 23.81 | 130.43 | 447.8% | +447.8% sim slower |
| ITL mean (ms) | 23.81 | 130.43 | 447.8% | +447.8% sim slower |
| ITL median (ms) | 23.81 | 130.39 | 447.5% | +447.5% sim slower |
| ITL p99 (ms) | 39.88 | 136.82 | 243.1% | +243.1% sim slower |
| E2E mean (ms) | 3140.15 | 16728.23 | 432.7% | +432.7% sim slower |
| E2E p99 (ms) | 3150.28 | 16907.26 | 436.7% | +436.7% sim slower |
| Output throughput (tok/s) | 40.75 | 7.65 | 81.2% | -81.2% sim lower |
| Request throughput (req/s) | 0.32 | 0.06 | 81.2% | -81.2% sim lower |

### ISL=1024 OSL=128 c=16

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 1019.10 | 2929.76 | 187.5% | +187.5% sim slower |
| TTFT median (ms) | 1031.17 | 500.27 | 51.5% | -51.5% sim faster |
| TTFT p90 (ms) | 1608.11 | 15813.71 | 883.4% | +883.4% sim slower |
| TTFT p99 (ms) | 2153.13 | 18585.91 | 763.2% | +763.2% sim slower |
| TPOT mean (ms) | 29.80 | 147.21 | 393.9% | +393.9% sim slower |
| ITL mean (ms) | 29.80 | 147.21 | 393.9% | +393.9% sim slower |
| ITL median (ms) | 25.55 | 146.62 | 474.0% | +474.0% sim slower |
| ITL p99 (ms) | 214.97 | 291.58 | 35.6% | +35.6% sim slower |
| E2E mean (ms) | 4804.10 | 21625.16 | 350.1% | +350.1% sim slower |
| E2E p99 (ms) | 5112.60 | 37376.80 | 631.1% | +631.1% sim slower |
| Output throughput (tok/s) | 425.96 | 91.53 | 78.5% | -78.5% sim lower |
| Request throughput (req/s) | 3.33 | 0.72 | 78.5% | -78.5% sim lower |

### ISL=1024 OSL=128 c=64

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 10317.65 | 63098.66 | 511.6% | +511.6% sim slower |
| TTFT median (ms) | 11184.02 | 73251.38 | 555.0% | +555.0% sim slower |
| TTFT p90 (ms) | 12621.50 | 74976.46 | 494.0% | +494.0% sim slower |
| TTFT p99 (ms) | 13287.74 | 75546.09 | 468.5% | +468.5% sim slower |
| TPOT mean (ms) | 41.98 | 146.16 | 248.2% | +248.2% sim slower |
| ITL mean (ms) | 41.98 | 146.16 | 248.2% | +248.2% sim slower |
| ITL median (ms) | 27.45 | 145.51 | 430.1% | +430.1% sim slower |
| ITL p99 (ms) | 404.94 | 163.60 | 59.6% | -59.6% sim faster |
| E2E mean (ms) | 15648.95 | 81661.47 | 421.8% | +421.8% sim slower |
| E2E p99 (ms) | 19021.15 | 94036.03 | 394.4% | +394.4% sim slower |
| Output throughput (tok/s) | 496.90 | 94.53 | 81.0% | -81.0% sim lower |
| Request throughput (req/s) | 3.88 | 0.74 | 81.0% | -81.0% sim lower |

### ISL=256 OSL=256 c=32

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 602.81 | 378.94 | 37.1% | -37.1% sim faster |
| TTFT median (ms) | 790.40 | 355.38 | 55.0% | -55.0% sim faster |
| TTFT p90 (ms) | 838.24 | 508.72 | 39.3% | -39.3% sim faster |
| TTFT p99 (ms) | 842.58 | 690.74 | 18.0% | -18.0% sim faster |
| TPOT mean (ms) | 26.62 | 167.14 | 527.8% | +527.8% sim slower |
| ITL mean (ms) | 26.62 | 167.14 | 527.8% | +527.8% sim slower |
| ITL median (ms) | 26.05 | 166.94 | 540.9% | +540.9% sim slower |
| ITL p99 (ms) | 49.89 | 183.31 | 267.4% | +267.4% sim slower |
| E2E mean (ms) | 7391.64 | 43000.70 | 481.7% | +481.7% sim slower |
| E2E p99 (ms) | 7500.47 | 43936.61 | 485.8% | +485.8% sim slower |
| Output throughput (tok/s) | 1107.71 | 182.19 | 83.6% | -83.6% sim lower |
| Request throughput (req/s) | 4.33 | 0.71 | 83.6% | -83.6% sim lower |

## Aggregate (median APE across points)

| Metric | Median APE |
|---|--:|
| TTFT mean | 114.0% |
| TTFT median | 53.3% |
| TTFT p90 | 266.7% |
| TTFT p99 | 304.2% |
| TPOT mean | 420.9% |
| ITL mean | 420.9% |
| ITL median | 460.7% |
| ITL p99 | 151.3% |
| E2E mean | 427.3% |
| E2E p99 | 461.2% |
| Output throughput | 81.1% |
| Request throughput | 81.1% |

**Overall median MAPE: 285.4%**
