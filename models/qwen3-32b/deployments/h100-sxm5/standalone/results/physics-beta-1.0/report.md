# Sim-vs-Real Evaluation Report

### ISL=1024 OSL=128 c=1

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 207.33 | 175.78 | 15.2% | -15.2% sim faster |
| TTFT median (ms) | 207.72 | 175.70 | 15.4% | -15.4% sim faster |
| TTFT p90 (ms) | 221.15 | 186.32 | 15.8% | -15.8% sim faster |
| TTFT p99 (ms) | 232.65 | 188.83 | 18.8% | -18.8% sim faster |
| TPOT mean (ms) | 23.88 | 21.01 | 12.0% | -12.0% sim faster |
| ITL mean (ms) | 23.88 | 21.01 | 12.0% | -12.0% sim faster |
| ITL median (ms) | 23.98 | 21.00 | 12.4% | -12.4% sim faster |
| ITL p99 (ms) | 30.62 | 28.03 | 8.5% | -8.5% sim faster |
| E2E mean (ms) | 3240.37 | 2844.46 | 12.2% | -12.2% sim faster |
| E2E p99 (ms) | 3265.19 | 2893.78 | 11.4% | -11.4% sim faster |
| Output throughput (tok/s) | 39.49 | 44.99 | 13.9% | +13.9% sim higher |
| Request throughput (req/s) | 0.31 | 0.35 | 13.9% | +13.9% sim higher |

### ISL=1024 OSL=128 c=16

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 1149.52 | 1005.27 | 12.5% | -12.5% sim faster |
| TTFT median (ms) | 1155.97 | 418.14 | 63.8% | -63.8% sim faster |
| TTFT p90 (ms) | 1708.46 | 4329.66 | 153.4% | +153.4% sim slower |
| TTFT p99 (ms) | 1829.87 | 5234.41 | 186.1% | +186.1% sim slower |
| TPOT mean (ms) | 29.27 | 44.29 | 51.3% | +51.3% sim slower |
| ITL mean (ms) | 29.27 | 44.29 | 51.3% | +51.3% sim slower |
| ITL median (ms) | 25.34 | 38.87 | 53.4% | +53.4% sim slower |
| ITL p99 (ms) | 228.37 | 108.29 | 52.6% | -52.6% sim faster |
| E2E mean (ms) | 4866.91 | 6630.00 | 36.2% | +36.2% sim slower |
| E2E p99 (ms) | 5086.57 | 10968.52 | 115.6% | +115.6% sim slower |
| Output throughput (tok/s) | 420.52 | 302.25 | 28.1% | -28.1% sim lower |
| Request throughput (req/s) | 3.29 | 2.36 | 28.1% | -28.1% sim lower |

### ISL=1024 OSL=128 c=64

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 10880.62 | 19834.43 | 82.3% | +82.3% sim slower |
| TTFT median (ms) | 12323.04 | 21903.01 | 77.7% | +77.7% sim slower |
| TTFT p90 (ms) | 13252.53 | 22929.12 | 73.0% | +73.0% sim slower |
| TTFT p99 (ms) | 13705.71 | 24129.16 | 76.1% | +76.1% sim slower |
| TPOT mean (ms) | 36.98 | 44.87 | 21.4% | +21.4% sim slower |
| ITL mean (ms) | 36.98 | 44.87 | 21.4% | +21.4% sim slower |
| ITL median (ms) | 26.76 | 38.60 | 44.3% | +44.3% sim slower |
| ITL p99 (ms) | 618.47 | 138.22 | 77.7% | -77.7% sim faster |
| E2E mean (ms) | 15576.89 | 25533.44 | 63.9% | +63.9% sim slower |
| E2E p99 (ms) | 18005.99 | 29875.26 | 65.9% | +65.9% sim slower |
| Output throughput (tok/s) | 495.50 | 304.06 | 38.6% | -38.6% sim lower |
| Request throughput (req/s) | 3.87 | 2.38 | 38.6% | -38.6% sim lower |

### ISL=256 OSL=256 c=32

| Metric | Real | Sim | APE | Signed |
|---|--:|--:|--:|:--|
| TTFT mean (ms) | 411.48 | 430.55 | 4.6% | +4.6% sim slower |
| TTFT median (ms) | 374.64 | 401.09 | 7.1% | +7.1% sim slower |
| TTFT p90 (ms) | 648.14 | 750.14 | 15.7% | +15.7% sim slower |
| TTFT p99 (ms) | 888.76 | 944.09 | 6.2% | +6.2% sim slower |
| TPOT mean (ms) | 27.18 | 61.62 | 126.7% | +126.7% sim slower |
| ITL mean (ms) | 27.18 | 61.62 | 126.7% | +126.7% sim slower |
| ITL median (ms) | 24.66 | 60.67 | 146.0% | +146.0% sim slower |
| ITL p99 (ms) | 163.92 | 178.55 | 8.9% | +8.9% sim slower |
| E2E mean (ms) | 7342.51 | 16142.62 | 119.9% | +119.9% sim slower |
| E2E p99 (ms) | 7868.32 | 16494.06 | 109.6% | +109.6% sim slower |
| Output throughput (tok/s) | 1113.54 | 498.90 | 55.2% | -55.2% sim lower |
| Request throughput (req/s) | 4.35 | 1.95 | 55.2% | -55.2% sim lower |

## Aggregate (median APE across points)

| Metric | Median APE |
|---|--:|
| TTFT mean | 13.9% |
| TTFT median | 39.6% |
| TTFT p90 | 44.4% |
| TTFT p99 | 47.4% |
| TPOT mean | 36.3% |
| ITL mean | 36.3% |
| ITL median | 48.8% |
| ITL p99 | 30.8% |
| E2E mean | 50.1% |
| E2E p99 | 87.8% |
| Output throughput | 33.4% |
| Request throughput | 33.4% |

**Overall median MAPE: 38.0%**
