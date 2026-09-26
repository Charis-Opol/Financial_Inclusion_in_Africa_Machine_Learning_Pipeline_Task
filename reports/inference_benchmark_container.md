# Inference latency: native XGBoost vs ONNX Runtime

Model `xgb-cw-5ec2d564be13` · 2000 timed runs per cell after 200 warmup · 1 thread(s) per runtime · latencies in microseconds (µs)

| Batch | Runtime | Mean ± 95% CI | p50 | p95 | p99 | Max | Throughput (rows/s) | p50 speed-up vs native |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | native xgboost | 574.9 ± 7.6 | 519.4 | 884.9 | 1,151.6 | 3,620.4 | 1,739 | 1.00× (baseline) |
| 1 | onnx runtime | 41.9 ± 1.2 | 33.2 | 78.4 | 188.4 | 300.4 | 23,859 | 15.65× |
| 32 | native xgboost | 888.1 ± 121.8 | 678.6 | 1,110.2 | 1,454.1 | 54,092.1 | 36,032 | 1.00× (baseline) |
| 32 | onnx runtime | 512.6 ± 94.4 | 399.1 | 522.3 | 702.1 | 49,619.2 | 62,431 | 1.70× |
| 1024 | native xgboost | 5,491.6 ± 71.2 | 5,182.1 | 8,240.2 | 10,854.0 | 38,573.7 | 186,466 | 1.00× (baseline) |
| 1024 | onnx runtime | 11,792.4 ± 198.2 | 11,378.7 | 15,574.7 | 24,147.3 | 65,888.9 | 86,836 | 0.46× |

Shared input encoding (1 record, numpy): p50 12.3 µs, p99 53.8 µs -- paid identically by both runtimes in the API.

Environment: Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.41 · AMD Ryzen 7 5800U with Radeon Graphics (16 logical CPUs) · Python 3.13.15 · xgboost 3.3.0 · onnxruntime 1.29.0
