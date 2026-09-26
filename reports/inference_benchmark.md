# Inference latency: native XGBoost vs ONNX Runtime

Model `xgb-cw-5ec2d564be13` · 2000 timed runs per cell after 200 warmup · 1 thread(s) per runtime · latencies in microseconds (µs)

| Batch | Runtime | Mean ± 95% CI | p50 | p95 | p99 | Max | Throughput (rows/s) | p50 speed-up vs native |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | native xgboost | 758.2 ± 13.2 | 648.3 | 1,229.7 | 1,962.6 | 5,214.6 | 1,319 | 1.00× (baseline) |
| 1 | onnx runtime | 70.9 ± 2.2 | 54.5 | 124.2 | 282.5 | 712.5 | 14,113 | 11.90× |
| 32 | native xgboost | 1,073.1 ± 20.9 | 969.1 | 1,544.8 | 1,987.5 | 17,779.4 | 29,821 | 1.00× (baseline) |
| 32 | onnx runtime | 519.7 ± 4.1 | 488.8 | 690.6 | 864.0 | 1,323.8 | 61,576 | 1.98× |
| 1024 | native xgboost | 7,910.4 ± 60.7 | 7,452.2 | 10,883.7 | 12,682.5 | 20,078.0 | 129,449 | 1.00× (baseline) |
| 1024 | onnx runtime | 14,158.7 ± 123.5 | 13,478.4 | 17,046.3 | 24,305.5 | 78,753.0 | 72,323 | 0.55× |

Shared input encoding (1 record, numpy): p50 12.8 µs, p99 46.7 µs -- paid identically by both runtimes in the API.

Environment: Windows-11 · AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD (16 logical CPUs) · Python 3.13.7 · xgboost 3.3.0 · onnxruntime 1.29.0
