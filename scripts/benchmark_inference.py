"""Latency benchmark: native XGBoost vs ONNX Runtime on the production model.

    python scripts/benchmark_inference.py [--runs 2000] [--threads 1]

Writes reports/inference_benchmark.md (drop-in table) and .csv (raw stats).

Method, and why:
  - Model call only, on pre-encoded float32 inputs drawn from real training
    rows. Encoding is shared by both runtimes, so it is timed separately
    rather than diluting the comparison.
  - "Native" is exactly what the API does: `Booster.inplace_predict` with
    `validate_features=False` -- not the slower sklearn wrapper, which would
    flatter ONNX.
  - Same thread count for both (default 1, matching the service's
    INFERENCE_THREADS and the container's 1-CPU limit).
  - Warmup calls first (lazy allocations, caches, ORT's first-run setup).
  - The two runtimes alternate in blocks of BLOCK_SIZE calls (ABBA order),
    so drift over the run (boost clocks, thermals, background load) hits
    both equally. Blocks rather than call-by-call alternation: alternating
    every call makes each runtime evict the other's trees from CPU cache,
    which penalizes both and matches no real deployment (a service runs
    one runtime). Measured: call-by-call inflated ONNX batch-1 p50 ~1.9x.
  - A different input batch each iteration, so we're not timing one row
    sitting hot in cache.
  - GC disabled while timing so collection pauses don't land in one sample.
  - Reported: mean with 95% CI, p50/p95/p99 and max. Max is shown so
    one-off system stalls are visible rather than silently inflating the
    mean -- when mean and p50 disagree wildly, trust p50/p99. With the default 2000 runs,
    p99 rests on 20 samples above it -- enough to be meaningful, which is
    why --runs is floored at 100 (p99 would rest on a single sample).
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import xgboost

from fin_inclusion.serving.predictor import Predictor
from fin_inclusion.serving.raw_records import read_raw_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZES = (1, 32, 1024)  # single online request, small micro-batch, bulk scoring
BLOCK_SIZE = 50


@dataclass(frozen=True)
class LatencyStats:
    runtime: str
    batch_size: int
    runs: int
    mean_us: float
    ci95_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    max_us: float
    rows_per_sec: float


def summarize(runtime: str, batch_size: int, samples_ns: list[int]) -> LatencyStats:
    us = np.asarray(samples_ns, dtype=np.float64) / 1e3
    mean = float(us.mean())
    return LatencyStats(
        runtime=runtime,
        batch_size=batch_size,
        runs=len(us),
        mean_us=mean,
        ci95_us=float(1.96 * us.std(ddof=1) / np.sqrt(len(us))),
        p50_us=float(np.percentile(us, 50)),
        p95_us=float(np.percentile(us, 95)),
        p99_us=float(np.percentile(us, 99)),
        max_us=float(us.max()),
        rows_per_sec=batch_size / (mean / 1e6),
    )


def cpu_description() -> str:
    """`platform.processor()` is empty on Linux (e.g. in the container); fall back to /proc/cpuinfo."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def make_onnx_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def time_interleaved(native_fn, onnx_fn, batches: list[np.ndarray], warmup: int) -> tuple[list[int], list[int]]:
    for i in range(warmup):
        batch = batches[i % len(batches)]
        native_fn(batch)
        onnx_fn(batch)

    native_ns, onnx_ns = [], []
    clock = time.perf_counter_ns
    gc.disable()
    try:
        for block_no, block_start in enumerate(range(0, len(batches), BLOCK_SIZE)):
            block = batches[block_start:block_start + BLOCK_SIZE]
            order = ((native_fn, native_ns), (onnx_fn, onnx_ns))
            for fn, sink in (order if block_no % 2 == 0 else order[::-1]):
                for batch in block:
                    start = clock()
                    fn(batch)
                    sink.append(clock() - start)
    finally:
        gc.enable()
    return native_ns, onnx_ns


def time_encoding(predictor: Predictor, records: list[dict], runs: int, warmup: int) -> LatencyStats:
    encoder = predictor.encoder
    for record in records[:warmup]:
        encoder.encode_batch([record])
    samples = []
    gc.disable()
    try:
        for record in records[:runs]:
            start = time.perf_counter_ns()
            encoder.encode_batch([record])
            samples.append(time.perf_counter_ns() - start)
    finally:
        gc.enable()
    return summarize("encoder (shared)", 1, samples)


def to_markdown(results: list[LatencyStats], encoding: LatencyStats, env: dict) -> str:
    def fmt(x: float) -> str:
        return f"{x:,.1f}"

    by_key = {(r.runtime, r.batch_size): r for r in results}
    lines = [
        "# Inference latency: native XGBoost vs ONNX Runtime",
        "",
        f"Model `{env['model_version']}` · {env['runs']} timed runs per cell after {env['warmup']} warmup · "
        f"{env['threads']} thread(s) per runtime · latencies in microseconds (µs)",
        "",
        "| Batch | Runtime | Mean ± 95% CI | p50 | p95 | p99 | Max | Throughput (rows/s) | p50 speed-up vs native |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        native_p50 = by_key[("native xgboost", r.batch_size)].p50_us
        speedup = "1.00× (baseline)" if r.runtime == "native xgboost" else f"{native_p50 / r.p50_us:.2f}×"
        lines.append(
            f"| {r.batch_size} | {r.runtime} | {fmt(r.mean_us)} ± {fmt(r.ci95_us)} | {fmt(r.p50_us)} | "
            f"{fmt(r.p95_us)} | {fmt(r.p99_us)} | {fmt(r.max_us)} | {r.rows_per_sec:,.0f} | {speedup} |"
        )
    lines += [
        "",
        f"Shared input encoding (1 record, numpy): p50 {fmt(encoding.p50_us)} µs, p99 {fmt(encoding.p99_us)} µs "
        "-- paid identically by both runtimes in the API.",
        "",
        f"Environment: {env['platform']} · {env['cpu']} ({env['logical_cpus']} logical CPUs) · Python {env['python']} · "
        f"xgboost {env['xgboost']} · onnxruntime {env['onnxruntime']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark native XGBoost vs ONNX Runtime latency.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "production")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw" / "Train_v2.csv")
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "inference_benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.runs < 100:
        parser.error("--runs must be >= 100 for a meaningful p99")

    onnx_path = args.model_dir / "model.onnx"
    if not onnx_path.is_file():
        parser.error(f"{onnx_path} not found -- run scripts/export_onnx.py first")

    predictor = Predictor.load(args.model_dir, inference_threads=args.threads)
    booster = predictor.booster
    session = make_onnx_session(onnx_path, args.threads)
    input_name = session.get_inputs()[0].name

    records = read_raw_records(
        args.data, predictor.metadata["raw_categorical_levels"], predictor.metadata["numeric_columns"]
    )
    rng = np.random.default_rng(args.seed)
    X_all = predictor.encoder.encode_batch(records)

    def native(batch: np.ndarray) -> np.ndarray:
        return booster.inplace_predict(batch, validate_features=False)

    def onnx_runtime(batch: np.ndarray) -> np.ndarray:
        return session.run(["probabilities"], {input_name: batch})[0]

    results: list[LatencyStats] = []
    for batch_size in BATCH_SIZES:
        # Pre-sliced contiguous batches so array indexing isn't inside the timed call.
        batches = [
            np.ascontiguousarray(X_all[rng.integers(0, len(X_all), batch_size)])
            for _ in range(args.runs)
        ]
        native_ns, onnx_ns = time_interleaved(native, onnx_runtime, batches, args.warmup)
        results += [summarize("native xgboost", batch_size, native_ns), summarize("onnx runtime", batch_size, onnx_ns)]
        print(f"batch {batch_size}: native p50 {results[-2].p50_us:.1f} µs, onnx p50 {results[-1].p50_us:.1f} µs")

    shuffled = [records[i] for i in rng.permutation(len(records))]
    encoding = time_encoding(predictor, shuffled, args.runs, args.warmup)

    env = {
        "model_version": predictor.model_version,
        "runs": args.runs,
        "warmup": args.warmup,
        "threads": args.threads,
        "platform": platform.platform(terse=True),
        "cpu": cpu_description(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "onnxruntime": ort.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(results, encoding, env)
    args.out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in [*results, encoding])
    print("\n" + markdown)
    print(f"Wrote {args.out.with_suffix('.md')} and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
