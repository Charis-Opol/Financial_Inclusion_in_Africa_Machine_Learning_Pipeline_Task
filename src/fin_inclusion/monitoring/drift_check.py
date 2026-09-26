"""PSI drift check: recent logged predictions vs the training reference profile.

    python -m fin_inclusion.monitoring.drift_check --log-dir logs [--hours 168 | --last-n 5000]
                                                   [--country Uganda] [--json-out report.json]

Designed to be scheduled (cron, Windows Task Scheduler, `docker compose run
drift-check`): it prints a table for humans, optionally writes JSON for
machines, and signals the verdict through its exit code so a scheduler can
alert without parsing output:

    0   all monitored distributions stable       (PSI < 0.10)
    10  at least one moderate shift              (0.10 <= PSI <= 0.25)
    20  at least one significant shift           (PSI > 0.25)
    30  not enough predictions in the window to judge
    1   the check itself failed (missing/mismatched reference, corrupt log)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from fin_inclusion.monitoring import psi
from fin_inclusion.monitoring.reference_profile import POOLED_SEGMENT, SCORE_FEATURE

EXIT_STABLE, EXIT_MODERATE, EXIT_SIGNIFICANT, EXIT_INSUFFICIENT, EXIT_ERROR = 0, 10, 20, 30, 1
EXIT_BY_STATUS = {"stable": EXIT_STABLE, "moderate": EXIT_MODERATE, "significant": EXIT_SIGNIFICANT}
# A crash can truncate the last line of a file, so a handful of malformed
# lines is survivable; more than this share means the log itself is broken
# and any PSI computed from it can't be trusted.
MAX_MALFORMED_FRACTION = 0.01
LOG_GLOB = "predictions-*.jsonl"


class DriftCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    status: str
    largest_shift_bin: str
    reference_share: float
    recent_share: float


@dataclass
class LogScan:
    events: list[dict]
    lines_read: int = 0
    malformed: int = 0
    other_model_versions: int = 0


def _file_day(path: Path) -> datetime | None:
    """`predictions-2026-09-26-<host>.jsonl` -> that UTC day, so whole files outside the window are skipped unread."""
    try:
        return datetime.strptime(path.name[len("predictions-"):][:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _iter_lines(log_dir: Path, since: datetime | None) -> Iterator[str]:
    files = sorted(log_dir.glob(LOG_GLOB))
    if not files:
        raise DriftCheckError(f"No {LOG_GLOB} files in {log_dir.resolve()}")
    for path in files:
        day = _file_day(path)
        if since is not None and day is not None and day + timedelta(days=1) <= since:
            continue
        with open(path, encoding="utf-8") as f:
            yield from f


def scan_logs(log_dir: Path, since: datetime | None, model_version: str | None, country: str | None) -> LogScan:
    scan = LogScan(events=[])
    for line in _iter_lines(log_dir, since):
        if not line.strip():
            continue
        scan.lines_read += 1
        try:
            event = json.loads(line)
            timestamp = datetime.fromisoformat(event["timestamp"])
            features = event["features"]
            float(event["probability"])
        except (ValueError, KeyError, TypeError):
            scan.malformed += 1
            continue
        if since is not None and timestamp < since:
            continue
        if model_version is not None and event.get("model_version") != model_version:
            scan.other_model_versions += 1
            continue
        if country is not None and features.get("country") != country:
            continue
        scan.events.append(event)

    if scan.lines_read and scan.malformed / scan.lines_read > MAX_MALFORMED_FRACTION:
        raise DriftCheckError(
            f"{scan.malformed}/{scan.lines_read} log lines are malformed (> {MAX_MALFORMED_FRACTION:.0%}); "
            "refusing to compute drift from a corrupt log"
        )
    scan.events.sort(key=lambda e: e["timestamp"])
    return scan


def feature_drift(name: str, spec: dict, recent_values: list) -> FeatureDrift:
    if spec["kind"] == "numeric":
        recent = psi.numeric_proportions(recent_values, spec["edges"])
        labels = psi.numeric_bin_labels(spec["edges"])
    else:
        recent = psi.categorical_proportions(recent_values, spec["levels"])
        labels = psi.categorical_bin_labels(spec["levels"])
    reference = np.asarray(spec["proportions"])
    value = psi.psi(reference, recent)
    worst = int(np.argmax(psi.per_bin_contributions(reference, recent)))
    return FeatureDrift(
        feature=name,
        psi=value,
        status=psi.interpret(value),
        largest_shift_bin=labels[worst],
        reference_share=float(reference[worst]),
        recent_share=float(recent[worst]),
    )


def compute_drift(segment: dict, events: list[dict]) -> list[FeatureDrift]:
    results = []
    for name, spec in segment["features"].items():
        if name == SCORE_FEATURE:
            values = [e["probability"] for e in events]
        else:
            values = [e["features"][name] for e in events]
        results.append(feature_drift(name, spec, values))
    return sorted(results, key=lambda r: r.psi, reverse=True)


def render_table(results: list[FeatureDrift]) -> str:
    rows = [
        "| Feature | PSI | Status | Largest shift (bin: reference → recent) |",
        "|---|---:|---|---|",
    ]
    for r in results:
        rows.append(
            f"| {r.feature} | {r.psi:.4f} | {r.status} | "
            f"{r.largest_shift_bin}: {r.reference_share:.1%} → {r.recent_share:.1%} |"
        )
    return "\n".join(rows)


def run(args: argparse.Namespace) -> int:
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    segment_name = args.country or POOLED_SEGMENT
    if segment_name not in reference["segments"]:
        raise DriftCheckError(f"No reference segment {segment_name!r}; have {sorted(reference['segments'])}")
    segment = reference["segments"][segment_name]
    # PSI against another model's reference is meaningless, so by default
    # only events from the model the reference was built for count.
    model_version = None if args.any_model_version else reference["model_version"]

    now = datetime.now(timezone.utc)
    since = None if args.last_n else now - timedelta(hours=args.hours)
    scan = scan_logs(args.log_dir, since, model_version, args.country)
    events = scan.events[-args.last_n:] if args.last_n else scan.events

    window = f"last {args.last_n} predictions" if args.last_n else f"last {args.hours}h (since {since:%Y-%m-%d %H:%M} UTC)"
    print(f"Drift check · model {reference['model_version']} · segment '{segment_name}' "
          f"(reference n={segment['n']}) · window: {window}")
    print(f"Log lines read: {scan.lines_read}, malformed: {scan.malformed}, "
          f"other model versions skipped: {scan.other_model_versions}, in window: {len(events)}")

    report: dict = {
        "checked_at": now.isoformat(timespec="seconds"),
        "model_version": reference["model_version"],
        "segment": segment_name,
        "window": window,
        "n_recent": len(events),
        "scan": {k: v for k, v in asdict(scan).items() if k != "events"},
    }
    if len(events) < args.min_samples:
        # With few samples, sampling noise alone produces PSI in the
        # "moderate" range, so a verdict would be noise, not signal.
        print(f"\nINSUFFICIENT DATA: {len(events)} predictions < --min-samples {args.min_samples}; no verdict.")
        report["status"] = "insufficient_data"
        _write_json(args.json_out, report)
        return EXIT_INSUFFICIENT

    results = compute_drift(segment, events)
    overall = max((r.status for r in results), key=lambda s: EXIT_BY_STATUS[s])
    recent_positive_rate = float(np.mean([e["prediction"] == "Yes" for e in events]))

    print("\n" + render_table(results))
    print(f"\nPredicted-'Yes' rate: reference {segment['positive_rate']:.1%} → recent {recent_positive_rate:.1%}")
    print(f"PSI thresholds: < {psi.PSI_STABLE_BELOW} stable, {psi.PSI_STABLE_BELOW}-{psi.PSI_SIGNIFICANT_ABOVE} moderate, "
          f"> {psi.PSI_SIGNIFICANT_ABOVE} significant")
    print(f"OVERALL: {overall.upper()}")

    report |= {
        "status": overall,
        "reference_positive_rate": segment["positive_rate"],
        "recent_positive_rate": recent_positive_rate,
        "features": [asdict(r) for r in results],
    }
    _write_json(args.json_out, report)
    return EXIT_BY_STATUS[overall]


def _write_json(path: Path | None, report: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSI drift check of logged predictions against the training reference.",
        epilog="Exit codes: 0 stable, 10 moderate, 20 significant, 30 insufficient data, 1 error.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--reference", type=Path, default=Path("models/production/reference_profile.json"))
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--hours", type=float, default=168.0, help="time window (default: 7 days)")
    window.add_argument("--last-n", type=int, help="use the most recent N predictions instead of a time window")
    parser.add_argument("--country", help="compare only this country's traffic to its own training reference")
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--any-model-version", action="store_true",
                        help="include events from model versions other than the reference's")
    parser.add_argument("--json-out", type=Path, help="also write the full report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (DriftCheckError, OSError, json.JSONDecodeError) as exc:
        print(f"DRIFT CHECK FAILED: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
