"""Non-blocking JSON-lines prediction log.

The request path only does `queue.put_nowait(dict)` -- O(1), no I/O, no JSON
encoding. A single background thread drains the queue in batches, encodes,
and appends to `predictions-<UTC date>-<host>.jsonl`:

  - Daily files make "the last N days" a cheap file-level filter for the
    drift job, and let old days be archived/deleted without touching the
    live file.
  - The hostname suffix keeps replicas that share one volume from
    interleaving writes into the same file (in a container it's the
    container ID).
  - The file is reopened per batch (cheap, off the hot path), so an external
    archiver moving files away can't leave us writing to a deleted inode.

Back-pressure policy: the queue is bounded. If the disk stalls long enough
to fill it, new events are DROPPED rather than blocking requests or growing
memory without limit -- but never silently: every drop is counted, exposed
via `stats()` (surfaced by /health), and logged at WARNING.
"""
from __future__ import annotations

import json
import logging
import queue
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STOP = object()  # sentinel telling the writer thread to flush and exit
_MAX_BATCH = 512


class PredictionLogError(RuntimeError):
    """The log directory is unusable -- predictions would go unrecorded."""


class PredictionLogger:
    def __init__(self, log_dir: Path, max_queue_size: int = 10_000, flush_interval_s: float = 0.5):
        self.log_dir = log_dir
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._flush_interval_s = flush_interval_s
        self._host = socket.gethostname()
        self._thread: threading.Thread | None = None
        # Only the event-loop thread writes `_dropped`; only the writer thread
        # writes `_written`/`_write_errors`. Single-writer ints need no lock.
        self._dropped = 0
        self._written = 0
        self._write_errors = 0

    def start(self) -> None:
        """Verifies the directory is writable *now*, so a bad volume mount fails startup, not silently later."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            probe = self.log_dir / f".write-probe-{self._host}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise PredictionLogError(f"Prediction log directory {self.log_dir} is not writable: {exc}") from exc
        self._thread = threading.Thread(target=self._run, name="prediction-log-writer", daemon=True)
        self._thread.start()
        logger.info("Prediction log writing to %s", self.log_dir.resolve())

    def log(self, event: dict) -> None:
        """Enqueues one prediction event. Never blocks, never raises."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1
            # First drop, then every 1000th, so a sustained stall is loud but
            # doesn't itself flood the application log.
            if self._dropped == 1 or self._dropped % 1000 == 0:
                logger.warning(
                    "Prediction log queue full; %d event(s) dropped so far (disk slow or unavailable?)",
                    self._dropped,
                )

    def close(self, timeout_s: float = 5.0) -> None:
        """Flushes everything queued before shutdown; reports anything it couldn't write."""
        if self._thread is None:
            return
        try:
            self._queue.put(_STOP, timeout=timeout_s)
        except queue.Full:
            logger.error("Prediction log queue still full at shutdown; pending events will be lost")
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            logger.error("Prediction log writer did not finish within %.1fs; ~%d events unwritten",
                         timeout_s, self._queue.qsize())
        self._thread = None

    def stats(self) -> dict:
        return {
            "written": self._written,
            "queued": self._queue.qsize(),
            "dropped": self._dropped,
            "write_errors": self._write_errors,
            "writer_alive": self._thread is not None and self._thread.is_alive(),
        }

    def _current_path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"predictions-{day}-{self._host}.jsonl"

    def _run(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                continue
            batch, stop = [], first is _STOP
            if not stop:
                batch.append(first)
            # Drain whatever else is already waiting: one open/write/flush per
            # burst instead of per event.
            while not stop and len(batch) < _MAX_BATCH:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    stop = True
                else:
                    batch.append(item)
            if batch:
                self._write(batch)
            if stop:
                return

    def _write(self, batch: list[dict]) -> None:
        lines = []
        for event in batch:
            try:
                # allow_nan=False: a NaN would make the line invalid JSON for every reader.
                lines.append(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            except (TypeError, ValueError):
                self._write_errors += 1
                logger.exception("Unserializable prediction event (request_id=%s)", event.get("request_id"))
        if not lines:
            return
        try:
            with open(self._current_path(), "a", encoding="utf-8") as f:
                f.write("".join(lines))
            self._written += len(lines)
        except OSError:
            # Keep the writer alive (the disk may recover) but make the loss visible.
            self._write_errors += len(lines)
            logger.exception("Failed to write %d prediction log events", len(lines))
