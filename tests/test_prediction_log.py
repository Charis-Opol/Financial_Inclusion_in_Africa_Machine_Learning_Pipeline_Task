import json
import math

from fin_inclusion.serving.prediction_log import PredictionLogger


def _read_events(log_dir):
    return [json.loads(line) for f in sorted(log_dir.glob("predictions-*.jsonl")) for line in f.read_text().splitlines()]


def test_close_flushes_every_queued_event(tmp_path):
    logger = PredictionLogger(tmp_path)
    logger.start()
    for i in range(1500):  # more than one writer batch
        logger.log({"request_id": str(i)})
    logger.close()
    events = _read_events(tmp_path)
    assert [e["request_id"] for e in events] == [str(i) for i in range(1500)]
    assert logger.stats()["written"] == 1500


def test_full_queue_drops_and_counts_instead_of_blocking(tmp_path):
    logger = PredictionLogger(tmp_path, max_queue_size=3)  # writer not started: queue can't drain
    for i in range(10):
        logger.log({"request_id": str(i)})
    assert logger.stats()["dropped"] == 7
    assert logger.stats()["queued"] == 3


def test_unserializable_event_is_counted_without_losing_its_batch(tmp_path):
    logger = PredictionLogger(tmp_path)
    logger.start()
    logger.log({"request_id": "ok-1"})
    logger.log({"request_id": "bad", "probability": math.nan})
    logger.log({"request_id": "ok-2"})
    logger.close()
    assert [e["request_id"] for e in _read_events(tmp_path)] == ["ok-1", "ok-2"]
    assert logger.stats()["write_errors"] == 1
