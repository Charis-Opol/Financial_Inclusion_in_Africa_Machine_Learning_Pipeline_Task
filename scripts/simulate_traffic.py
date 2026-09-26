"""Sends realistic traffic to a running API so the prediction log has something to monitor.

    python scripts/simulate_traffic.py --n 2000 --scenario baseline
    python scripts/simulate_traffic.py --n 2000 --scenario rural-outreach [--country Uganda]

Scenarios (records drawn from the held-out Test_v2.csv, never seen in training):
  baseline        a random sample -- same population as training, so the
                  drift check should report STABLE.
  rural-outreach  only rural respondents with at most primary education --
                  what traffic looks like if the model is deployed for a
                  rural financial-outreach programme. A genuine population
                  shift; the drift check should flag it.

Stdlib only (urllib + threads) so it runs from any environment.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fin_inclusion.serving.raw_records import read_raw_records  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATEGORICAL = ["country", "location_type", "cellphone_access", "gender_of_respondent",
               "relationship_with_head", "marital_status", "education_level", "job_type"]
NUMERIC = ["household_size", "age_of_respondent"]
LOW_EDUCATION = {"No formal education", "Primary education"}


def select_records(records: list[dict], scenario: str, country: str | None) -> list[dict]:
    if country:
        records = [r for r in records if r["country"] == country]
    if scenario == "rural-outreach":
        records = [r for r in records if r["location_type"] == "Rural" and r["education_level"] in LOW_EDUCATION]
    if not records:
        raise SystemExit(f"No records match scenario={scenario!r} country={country!r}")
    return records


def post(url: str, record: dict) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(record).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "raw" / "Test_v2.csv")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--scenario", choices=["baseline", "rural-outreach"], default="baseline")
    parser.add_argument("--country")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pool = select_records(read_raw_records(args.data, CATEGORICAL, NUMERIC), args.scenario, args.country)
    sample = random.Random(args.seed).choices(pool, k=args.n)
    with ThreadPoolExecutor(args.concurrency) as executor:
        statuses = Counter(executor.map(lambda r: post(args.url, r), sample))
    print(f"Sent {args.n} '{args.scenario}' requests (pool of {len(pool)}): status codes {dict(statuses)}")
    if set(statuses) != {200}:
        sys.exit(1)


if __name__ == "__main__":
    main()
