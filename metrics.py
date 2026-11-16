"""Scrape the Furiosa LLM Prometheus exporter and record snapshots.

Equivalent to:

    curl -s localhost:8000/metrics | grep '^furiosa_llm_'

but parsed into families/samples, and optionally appended to a JSONL log
so the state of every metric can be compared over time.

Usage:
    python3 metrics.py                 # print current furiosa_llm_* samples
    python3 metrics.py --all           # print every metric, not just furiosa_llm_*
    python3 metrics.py --record        # print and append a snapshot to data/metrics.jsonl
    python3 metrics.py --watch 5       # re-scrape and record every 5 seconds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parent
METRICS_URL = os.environ.get("METRICS_URL", "http://localhost:8000/metrics")
METRICS_PREFIX = os.environ.get("METRICS_PREFIX", "furiosa_llm_")
RECORD_PATH = ROOT / "data" / "metrics.jsonl"


@dataclass
class Sample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass
class Family:
    name: str
    type: str
    documentation: str
    samples: list[Sample] = field(default_factory=list)


@dataclass
class Snapshot:
    timestamp: str
    url: str
    families: list[Family]

    @property
    def sample_count(self) -> int:
        return sum(len(family.samples) for family in self.families)


def fetch_text(url: str = METRICS_URL, timeout: float = 10.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach metrics exporter at {url}: {exc.reason}") from exc


def parse(text: str, prefix: str | None = METRICS_PREFIX) -> list[Family]:
    families: list[Family] = []
    for fam in text_string_to_metric_families(text):
        samples = [
            Sample(sample.name, dict(sample.labels), float(sample.value))
            for sample in fam.samples
            if not prefix or sample.name.startswith(prefix)
        ]
        if not samples:
            continue
        families.append(Family(fam.name, fam.type, fam.documentation, samples))
    families.sort(key=lambda family: family.name)
    return families


def snapshot(url: str = METRICS_URL, prefix: str | None = METRICS_PREFIX) -> Snapshot:
    return Snapshot(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        url=url,
        families=parse(fetch_text(url), prefix),
    )


def record(snap: Snapshot, path: Path = RECORD_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(snap)) + "\n")
    return path


def load_records(path: Path = RECORD_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def print_snapshot(snap: Snapshot) -> None:
    print(f"# {snap.timestamp}  {snap.url}  ({snap.sample_count} samples)")
    for family in snap.families:
        print(f"\n# {family.type.upper()} {family.name}: {family.documentation}")
        for sample in family.samples:
            labels = (
                "{" + ",".join(f'{k}="{v}"' for k, v in sorted(sample.labels.items())) + "}"
                if sample.labels
                else ""
            )
            print(f"{sample.name}{labels} {sample.value:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=METRICS_URL, help=f"exporter URL (default {METRICS_URL})")
    parser.add_argument("--all", action="store_true", help=f"show all metrics, not only {METRICS_PREFIX}*")
    parser.add_argument("--record", action="store_true", help=f"append snapshot to {RECORD_PATH.relative_to(ROOT)}")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="re-scrape every N seconds (implies --record)")
    args = parser.parse_args()

    prefix = None if args.all else METRICS_PREFIX
    do_record = args.record or args.watch is not None

    while True:
        try:
            snap = snapshot(args.url, prefix)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if args.watch is None:
                sys.exit(1)
            time.sleep(args.watch)
            continue
        print_snapshot(snap)
        if do_record:
            path = record(snap)
            print(f"\n# recorded to {path}")
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
