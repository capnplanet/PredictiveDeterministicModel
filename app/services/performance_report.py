from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.performance import performance_metrics_path


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def build_performance_report(metrics_path: Path | None = None) -> dict[str, Any]:
    path = metrics_path or performance_metrics_path()
    events = _load_events(path)

    event_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    duration_by_event: dict[str, list[float]] = defaultdict(list)

    for event in events:
        event_name = str(event.get("event", "unknown"))
        status_name = str(event.get("status", "unknown"))
        event_counts[event_name] += 1
        status_counts[status_name] += 1

        duration_raw = event.get("duration_ms")
        if isinstance(duration_raw, (int, float)):
            duration_by_event[event_name].append(float(duration_raw))

    event_stats: dict[str, Any] = {}
    for event_name, durations in sorted(duration_by_event.items()):
        event_stats[event_name] = {
            "count": len(durations),
            "avg_ms": round(sum(durations) / len(durations), 3),
            "p95_ms": round(_percentile(durations, 0.95), 3),
            "max_ms": round(max(durations), 3),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(path),
        "total_events": len(events),
        "event_counts": dict(sorted(event_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "event_duration_ms": event_stats,
    }


def write_performance_report(output_path: Path, metrics_path: Path | None = None) -> dict[str, Any]:
    report = build_performance_report(metrics_path=metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
