from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from app.core.config import get_settings


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def performance_metrics_path() -> Path:
    settings = get_settings()
    return Path(settings.data_root) / "performance_metrics.jsonl"


def emit_performance_event(
    event: str,
    *,
    status: str = "ok",
    duration_ms: float | None = None,
    **context: Any,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
    }
    if duration_ms is not None:
        payload["duration_ms"] = round(float(duration_ms), 3)
    payload.update({k: _to_jsonable(v) for k, v in context.items()})

    out_path = performance_metrics_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


@contextmanager
def timed_performance_event(event: str, **context: Any) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        emit_performance_event(
            event,
            status="error",
            duration_ms=(perf_counter() - start) * 1000.0,
            error_type=type(exc).__name__,
            error=str(exc),
            **context,
        )
        raise
    else:
        emit_performance_event(
            event,
            status="ok",
            duration_ms=(perf_counter() - start) * 1000.0,
            **context,
        )
