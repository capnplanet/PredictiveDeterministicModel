from __future__ import annotations

import csv
import json
import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from sqlalchemy.orm import Session

from app.db.models import Entity, Event, Interaction


class IngestionError(Exception):
    pass


@dataclass
class IngestionReport:
    total_rows: int
    success_rows: int
    failed_rows: int
    errors: List[str]


_MAX_ROWS = 1_000_000
_MAX_FILE_BYTES = 50 * 1024 * 1024


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise IngestionError(f"File too large: {size} bytes > max {_MAX_FILE_BYTES}")


def _read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    _validate_file(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_yielded = 0
        for row in reader:
            rows_yielded += 1
            if rows_yielded > _MAX_ROWS:
                raise IngestionError(f"Row limit exceeded: {_MAX_ROWS}")
            yield row


def ingest_entities_csv(session: Session, path: Path) -> IngestionReport:
    required_cols = {"entity_id", "attributes"}
    total = success = failed = 0
    errors: List[str] = []

    for row in _read_csv_rows(path):
        total += 1
        missing = required_cols - set(row.keys())
        if missing:
            failed += 1
            errors.append(f"Missing columns {missing} in row {total}")
            continue
        try:
            entity_id = row["entity_id"].strip()
            if not entity_id:
                raise ValueError("empty entity_id")
            attrs_raw = row["attributes"].strip()
            if attrs_raw:
                try:
                    attributes = json.loads(attrs_raw)
                except json.JSONDecodeError:
                    # Fallback for slightly non-JSON dict literals
                    attributes = ast.literal_eval(attrs_raw)
            else:
                attributes = {}
            created_at_str = row.get("created_at", "").strip()
            if created_at_str:
                created_at = datetime.fromisoformat(created_at_str)
            else:
                created_at = datetime.utcnow()
            entity = Entity(entity_id=entity_id, attributes=attributes, created_at=created_at)
            session.merge(entity)
            success += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            raw_attrs = row.get("attributes", "")
            errors.append(f"Row {total}: {exc} [attributes={raw_attrs!r}]")

    # Ensure pending inserts are flushed so callers can query immediately within the same session
    if success:
        session.flush()

    return IngestionReport(total_rows=total, success_rows=success, failed_rows=failed, errors=errors)


def ingest_events_csv(session: Session, path: Path) -> IngestionReport:
    required_cols = {"timestamp", "entity_id", "event_type", "event_value"}
    total = success = failed = 0
    errors: List[str] = []

    for row in _read_csv_rows(path):
        total += 1
        missing = required_cols - set(row.keys())
        if missing:
            failed += 1
            errors.append(f"Missing columns {missing} in row {total}")
            continue
        try:
            ts = datetime.fromisoformat(row["timestamp"].strip())
            entity_id = row["entity_id"].strip()
            event_type = row["event_type"].strip()
            event_value = float(row["event_value"].strip())
            metadata_raw = row.get("event_metadata", "").strip()
            event_metadata = json.loads(metadata_raw) if metadata_raw else {}
            event = Event(
                timestamp=ts,
                entity_id=entity_id,
                event_type=event_type,
                event_value=event_value,
                event_metadata=event_metadata,
            )
            session.add(event)
            success += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            errors.append(f"Row {total}: {exc}")

    if success:
        session.flush()

    return IngestionReport(total_rows=total, success_rows=success, failed_rows=failed, errors=errors)


def ingest_interactions_csv(session: Session, path: Path) -> IngestionReport:
    required_cols = {"timestamp", "src_entity_id", "dst_entity_id", "interaction_type", "interaction_value"}
    total = success = failed = 0
    errors: List[str] = []

    for row in _read_csv_rows(path):
        total += 1
        missing = required_cols - set(row.keys())
        if missing:
            failed += 1
            errors.append(f"Missing columns {missing} in row {total}")
            continue
        try:
            ts = datetime.fromisoformat(row["timestamp"].strip())
            src = row["src_entity_id"].strip()
            dst = row["dst_entity_id"].strip()
            itype = row["interaction_type"].strip()
            ivalue = float(row["interaction_value"].strip())
            metadata_raw = row.get("metadata", "").strip()
            metadata = json.loads(metadata_raw) if metadata_raw else {}
            interaction = Interaction(
                timestamp=ts,
                src_entity_id=src,
                dst_entity_id=dst,
                interaction_type=itype,
                interaction_value=ivalue,
                meta=metadata,
            )
            session.add(interaction)
            success += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            errors.append(f"Row {total}: {exc}")

    if success:
        session.flush()

    return IngestionReport(total_rows=total, success_rows=success, failed_rows=failed, errors=errors)
