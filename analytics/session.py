"""
Session recording and export for LookThePerson.

Collects what happened during a run — gestures, reps, posture samples, events —
and writes it out as JSON, CSV or JSONL so it can be reviewed later or fed into
another tool.

Landmark logging is optional and off by default: it produces large files, and
recording body position data deserves an explicit opt-in.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["SessionRecord", "SessionRecorder", "export_json", "export_csv", "export_jsonl"]


@dataclass
class SessionRecord:
    """One timestamped row in the session log."""

    timestamp: float
    kind: str                                  # gesture | rep | posture | event | sample
    label: str = ""
    value: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    #: Keys the record owns. A payload entry of the same name is renamed rather
    #: than merged, so an event carrying ``label`` cannot overwrite the record's
    #: real label in the export and misreport what happened.
    RESERVED = ("timestamp", "kind", "label", "value")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            (f"data_{key}" if key in self.RESERVED else key): value
            for key, value in self.data.items()
        }
        return {
            "timestamp": round(self.timestamp, 3),
            "kind": self.kind,
            "label": self.label,
            "value": self.value,
            **payload,
        }


class SessionRecorder:
    """
    Accumulates session records and writes them to disk.

    Records are held in memory (capped, so a long run cannot exhaust RAM) and
    flushed on demand or at shutdown.
    """

    def __init__(
        self,
        output_dir: str = "",
        max_records: int = 50_000,
        enabled: bool = True,
    ):
        base = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions"
        )
        self.output_dir = base
        self.max_records = max_records
        self.enabled = enabled

        self.started_at = time.time()
        self.started_monotonic = time.monotonic()
        self._records: List[SessionRecord] = []
        self._dropped = 0
        self._metadata: Dict[str, Any] = {}

    # -- Recording ----------------------------------------------------------

    def record(
        self,
        kind: str,
        label: str = "",
        value: float = 0.0,
        timestamp: Optional[float] = None,
        /,
        **data: Any,
    ) -> Optional[SessionRecord]:
        """
        Append a record. Returns None when recording is disabled or full.

        The fixed parameters are positional-only, matching
        :meth:`~core.events.EventBus.emit`, because ``**data`` is filled from an
        arbitrary event payload. A payload carrying its own ``label`` or
        ``value`` would otherwise collide with these names and raise, and the
        bus responds to a raising handler by disabling it — which silently ends
        session logging for the rest of the run.
        """
        if not self.enabled:
            return None
        if len(self._records) >= self.max_records:
            self._dropped += 1
            return None

        entry = SessionRecord(
            timestamp=timestamp if timestamp is not None else self.elapsed(),
            kind=kind,
            label=label,
            value=value,
            data=data,
        )
        self._records.append(entry)
        return entry

    def record_gesture(self, name: str, timestamp: Optional[float] = None) -> None:
        self.record("gesture", name, 1.0, timestamp)

    def record_rep(self, exercise: str, index: int, form_score: float, duration: float) -> None:
        self.record(
            "rep", exercise, float(index),
            form_score=round(form_score, 1), duration=round(duration, 2),
        )

    def record_posture(self, score: float, issues: Sequence[str] = ()) -> None:
        self.record("posture", "score", round(score, 1), issues=list(issues))

    def record_event(self, name: str, /, **data: Any) -> None:
        """Log a bus event. *name* is positional-only — payloads carry one too."""
        self.record("event", name, 1.0, None, **data)

    def set_metadata(self, **data: Any) -> None:
        """Attach session-level metadata included in every export."""
        self._metadata.update(data)

    # -- Introspection ------------------------------------------------------

    def elapsed(self) -> float:
        return time.monotonic() - self.started_monotonic

    @property
    def records(self) -> List[SessionRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def by_kind(self, kind: str) -> List[SessionRecord]:
        return [r for r in self._records if r.kind == kind]

    def counts(self) -> Dict[str, int]:
        """How many records of each kind were captured."""
        totals: Dict[str, int] = {}
        for record in self._records:
            totals[record.kind] = totals.get(record.kind, 0) + 1
        return totals

    def label_counts(self, kind: str) -> Dict[str, int]:
        """Frequency of each label within one kind — e.g. gestures used."""
        totals: Dict[str, int] = {}
        for record in self.by_kind(kind):
            totals[record.label] = totals.get(record.label, 0) + 1
        return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    def summary(self) -> Dict[str, Any]:
        """Session-level summary document."""
        return {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at)),
            "duration_seconds": round(self.elapsed(), 1),
            "record_count": len(self._records),
            "dropped_records": self._dropped,
            "counts": self.counts(),
            "gestures": self.label_counts("gesture"),
            **self._metadata,
        }

    # -- Export -------------------------------------------------------------

    def _path(self, extension: str, prefix: str = "session") -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started_at))
        return os.path.join(self.output_dir, f"{prefix}_{stamp}.{extension}")

    def export(self, fmt: str = "json", path: Optional[str] = None) -> Optional[str]:
        """
        Write the session to disk in *fmt* (``json``, ``csv`` or ``jsonl``).

        Returns the path written, or None when there was nothing to write.
        """
        if not self._records:
            return None

        fmt = fmt.lower()
        if fmt == "csv":
            target = path or self._path("csv")
            export_csv(self._records, target)
        elif fmt == "jsonl":
            target = path or self._path("jsonl")
            export_jsonl(self._records, target)
        else:
            target = path or self._path("json")
            export_json(self._records, target, self.summary())

        print(f"[session] Exportado: {target}", flush=True)
        return target

    def clear(self) -> None:
        self._records.clear()
        self._dropped = 0


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def export_json(
    records: Iterable[SessionRecord],
    path: str,
    summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Write records plus an optional summary as a single JSON document."""
    document = {
        "summary": summary or {},
        "records": [r.to_dict() for r in records],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def export_jsonl(records: Iterable[SessionRecord], path: str) -> str:
    """Write one JSON object per line — convenient for streaming consumers."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def export_csv(records: Iterable[SessionRecord], path: str) -> str:
    """
    Write records as CSV.

    The column set is the union of every record's keys, so rows with extra
    fields do not lose data and rows missing them stay blank.
    """
    rows = [r.to_dict() for r in records]
    if not rows:
        return path

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify(v) for k, v in row.items()})
    return path


def _stringify(value: Any) -> Any:
    """Flatten lists and dicts so they survive a CSV round-trip readably."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return value
