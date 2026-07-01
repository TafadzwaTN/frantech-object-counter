from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import cv2

from .counting import CountEvent


CSV_FIELDS = [
    "timestamp",
    "session_id",
    "detector",
    "target_class",
    "track_id",
    "count_mode",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "snapshot_path",
]


class SessionStorage:
    def __init__(self, root: Path, session_id: str, save_snapshots: bool):
        self.session_dir = root / session_id
        self.snapshots_dir = self.session_dir / "snapshots"
        self.csv_path = self.session_dir / "events.csv"
        self.save_snapshots = save_snapshots
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if save_snapshots:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()

    def write_event(self, event: CountEvent, frame_bgr) -> CountEvent:
        snapshot_path = ""
        if self.save_snapshots:
            snapshot = self.snapshots_dir / f"track-{event.track_id}-{event.timestamp.replace(':', '-')}.jpg"
            cv2.imwrite(str(snapshot), frame_bgr)
            snapshot_path = str(snapshot)

        event = replace(event, snapshot_path=snapshot_path)
        x1, y1, x2, y2 = event.box
        row = {
            "timestamp": event.timestamp,
            "session_id": event.session_id,
            "detector": event.detector,
            "target_class": event.target_class,
            "track_id": event.track_id,
            "count_mode": event.count_mode,
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
            "score": "" if event.score is None else round(event.score, 4),
            "snapshot_path": event.snapshot_path,
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writerow(row)
        return event
