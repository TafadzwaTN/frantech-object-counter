from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .detections import TrackedDetection


Point = tuple[float, float]
Line = tuple[Point, Point]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class CountEvent:
    timestamp: str
    session_id: str
    detector: str
    target_class: str
    track_id: int
    count_mode: str
    box: tuple[float, float, float, float]
    score: float | None
    snapshot_path: str


def _side_of_line(point: Point, line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _passes_direction(previous: Point, current: Point, direction: str) -> bool:
    px, py = previous
    cx, cy = current
    if direction == "Any":
        return True
    if direction == "Left to right":
        return cx > px
    if direction == "Right to left":
        return cx < px
    if direction == "Top to bottom":
        return cy > py
    if direction == "Bottom to top":
        return cy < py
    return True


def _inside_rect(point: Point, rect: Rect) -> bool:
    x1, y1, x2, y2 = rect
    px, py = point
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return left <= px <= right and top <= py <= bottom


class ObjectCounter:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.previous_centers: dict[int, Point] = {}
        self.counted_track_ids: set[int] = set()
        self.total = 0

    def update_line(
        self,
        tracks: list[TrackedDetection],
        target_class: str,
        line: Line,
        direction: str,
    ) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks:
            current = track.center
            previous = self.previous_centers.get(track.track_id)
            self.previous_centers[track.track_id] = current

            if previous is None or track.track_id in self.counted_track_ids:
                continue

            previous_side = _side_of_line(previous, line)
            current_side = _side_of_line(current, line)
            crossed = previous_side == 0 or current_side == 0 or previous_side * current_side < 0
            if crossed and _passes_direction(previous, current, direction):
                events.append(self._count(track, target_class, "line_crossing"))
        return events

    def update_zone(
        self,
        tracks: list[TrackedDetection],
        target_class: str,
        zone: Rect,
    ) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks:
            current = track.center
            previous = self.previous_centers.get(track.track_id)
            self.previous_centers[track.track_id] = current

            if track.track_id in self.counted_track_ids:
                continue

            was_inside = False if previous is None else _inside_rect(previous, zone)
            is_inside = _inside_rect(current, zone)
            if is_inside and not was_inside:
                events.append(self._count(track, target_class, "zone_entry"))
        return events

    def _count(
        self,
        track: TrackedDetection,
        target_class: str,
        count_mode: str,
    ) -> CountEvent:
        self.counted_track_ids.add(track.track_id)
        self.total += 1
        return CountEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            session_id=self.session_id,
            detector=track.detector,
            target_class=target_class,
            track_id=track.track_id,
            count_mode=count_mode,
            box=track.xyxy,
            score=track.score,
            snapshot_path="",
        )
