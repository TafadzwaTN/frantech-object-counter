from __future__ import annotations

import cv2

from .counting import Line, Rect
from .detections import TrackedDetection


def draw_overlay(
    frame_bgr,
    tracks: list[TrackedDetection],
    total: int,
    mode: str,
    line: Line | None = None,
    zone: Rect | None = None,
):
    output = frame_bgr.copy()

    if line is not None:
        p1, p2 = line
        cv2.line(
            output,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 255, 255),
            2,
        )
    if zone is not None:
        x1, y1, x2, y2 = zone
        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 180, 0),
            2,
        )

    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.xyxy]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"#{track.track_id} {track.label}"
        if track.score is not None:
            label += f" {track.score:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"{mode}: {total}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 255),
        3,
        cv2.LINE_AA,
    )
    return output
