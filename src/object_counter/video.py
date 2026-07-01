from __future__ import annotations

import cv2


def parse_video_source(raw_source: str) -> int | str:
    value = raw_source.strip()
    if value.isdigit():
        return int(value)
    return value


def capture_one_frame(raw_source: str) -> tuple[bool, object | None, str | None]:
    source = parse_video_source(raw_source)
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return False, None, f"Could not open video source: {raw_source}"
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None, "Video source opened, but no frame was returned."
        return True, frame, None
    finally:
        cap.release()
