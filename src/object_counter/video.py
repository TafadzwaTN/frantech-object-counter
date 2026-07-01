from __future__ import annotations

import cv2


def is_rtsp_source(raw_source: str) -> bool:
    value = raw_source.strip().lower()
    return value.startswith(("rtsp://", "rtsps://"))


def parse_video_source(raw_source: str) -> int | str:
    value = raw_source.strip()
    if value.isdigit():
        return int(value)
    return value


def resolve_frame_rotation(raw_source: str, setting: str) -> int:
    value = str(setting).strip().lower()
    if value in ("", "auto"):
        return 90 if is_rtsp_source(raw_source) else 0

    normalized = value.replace("degrees", "").replace("degree", "").replace("deg", "")
    normalized = normalized.replace("clockwise", "").replace("cw", "")
    normalized = normalized.replace("°", "").strip()
    if normalized not in {"0", "90", "180", "270"}:
        raise ValueError("Frame rotation must be auto, 0, 90, 180, or 270.")
    return int(normalized)


def apply_frame_rotation(frame, degrees: int):
    rotation = degrees % 360
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Frame rotation must be 0, 90, 180, or 270.")


def capture_one_frame(
    raw_source: str,
    rotation_degrees: int = 0,
) -> tuple[bool, object | None, str | None]:
    source = parse_video_source(raw_source)
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return False, None, f"Could not open video source: {raw_source}"
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None, "Video source opened, but no frame was returned."
        return True, apply_frame_rotation(frame, rotation_degrees), None
    finally:
        cap.release()
