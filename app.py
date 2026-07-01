from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from object_counter.config import AppConfig, load_config
from object_counter.counting import Line, ObjectCounter, Rect
from object_counter.detections import Detection, TrackedDetection
from object_counter.detectors import ALL_CLASSES, load_detector, load_yolo_class_names
from object_counter.drawing import draw_overlay
from object_counter.storage import SessionStorage
from object_counter.tracking import ByteTrackTracker
from object_counter.video import capture_one_frame, parse_video_source


try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - Streamlit shows the actionable message.
    st_canvas = None


def _as_rgb_image(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _canvas_frame(frame_bgr, max_width: int = 900):
    height, width = frame_bgr.shape[:2]
    display_width = min(max_width, width)
    scale = display_width / width
    display_height = int(height * scale)
    rgb = _as_rgb_image(frame_bgr)
    resized = cv2.resize(rgb, (display_width, display_height))
    return resized, width / display_width, height / display_height


def _last_canvas_object(canvas_result):
    if not canvas_result or not canvas_result.json_data:
        return None
    objects = canvas_result.json_data.get("objects", [])
    return objects[-1] if objects else None


def _extract_line(obj: dict, scale_x: float, scale_y: float) -> Line | None:
    if obj is None:
        return None
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    if "x1" in obj and "x2" in obj:
        x1 = left + float(obj.get("x1", 0))
        y1 = top + float(obj.get("y1", 0))
        x2 = left + float(obj.get("x2", 0))
        y2 = top + float(obj.get("y2", 0))
    else:
        x1 = left
        y1 = top
        x2 = left + float(obj.get("width", 0))
        y2 = top + float(obj.get("height", 0))
    return ((x1 * scale_x, y1 * scale_y), (x2 * scale_x, y2 * scale_y))


def _extract_rect(obj: dict, scale_x: float, scale_y: float) -> Rect | None:
    if obj is None:
        return None
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
    height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
    if width <= 0 or height <= 0:
        return None
    return (
        left * scale_x,
        top * scale_y,
        (left + width) * scale_x,
        (top + height) * scale_y,
    )


@st.cache_resource(show_spinner=False)
def _cached_detector(
    kind: str,
    locateanything_root: str,
    locateanything_model_path: str,
    locateanything_device: str,
    yolo_weights_path: str,
    yolo_confidence: float,
):
    config = AppConfig(
        locateanything_root=Path(locateanything_root),
        locateanything_model_path=locateanything_model_path,
        locateanything_device=locateanything_device,
        yolo_weights_path=Path(yolo_weights_path),
        sessions_dir=ROOT / "sessions",
    )
    return load_detector(kind, config, yolo_confidence=yolo_confidence)


def _default_target(detector_kind: str) -> str:
    if detector_kind == "YOLO":
        return ALL_CLASSES
    return "Arduino Uno"


def _default_detector_index() -> int:
    return 0 if torch.cuda.is_available() else 1


@st.cache_resource(show_spinner=False)
def _cached_yolo_class_names(yolo_weights_path: str) -> list[str]:
    return load_yolo_class_names(Path(yolo_weights_path))


def _preview_tracks(detections: list[Detection]) -> list[TrackedDetection]:
    return [
        TrackedDetection(
            x1=det.x1,
            y1=det.y1,
            x2=det.x2,
            y2=det.y2,
            label=det.label,
            score=det.score,
            detector=det.detector,
            track_id=index + 1,
        )
        for index, det in enumerate(detections)
    ]


def main():
    st.set_page_config(page_title="Frantech Object Counter", layout="wide")
    config = load_config()

    st.title("Frantech Object Counter")

    with st.sidebar:
        detector_kind = st.selectbox(
            "Detector",
            ["LocateAnything", "YOLO"],
            index=_default_detector_index(),
        )
        if detector_kind == "LocateAnything" and not torch.cuda.is_available():
            st.warning(
                "This environment has CPU-only PyTorch. LocateAnything may be "
                "very slow unless you install a CUDA-enabled PyTorch build."
            )
        if detector_kind == "YOLO":
            try:
                yolo_classes = _cached_yolo_class_names(str(config.yolo_weights_path))
                target_options = [ALL_CLASSES, *yolo_classes]
                target_class = st.selectbox("Target class", target_options, index=0)
                if len(yolo_classes) == 80 and "Arduino-Uno" not in yolo_classes:
                    st.warning(
                        "The selected YOLO weights look like a generic COCO model, "
                        "not the 61-class electronics model."
                    )
            except Exception as exc:
                st.error(f"Could not read YOLO classes: {exc}")
                target_class = st.text_input(
                    "Target class", value=_default_target(detector_kind)
                )
        else:
            target_class = st.text_input("Target description", value=_default_target(detector_kind))
        source_raw = st.text_input("Video source", value="0")
        count_mode = st.radio("Counting mode", ["Line crossing", "Zone entry"], horizontal=True)
        direction = st.selectbox(
            "Line direction",
            ["Any", "Left to right", "Right to left", "Top to bottom", "Bottom to top"],
            disabled=count_mode != "Line crossing",
        )
        yolo_confidence = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
        max_frames = st.number_input("Frames per run", min_value=1, max_value=10000, value=300)
        save_snapshots = st.checkbox("Save event snapshots", value=True)

    if "calibration_frame" not in st.session_state:
        st.session_state.calibration_frame = None
    if "line" not in st.session_state:
        st.session_state.line = None
    if "zone" not in st.session_state:
        st.session_state.zone = None

    left, right = st.columns([1.05, 0.95])

    with left:
        if st.button("Capture calibration frame", use_container_width=True):
            ok, frame, error = capture_one_frame(source_raw)
            if ok:
                st.session_state.calibration_frame = frame
            else:
                st.error(error)

        if st.session_state.calibration_frame is not None:
            display_frame, scale_x, scale_y = _canvas_frame(st.session_state.calibration_frame)
            drawing_mode = "line" if count_mode == "Line crossing" else "rect"
            if st_canvas is None:
                st.error("Install streamlit-drawable-canvas to calibrate by drawing.")
            else:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 180, 0, 0.18)",
                    stroke_width=3,
                    stroke_color="#ffe600" if drawing_mode == "line" else "#00b4ff",
                    background_image=Image.fromarray(display_frame),
                    update_streamlit=True,
                    height=display_frame.shape[0],
                    width=display_frame.shape[1],
                    drawing_mode=drawing_mode,
                    key=f"canvas-{count_mode}",
                )
                obj = _last_canvas_object(canvas_result)
                if count_mode == "Line crossing":
                    line = _extract_line(obj, scale_x, scale_y)
                    if line is not None:
                        st.session_state.line = line
                else:
                    zone = _extract_rect(obj, scale_x, scale_y)
                    if zone is not None:
                        st.session_state.zone = zone

            if st.button("Test detector on calibration frame", use_container_width=True):
                try:
                    detector = _cached_detector(
                        detector_kind,
                        str(config.locateanything_root),
                        config.locateanything_model_path,
                        config.locateanything_device,
                        str(config.yolo_weights_path),
                        yolo_confidence,
                    )
                    detections = detector.detect(
                        st.session_state.calibration_frame,
                        target_class.strip(),
                    )
                    preview_tracks = _preview_tracks(detections)
                    annotated = draw_overlay(
                        st.session_state.calibration_frame,
                        preview_tracks,
                        len(detections),
                        "Detections",
                        line=st.session_state.line if count_mode == "Line crossing" else None,
                        zone=st.session_state.zone if count_mode == "Zone entry" else None,
                    )
                    st.image(_as_rgb_image(annotated), channels="RGB", use_column_width=True)
                    st.write(f"Detections found: {len(detections)}")
                    if detections:
                        st.dataframe(
                            [
                                {
                                    "label": det.label,
                                    "score": det.score,
                                    "x1": round(det.x1, 1),
                                    "y1": round(det.y1, 1),
                                    "x2": round(det.x2, 1),
                                    "y2": round(det.y2, 1),
                                }
                                for det in detections
                            ],
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"Detector test failed: {exc}")

        calibration_ready = (
            st.session_state.line is not None
            if count_mode == "Line crossing"
            else st.session_state.zone is not None
        )
        st.write("Calibration:", "ready" if calibration_ready else "missing")

    with right:
        st.write("Model paths")
        st.code(
            "\n".join(
                [
                    f"LOCATEANYTHING_ROOT={config.locateanything_root}",
                    f"LOCATEANYTHING_MODEL_PATH={config.locateanything_model_path}",
                    f"LOCATEANYTHING_DEVICE={config.locateanything_device}",
                    f"YOLO_WEIGHTS_PATH={config.yolo_weights_path}",
                ]
            )
        )

    preview = st.empty()
    metrics = st.empty()
    events_box = st.empty()

    start = st.button("Start counting", type="primary", use_container_width=True)
    if not start:
        return

    if not target_class.strip():
        st.error("Target class is required.")
        return
    if count_mode == "Line crossing" and st.session_state.line is None:
        st.error("Capture a calibration frame and draw a line first.")
        return
    if count_mode == "Zone entry" and st.session_state.zone is None:
        st.error("Capture a calibration frame and draw a zone first.")
        return

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    storage = SessionStorage(config.sessions_dir, session_id, save_snapshots)
    counter = ObjectCounter(session_id)
    tracker = ByteTrackTracker()

    try:
        detector = _cached_detector(
            detector_kind,
            str(config.locateanything_root),
            config.locateanything_model_path,
            config.locateanything_device,
            str(config.yolo_weights_path),
            yolo_confidence,
        )
    except Exception as exc:
        st.error(f"Could not load detector: {exc}")
        return

    cap = cv2.VideoCapture(parse_video_source(source_raw))
    if not cap.isOpened():
        st.error(f"Could not open video source: {source_raw}")
        return

    recent_events = []
    frame_count = 0
    started_at = time.perf_counter()

    try:
        while frame_count < int(max_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                st.warning("Video source stopped returning frames.")
                break

            frame_count += 1
            detections = detector.detect(frame, target_class.strip())
            tracks = tracker.update(detections)

            if count_mode == "Line crossing":
                events = counter.update_line(
                    tracks,
                    target_class.strip(),
                    st.session_state.line,
                    direction,
                )
                annotated = draw_overlay(
                    frame,
                    tracks,
                    counter.total,
                    "Line count",
                    line=st.session_state.line,
                )
            else:
                events = counter.update_zone(
                    tracks,
                    target_class.strip(),
                    st.session_state.zone,
                )
                annotated = draw_overlay(
                    frame,
                    tracks,
                    counter.total,
                    "Zone count",
                    zone=st.session_state.zone,
                )

            for event in events:
                saved_event = storage.write_event(event, annotated)
                recent_events.insert(0, saved_event)
                recent_events = recent_events[:10]

            elapsed = max(time.perf_counter() - started_at, 1e-6)
            preview.image(_as_rgb_image(annotated), channels="RGB", use_column_width=True)
            with metrics.container():
                count_col, frames_col, fps_col = st.columns(3)
                count_col.metric("Total count", counter.total)
                frames_col.metric("Frames", frame_count)
                fps_col.metric("FPS", f"{frame_count / elapsed:.2f}")
            if recent_events:
                events_box.dataframe(
                    [
                        {
                            "time": event.timestamp,
                            "track": event.track_id,
                            "class": event.target_class,
                            "mode": event.count_mode,
                            "score": event.score,
                        }
                        for event in recent_events
                    ],
                    use_container_width=True,
                )
    finally:
        cap.release()

    st.success(f"Session saved to {storage.session_dir}")


if __name__ == "__main__":
    main()
