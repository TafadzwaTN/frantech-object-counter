from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
from PIL import Image

from .config import AppConfig
from .detections import Detection


class Detector(ABC):
    name: str

    @abstractmethod
    def detect(self, frame_bgr, target_class: str) -> list[Detection]:
        """Return detections for one BGR OpenCV frame."""


class YoloDetector(Detector):
    name = "YOLO"

    def __init__(self, weights_path: Path, confidence: float = 0.25):
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.confidence = confidence

    def detect(self, frame_bgr, target_class: str) -> list[Detection]:
        results = self.model.predict(frame_bgr, conf=self.confidence, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names
        detections: list[Detection] = []
        target_norm = target_class.strip().lower()

        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id))
            if target_norm and label.lower() != target_norm:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0])
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,
                    score=score,
                    detector=self.name,
                )
            )
        return detections


class LocateAnythingDetector(Detector):
    name = "LocateAnything"

    def __init__(self, locateanything_root: Path, model_path: str):
        if not locateanything_root.exists():
            raise FileNotFoundError(f"LocateAnything root not found: {locateanything_root}")
        root = str(locateanything_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            from locateanything_worker import LocateAnythingWorker
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise ModuleNotFoundError(
                "LocateAnything is missing a Python dependency "
                f"({missing!r}). Run `pip install -r requirements.txt` inside "
                "this project's virtual environment, then restart Streamlit."
            ) from exc

        self.worker_cls = LocateAnythingWorker
        try:
            self.worker = LocateAnythingWorker(model_path)
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise ModuleNotFoundError(
                "LocateAnything could not load because a model dependency is "
                f"missing ({missing!r}). Run `pip install -r requirements.txt` "
                "inside this project's virtual environment, then restart Streamlit."
            ) from exc

    def detect(self, frame_bgr, target_class: str) -> list[Detection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        result = self.worker.detect(image, [target_class], verbose=False)
        boxes = self.worker_cls.parse_boxes(result["answer"], image.width, image.height)
        return [
            Detection(
                x1=float(box["x1"]),
                y1=float(box["y1"]),
                x2=float(box["x2"]),
                y2=float(box["y2"]),
                label=target_class,
                score=None,
                detector=self.name,
            )
            for box in boxes
        ]


def load_detector(kind: str, config: AppConfig, yolo_confidence: float = 0.25) -> Detector:
    if kind == "YOLO":
        return YoloDetector(config.yolo_weights_path, confidence=yolo_confidence)
    if kind == "LocateAnything":
        return LocateAnythingDetector(
            config.locateanything_root, config.locateanything_model_path
        )
    raise ValueError(f"Unsupported detector kind: {kind}")
