from __future__ import annotations

import sys
import re
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import torch
from PIL import Image

from .config import AppConfig
from .detections import Detection


ALL_CLASSES = "All classes"


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def naturalize_label(label: str) -> str:
    return re.sub(r"[-_]+", " ", label).strip()


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
        self.class_names = [
            str(name) for _, name in sorted(self.model.names.items(), key=lambda item: item[0])
        ]

    def detect(self, frame_bgr, target_class: str) -> list[Detection]:
        results = self.model.predict(frame_bgr, conf=self.confidence, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names
        detections: list[Detection] = []
        target = target_class.strip()
        target_norm = normalize_label(target)
        filter_all = target_norm in ("", normalize_label(ALL_CLASSES), "all")

        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id))
            if not filter_all and normalize_label(label) != target_norm:
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

    def __init__(self, locateanything_root: Path, model_path: str, device: str = "auto"):
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
        self.device = self._resolve_device(device)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            self.worker = LocateAnythingWorker(model_path, device=self.device, dtype=dtype)
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise ModuleNotFoundError(
                "LocateAnything could not load because a model dependency is "
                f"missing ({missing!r}). Run `pip install -r requirements.txt` "
                "inside this project's virtual environment, then restart Streamlit."
            ) from exc
        except AssertionError as exc:
            if "Torch not compiled with CUDA enabled" in str(exc):
                raise RuntimeError(
                    "LocateAnything tried to use CUDA, but this virtual environment "
                    "has a CPU-only PyTorch build. Set LOCATEANYTHING_DEVICE=cpu in "
                    ".env, or install a CUDA-enabled PyTorch build and restart Streamlit."
                ) from exc
            raise

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = device.strip().lower()
        if requested in ("", "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LOCATEANYTHING_DEVICE=cuda was requested, but this virtual "
                "environment has a CPU-only PyTorch build. Install CUDA-enabled "
                "PyTorch or set LOCATEANYTHING_DEVICE=cpu."
            )
        if requested not in ("cpu", "cuda"):
            raise ValueError("LOCATEANYTHING_DEVICE must be 'auto', 'cuda', or 'cpu'.")
        return requested

    def detect(self, frame_bgr, target_class: str) -> list[Detection]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        prompt_label = naturalize_label(target_class)
        result = self.worker.detect(image, [prompt_label], verbose=False)
        boxes = self.worker_cls.parse_boxes(result["answer"], image.width, image.height)
        return [
            Detection(
                x1=float(box["x1"]),
                y1=float(box["y1"]),
                x2=float(box["x2"]),
                y2=float(box["y2"]),
                label=prompt_label,
                score=None,
                detector=self.name,
            )
            for box in boxes
        ]


def load_yolo_class_names(weights_path: Path) -> list[str]:
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    return [str(name) for _, name in sorted(model.names.items(), key=lambda item: item[0])]


def load_detector(kind: str, config: AppConfig, yolo_confidence: float = 0.25) -> Detector:
    if kind == "YOLO":
        return YoloDetector(config.yolo_weights_path, confidence=yolo_confidence)
    if kind == "LocateAnything":
        return LocateAnythingDetector(
            config.locateanything_root,
            config.locateanything_model_path,
            device=config.locateanything_device,
        )
    raise ValueError(f"Unsupported detector kind: {kind}")
