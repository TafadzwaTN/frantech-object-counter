from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppConfig:
    locateanything_root: Path
    locateanything_model_path: str
    yolo_weights_path: Path
    sessions_dir: Path


def load_config() -> AppConfig:
    load_dotenv(PROJECT_ROOT / ".env")

    return AppConfig(
        locateanything_root=Path(
            os.getenv(
                "LOCATEANYTHING_ROOT",
                r"C:\Users\Frantech Tafadzwa\Machine Learning (ML)\eagle\Embodied",
            )
        ),
        locateanything_model_path=os.getenv(
            "LOCATEANYTHING_MODEL_PATH", "nvidia/LocateAnything-3B"
        ),
        yolo_weights_path=Path(
            os.getenv(
                "YOLO_WEIGHTS_PATH",
                r"C:\Users\Frantech Tafadzwa\Machine Learning (ML)\Counter ML\yolo26n.pt",
            )
        ),
        sessions_dir=PROJECT_ROOT / "sessions",
    )
