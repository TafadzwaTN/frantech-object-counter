import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_counter.detectors import naturalize_label, normalize_label


def test_normalize_label_matches_hyphenated_model_names():
    assert normalize_label("Soil-Moisture-Sensor") == normalize_label("Soil Moisture Sensor")
    assert normalize_label("Arduino-Uno") == normalize_label("Arduino Uno")


def test_naturalize_label_converts_dataset_names_to_prompt_text():
    assert naturalize_label("Soil-Moisture-Sensor") == "Soil Moisture Sensor"
