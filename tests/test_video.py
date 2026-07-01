import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_counter.video import apply_frame_rotation, resolve_frame_rotation


def test_auto_rotation_corrects_rtsp_only():
    assert resolve_frame_rotation("0", "auto") == 0
    assert resolve_frame_rotation("rtsp://camera.local/stream", "auto") == 90
    assert resolve_frame_rotation("rtsps://camera.local/stream", "auto") == 90


def test_manual_rotation_values():
    assert resolve_frame_rotation("0", "0") == 0
    assert resolve_frame_rotation("0", "90 clockwise") == 90
    assert resolve_frame_rotation("0", "180°") == 180
    assert resolve_frame_rotation("0", "270 cw") == 270


def test_apply_frame_rotation_90_clockwise():
    frame = np.array(
        [
            [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            [[4, 0, 0], [5, 0, 0], [6, 0, 0]],
        ],
        dtype=np.uint8,
    )

    rotated = apply_frame_rotation(frame, 90)

    assert rotated.shape == (3, 2, 3)
    assert rotated[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]
