import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from object_counter.counting import ObjectCounter
from object_counter.detections import TrackedDetection


def _track(track_id, x1, y1, x2, y2):
    return TrackedDetection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        label="part",
        score=0.9,
        detector="test",
        track_id=track_id,
    )


def test_line_crossing_counts_track_once():
    counter = ObjectCounter("test-session")
    line = ((50, 0), (50, 100))

    assert counter.update_line([_track(1, 10, 10, 20, 20)], "part", line, "Any") == []
    events = counter.update_line([_track(1, 70, 10, 80, 20)], "part", line, "Any")
    assert len(events) == 1
    assert counter.total == 1

    assert counter.update_line([_track(1, 10, 10, 20, 20)], "part", line, "Any") == []
    assert counter.total == 1


def test_zone_entry_counts_track_once():
    counter = ObjectCounter("test-session")
    zone = (40, 40, 90, 90)

    assert counter.update_zone([_track(7, 0, 0, 10, 10)], "part", zone) == []
    events = counter.update_zone([_track(7, 50, 50, 60, 60)], "part", zone)
    assert len(events) == 1
    assert counter.total == 1

    assert counter.update_zone([_track(7, 60, 60, 70, 70)], "part", zone) == []
    assert counter.total == 1
