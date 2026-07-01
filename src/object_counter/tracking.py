from __future__ import annotations

import numpy as np

from .detections import Detection, TrackedDetection


class ByteTrackTracker:
    def __init__(self):
        import supervision as sv

        self.sv = sv
        self.tracker = sv.ByteTrack()

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        if detections:
            xyxy = np.array([det.xyxy for det in detections], dtype=float)
            confidence = np.array(
                [det.score if det.score is not None else 1.0 for det in detections],
                dtype=float,
            )
            class_id = np.zeros(len(detections), dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=float)
            confidence = np.empty((0,), dtype=float)
            class_id = np.empty((0,), dtype=int)

        sv_detections = self.sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        tracked = self.tracker.update_with_detections(sv_detections)
        track_ids = tracked.tracker_id
        if track_ids is None:
            return []

        output: list[TrackedDetection] = []
        for i, track_id in enumerate(track_ids):
            if track_id is None or int(track_id) < 0:
                continue
            source = detections[i]
            output.append(
                TrackedDetection(
                    x1=source.x1,
                    y1=source.y1,
                    x2=source.x2,
                    y2=source.y2,
                    label=source.label,
                    score=source.score,
                    detector=source.detector,
                    track_id=int(track_id),
                )
            )
        return output
