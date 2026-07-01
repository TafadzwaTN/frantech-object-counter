# Frantech Object Counter

Local proof of concept for counting objects passing through a stationary camera frame. The first MVP supports webcam input, line-crossing counts, zone-entry counts, LocateAnything detection, and YOLO detection.

## Setup

```powershell
cd "C:\Users\Frantech Tafadzwa\Machine Learning (ML)\frantech-object-counter"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` if your model paths differ:

```text
LOCATEANYTHING_ROOT="C:\Users\Frantech Tafadzwa\Machine Learning (ML)\eagle\Embodied"
LOCATEANYTHING_MODEL_PATH=nvidia/LocateAnything-3B
YOLO_WEIGHTS_PATH="C:\Users\Frantech Tafadzwa\Machine Learning (ML)\Counter ML\yolo26n.pt"
```

## Run

```powershell
streamlit run app.py
```

Default video source is webcam index `0`. RTSP feeds can be entered in the same source field later, for example `rtsp://user:pass@camera-ip/stream`.

## Workflow

1. Select a detector.
2. Enter one target class.
3. Choose `Line crossing` or `Zone entry`.
4. Capture a calibration frame.
5. Draw the line or zone on the snapshot.
6. Start counting.

Events are saved under `sessions/<session-id>/events.csv`. If snapshots are enabled, event frames are saved under `sessions/<session-id>/snapshots/`.

## GitHub Push

Create an empty private GitHub repo named `frantech-object-counter`, then run:

```powershell
git remote add origin https://github.com/<your-org-or-user>/frantech-object-counter.git
git push -u origin main
```

## Model License Note

The LocateAnything source code is Apache-licensed, but the released LocateAnything model weights in the cloned repo are marked non-commercial/research-evaluation only. Do not use those weights in a commercial product without separate rights or a replacement model.
