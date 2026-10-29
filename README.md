# Driver Drowsiness Detection

Real-time drowsiness detection using a webcam, MediaPipe Face Mesh and the
Eye Aspect Ratio (EAR). FastAPI serves a live video page with status, an
alarm sound and an alert history.

## Requirements
- Python 3.11 (mediapipe 0.10.14 does not support newer Python versions)
- A webcam

## Setup
    py -3.11 -m venv .venv311
    .venv311\Scripts\activate
    pip install -r requirements.txt

## Run (browser version)
    uvicorn app:app

Open http://127.0.0.1:8000 and click "Enable Alarm Sound".

## Run (desktop window version, Windows)
    python drowsiness.py

Press q to quit.

## How it works
1. The webcam frame is passed to MediaPipe Face Mesh.
2. Six landmarks per eye give the Eye Aspect Ratio (EAR).
3. If EAR stays below 0.22 for 20 frames in a row, the driver is DROWSY.
4. Each drowsy event is saved to alerts.csv and shown in Alert History.

## Project structure
    app.py            FastAPI backend
    drowsiness.py     OpenCV desktop version
    static/           index.html, style.css, script.js
    alerts.csv        created automatically