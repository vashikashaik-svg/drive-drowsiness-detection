import csv
import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_FILE = os.path.join(BASE_DIR, "alerts.csv")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22
CLOSED_FRAMES = 20

state = {"status": "NO FACE", "ear": 0.0}


def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal), pts


def log_alert(ear):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["time", "ear"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(float(ear), 3),
        ])


def generate_frames():
    cap = cv2.VideoCapture(0)
    counter = 0
    was_drowsy = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = "NO FACE"
            color = (200, 200, 200)
            ear = 0.0

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left_ear, left_pts = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                right_ear, right_pts = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                for p in left_pts + right_pts:
                    cv2.circle(frame, p, 2, (0, 255, 0), -1)

                if ear < EAR_THRESHOLD:
                    counter += 1
                else:
                    counter = 0

                if counter >= CLOSED_FRAMES:
                    status = "DROWSY"
                    color = (0, 0, 255)
                    if not was_drowsy:
                        log_alert(ear)
                        was_drowsy = True
                else:
                    status = "ALERT"
                    color = (0, 255, 0)
                    was_drowsy = False
            else:
                counter = 0
                was_drowsy = False

            state["status"] = status
            state["ear"] = round(float(ear), 3)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, status, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/video")
def video():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def get_status():
    return JSONResponse(state)


@app.get("/alerts")
def get_alerts():
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="") as f:
            rows = list(csv.DictReader(f))
    return JSONResponse(rows[-20:][::-1])


@app.post("/alerts/clear")
def clear_alerts():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    return {"cleared": True}