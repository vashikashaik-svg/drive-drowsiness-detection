import cv2
import mediapipe as mp
import numpy as np
import winsound
import time

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
last_beep = 0


def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    p1, p2, p3, p4, p5, p6 = [np.array(p) for p in pts]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return vertical / (2.0 * horizontal), pts


cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "NO FACE"
    color = (200, 200, 200)

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
            status = "DROWSY! WAKE UP!"
            color = (0, 0, 255)
            if time.time() - last_beep > 1.5:
                winsound.Beep(2000, 500)
                last_beep = time.time()
        else:
            status = "ALERT"
            color = (0, 255, 0)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, status, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()