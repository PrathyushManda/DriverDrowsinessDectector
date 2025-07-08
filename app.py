import cv2, time, threading, os
import sqlite3
import numpy as np
import mediapipe as mp
import streamlit as st
from datetime import datetime
from twilio.rest import Client
import winsound
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# --- REST API (FastAPI) in background thread ---
api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@api_app.get("/events")
def get_events():
    cursor.execute("SELECT timestamp, ear, event, pitch, yaw FROM events")
    rows = cursor.fetchall()
    return {"events": [
        {"timestamp": r[0], "ear": r[1], "event": r[2], "pitch": r[3], "yaw": r[4]}
        for r in rows
    ]}

def run_api():
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# start API
threading.Thread(target=run_api, daemon=True).start()

# --- Twilio SMS Notifications ---
TW_SID = os.getenv("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TW_FROM = os.getenv("TWILIO_FROM")
TW_TO = os.getenv("TWILIO_TO")
if TW_SID and TW_TOKEN:
    twilio_client = Client(TW_SID, TW_TOKEN)
else:
    twilio_client = None

# --- SQLite Persistent Storage ---
conn = sqlite3.connect("drowsy_events.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    ear REAL,
    event TEXT,
    pitch REAL,
    yaw REAL
)""")
conn.commit()

# --- MediaPipe Face Mesh Setup ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Head-pose landmarks
HP_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "mouth_left": 78,
    "mouth_right": 308
}
# 3D model points for head-pose
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose_tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye
    (43.3, 32.7, -26.0),      # right eye
    (-28.9, -28.9, -24.1),    # mouth_left
    (28.9, -28.9, -24.1)      # mouth_right
])

# --- Streamlit UI ---
st.set_page_config(page_title="Driver Drowsiness", layout="wide")
st.title("🚗 Driver Drowsiness Detection")

# Sidebar controls
st.sidebar.header("Controls")
EAR_THRESH = st.sidebar.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01)
DROWSY_SEC = st.sidebar.slider("Drowsy Duration (s)", 0.5, 3.0, 1.0, 0.5)
BLINK_SEC = st.sidebar.slider("Blink Max Duration (s)", 0.1, 0.5, 0.3, 0.1)
run_detector = st.sidebar.checkbox("Run Detection", value=True)

# Initialize state
FPS = 10
DROWSY_FRAMES = int(DROWSY_SEC * FPS)
BLINK_FRAMES = int(BLINK_SEC * FPS)
counter_closed = 0
count_blinks = 0
count_drowsy = 0
history_ear = []

# Video & Display
frame_disp = st.empty()
chart_disp = st.empty()
stats_disp = st.empty()
cap = cv2.VideoCapture(0)

while run_detector and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    start = time.time()

    # Preprocess & landmarks
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    ear = pitch = yaw = 0.0

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        # Compute EAR
        def comp_ear(idxs):
            pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
            A = np.linalg.norm(np.subtract(pts[1], pts[5]))
            B = np.linalg.norm(np.subtract(pts[2], pts[4]))
            C = np.linalg.norm(np.subtract(pts[0], pts[3]))
            return (A + B) / (2.0 * C) if C else 0
        ear = (comp_ear(LEFT_EYE) + comp_ear(RIGHT_EYE)) / 2.0
        history_ear.append(ear)

        # Head-pose estimation
        pts2d = np.array([
            [lm[HP_LANDMARKS['nose_tip']].x*w, lm[HP_LANDMARKS['nose_tip']].y*h],
            [lm[HP_LANDMARKS['chin']].x*w, lm[HP_LANDMARKS['chin']].y*h],
            [lm[HP_LANDMARKS['left_eye']].x*w, lm[HP_LANDMARKS['left_eye']].y*h],
            [lm[HP_LANDMARKS['right_eye']].x*w, lm[HP_LANDMARKS['right_eye']].y*h],
            [lm[HP_LANDMARKS['mouth_left']].x*w, lm[HP_LANDMARKS['mouth_left']].y*h],
            [lm[HP_LANDMARKS['mouth_right']].x*w, lm[HP_LANDMARKS['mouth_right']].y*h]
        ], dtype=np.float64)
        focal = w
        cam_mtx = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]])
        dist = np.zeros((4,1))
        _, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts2d, cam_mtx, dist)
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, _ = angles

    # Detection logic
    if ear < EAR_THRESH:
        counter_closed += 1
    else:
        if 1 <= counter_closed <= BLINK_FRAMES:
            count_blinks += 1
        counter_closed = 0

    # On drowsy
    if counter_closed >= DROWSY_FRAMES:
        count_drowsy += 1
        cursor.execute(
            "INSERT INTO events(timestamp, ear, event, pitch, yaw) VALUES(?,?,?,?,?)",
            (datetime.now().isoformat(), ear, 'DROWSY', pitch, yaw)
        )
        conn.commit()
        if twilio_client:
            twilio_client.messages.create(
                body=f"Drowsy detected! EAR={ear:.2f}",
                from_=TW_FROM, to=TW_TO
            )
        # Audio alert using winsound
        threading.Thread(
            target=lambda: winsound.Beep(1000, 500),
            daemon=True
        ).start()
        counter_closed = 0

    # Visual cue
    if counter_closed >= DROWSY_FRAMES:
        frame = cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 20)

    # Display in Streamlit
    frame_disp.image(frame[:, :, ::-1], channels='BGR')
    chart_disp.line_chart(history_ear[-100:])
    stats_disp.text(
        f"EAR: {ear:.2f} | Pitch: {pitch:.1f} | Yaw: {yaw:.1f}\n"
        f"Blinks: {count_blinks} | Drowsy: {count_drowsy} | ClosedFrames: {counter_closed}"
    )

    # Maintain FPS
    elapsed = time.time() - start
    if elapsed < 1/FPS:
        time.sleep((1/FPS) - elapsed)

cap.release()
