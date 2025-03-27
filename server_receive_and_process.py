# server_receive_and_process.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time

app = Flask(__name__)
AUTHORIZED_TOKEN = "secret123"  # Basic API key (you can improve this with JWT)

log_data = []
start_time = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


def get_pixel_coords(landmark, img_width, img_height):
    return int(landmark.x * img_width), int(landmark.y * img_height)


def get_eye_ratio(eye_corner1, eye_corner2, iris):
    eye_width = abs(eye_corner2[0] - eye_corner1[0])
    iris_offset = iris[0] - eye_corner1[0]
    return iris_offset / eye_width if eye_width != 0 else 0.5


@app.route('/frame', methods=['POST'])
def receive_frame():
    if request.headers.get("Authorization") != f"Bearer {AUTHORIZED_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files['frame']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape
    results = face_mesh.process(rgb_frame)

    gaze_direction = "Not Detected"
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        l_start = get_pixel_coords(landmarks[33], img_w, img_h)
        l_end = get_pixel_coords(landmarks[133], img_w, img_h)
        l_iris = get_pixel_coords(landmarks[468], img_w, img_h)

        r_start = get_pixel_coords(landmarks[362], img_w, img_h)
        r_end = get_pixel_coords(landmarks[263], img_w, img_h)
        r_iris = get_pixel_coords(landmarks[473], img_w, img_h)

        l_ratio = get_eye_ratio(l_start, l_end, l_iris)
        r_ratio = get_eye_ratio(r_start, r_end, r_iris)
        avg_ratio = (l_ratio + r_ratio) / 2

        if avg_ratio < 0.35:
            gaze_direction = "Left"
        elif avg_ratio > 0.65:
            gaze_direction = "Right"
        else:
            gaze_direction = "Center"

    timestamp = round(time.time() - start_time, 2)
    log_data.append({
        "Time (s)": timestamp,
        "Gaze": gaze_direction
    })

    print(f"[{timestamp}s] Gaze: {gaze_direction}")
    return jsonify({"gaze": gaze_direction})


@app.route('/save', methods=['GET'])
def save_log():
    pd.DataFrame(log_data).to_csv("gaze_log.csv", index=False)
    return "Saved gaze_log.csv"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
