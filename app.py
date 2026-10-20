"""
app.py
------
Flask backend for the Sleep Detecting System demo.

Routes
------
GET  /            -> HUD dashboard page
GET  /video_feed  -> MJPEG stream of the annotated webcam feed
GET  /api/status  -> JSON snapshot of current EAR / drowsy state (polled by JS)
POST /api/reset   -> resets blink counter / session timer

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from detector import DrowsinessDetector

app = Flask(__name__)

detector = DrowsinessDetector()

_camera_lock = threading.Lock()
_camera = None


def get_camera():
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return _camera


def generate_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # mirror, feels natural on a "dashboard cam"
        frame = detector.process(frame)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(detector.state.snapshot())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    detector.state.blink_count = 0
    detector.state.session_start = time.time()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 60)
    print(" SLEEP DETECTING SYSTEM - starting local server")
    print(" Open http://127.0.0.1:5000 in your browser")
    print(" Grant webcam permission when your browser asks.")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
