#!/usr/bin/env python3
"""
stream_light.py

Ban cuc nhe cho Raspberry Pi Zero 2W:
- Stream MJPEG qua Flask
- Detect mat bang Haar Cascade cua OpenCV
- Trich embedding bang output_model.tflite
- So sanh face_database.json
- Khong GPIO, khong cam bien, khong UI phuc tap

Chay Pi Camera:
    python3 stream_light.py

Chay USB camera:
    python3 stream_light.py --use-usb-camera --camera 0

Mo browser:
    http://<PI_IP>:5000
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify


try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "output_model.tflite"
DB_PATH = BASE_DIR / "face_database.json"

EXPECTED_EMB_DIM = 128

DEFAULT_THRESHOLD = 0.85


# =========================
# MobileFaceNet
# =========================
class MobileFaceNet:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay model: {model_path}")

        self.interp = Interpreter(model_path=str(model_path), num_threads=2)
        self.interp.allocate_tensors()

        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]

        self.input_shape = self.inp["shape"]
        self.input_dtype = self.inp["dtype"]
        self.output_shape = self.outp["shape"]

        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])
        self.embedding_dim = int(self.output_shape[-1])

        print("[INFO] MobileFaceNet ready")
        print("[INFO] Model :", model_path)
        print("[INFO] Input :", self.input_shape, self.input_dtype)
        print("[INFO] Output:", self.output_shape)

    def embed(self, face_bgr):
        face = cv2.resize(face_bgr, (self.input_w, self.input_h))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.float32:
            face = face.astype(np.float32)
            face = (face - 127.5) / 128.0
        else:
            face = face.astype(self.input_dtype)

        face = np.expand_dims(face, axis=0)

        self.interp.set_tensor(self.inp["index"], face)
        self.interp.invoke()

        emb = self.interp.get_tensor(self.outp["index"])[0]
        emb = emb.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb


# =========================
# Database
# =========================
def load_database(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"Khong tim thay database: {db_path}")

    with open(db_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    db = {}

    for name, value in raw.items():
        if isinstance(value, list):
            embs = value
        elif isinstance(value, dict) and "embeddings" in value:
            embs = value["embeddings"]
        else:
            continue

        valid = []

        for e in embs:
            arr = np.array(e, dtype=np.float32)

            if arr.ndim != 1 or arr.shape[0] != EXPECTED_EMB_DIM:
                print(f"[WARN] Bo qua embedding sai dim cua {name}: {arr.shape}")
                continue

            arr = arr / (np.linalg.norm(arr) + 1e-8)
            valid.append(arr)

        if valid:
            db[name] = valid

    total = sum(len(v) for v in db.values())

    print(f"[INFO] Database: {len(db)} nguoi, {total} embeddings")

    if total == 0:
        raise RuntimeError("Database rong hoac khong co embedding 128 chieu hop le.")

    return db


def recognize(embedding, db, threshold):
    best_name = None
    best_sim = -1.0

    for name, embs in db.items():
        for e in embs:
            if e.shape != embedding.shape:
                continue

            sim = float(np.dot(embedding, e))

            if sim > best_sim:
                best_sim = sim
                best_name = name

    if best_sim >= threshold:
        return best_name, best_sim

    return None, best_sim


# =========================
# Haar face detector
# =========================
class HaarFaceDetector:
    def __init__(self):
        possible_paths = []

        # Cách 1: OpenCV bản đầy đủ có cv2.data
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            possible_paths.append(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # Cách 2: đường dẫn thường gặp trên Raspberry Pi / Debian
        possible_paths += [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ]

        cascade_path = None

        for p in possible_paths:
            if p and Path(p).exists():
                cascade_path = p
                break

        if cascade_path is None:
            raise RuntimeError(
                "Khong tim thay haarcascade_frontalface_default.xml.\n"
                "Cai them bang lenh:\n"
                "sudo apt install -y opencv-data\n"
                "Hoac dung ban YuNet nhe thay cho Haar."
            )

        self.detector = cv2.CascadeClassifier(str(cascade_path))

        if self.detector.empty():
            raise RuntimeError(f"Khong load duoc Haar Cascade: {cascade_path}")

        print(f"[INFO] Haar face detector ready: {cascade_path}")

    def detect_largest(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
        )

        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]

        return int(x), int(y), int(w), int(h)

    @staticmethod
    def crop_face(frame_bgr, box, pad_ratio=0.25):
        x, y, w, h = box

        pad = int(max(w, h) * pad_ratio)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)

        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

# =========================
# Camera
# =========================
def open_camera(args):
    if not args.use_usb_camera:
        try:
            from picamera2 import Picamera2

            class PiCamWrap:
                def __init__(self):
                    self.picam2 = Picamera2()

                    cfg = self.picam2.create_preview_configuration(
                        main={
                            "size": (args.width, args.height),
                            "format": "RGB888",
                        },
                        controls={
                            "FrameRate": args.fps,
                        },
                    )

                    self.picam2.configure(cfg)
                    self.picam2.start()
                    time.sleep(0.5)

                    print(f"[INFO] Pi Camera {args.width}x{args.height}@{args.fps}")

                def isOpened(self):
                    return True

                def read(self):
                    rgb = self.picam2.capture_array()

                    if rgb is None:
                        return False, None

                    #bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    return True, rgb

                def release(self):
                    self.picam2.stop()

            return PiCamWrap()

        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")
            print("[WARN] Chuyen sang USB camera")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print(f"[INFO] USB/OpenCV camera id={args.camera}")

    return cap


# =========================
# Shared state
# =========================
app = Flask(__name__)

_lock = threading.Lock()
_latest_jpeg = b""

_status = {
    "state": "starting",
    "name": "",
    "sim": 0.0,
    "fps": 0.0,
}


# =========================
# Inference loop
# =========================
def inference_loop(args):
    global _latest_jpeg, _status

    cap = open_camera(args)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera")
        with _lock:
            _status["state"] = "camera_error"
        return

    detector = HaarFaceDetector()
    embedder = MobileFaceNet(MODEL_PATH)
    db = load_database(DB_PATH)

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    frame_id = 0
    fps_count = 0
    fps_t0 = time.time()
    fps_display = 0.0

    last_box = None
    last_name = None
    last_sim = 0.0
    last_state = "idle"

    print("[INFO] Stream light started")
    print(f"[INFO] threshold={args.threshold}, infer_every={args.infer_every}")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.02)
            continue

        if args.flip:
            frame = cv2.flip(frame, 1)

        frame_id += 1
        fps_count += 1

        now = time.time()

        if fps_count >= 15:
            fps_display = fps_count / max(now - fps_t0, 1e-6)
            fps_t0 = now
            fps_count = 0

        # Chi detect + recognize moi N frame de giam tai CPU
        if frame_id % args.infer_every == 0:
            box = detector.detect_largest(frame)

            if box is not None:
                crop = detector.crop_face(frame, box)

                if crop is not None:
                    emb = embedder.embed(crop)
                    name, sim = recognize(emb, db, args.threshold)

                    last_box = box
                    last_name = name
                    last_sim = sim
                    last_state = "ok" if name else "unknown"

                else:
                    last_box = None
                    last_name = None
                    last_sim = 0.0
                    last_state = "no_face"
            else:
                last_box = None
                last_name = None
                last_sim = 0.0
                last_state = "no_face"

        # Ve overlay cuc nhe
        if args.draw:
            if last_box is not None:
                x, y, w, h = last_box

                if last_name:
                    color = (0, 220, 120)
                    text = f"{last_name} {last_sim:.2f}"
                else:
                    color = (40, 40, 220)
                    text = f"Unknown {last_sim:.2f}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"FPS {fps_display:.1f}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(".jpg", frame, encode_param)

        if ok:
            with _lock:
                _latest_jpeg = buf.tobytes()
                _status = {
                    "state": last_state,
                    "name": last_name or "",
                    "sim": round(float(last_sim), 4),
                    "fps": round(float(fps_display), 1),
                }

        # Giam CPU khi stream FPS thap
        if args.sleep > 0:
            time.sleep(args.sleep)


# =========================
# Flask routes
# =========================
@app.route("/")
def index():
    return """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Face Stream Light</title>
    <style>
        body {
            margin: 0;
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        img {
            width: 100vw;
            max-width: 800px;
        }
        pre {
            display: inline-block;
            text-align: left;
            background: #222;
            padding: 8px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <img src="/video_feed">
    <br>
    <pre id="s">loading...</pre>

    <script>
        async function update() {
            try {
                const r = await fetch('/status');
                const j = await r.json();
                document.getElementById('s').textContent = JSON.stringify(j, null, 2);
            } catch(e) {}
        }
        setInterval(update, 1000);
        update();
    </script>
</body>
</html>
"""


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _lock:
                frame = _latest_jpeg

            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with _lock:
        return jsonify(dict(_status))


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)

    p.add_argument("--use-usb-camera", action="store_true")
    p.add_argument("--camera", type=int, default=0)

    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=int, default=10)

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    # N frame moi detect/recognize 1 lan.
    # Tang len 8, 10, 12 neu Pi yeu.
    p.add_argument("--infer-every", type=int, default=8)

    # Chat luong JPEG cang thap cang nhe.
    p.add_argument("--jpeg-quality", type=int, default=50)

    # Ve box + text. Muon nhe hon nua thi them --no-draw.
    p.add_argument("--draw", dest="draw", action="store_true", default=True)
    p.add_argument("--no-draw", dest="draw", action="store_false")

    p.add_argument("--flip", action="store_true", default=True)
    p.add_argument("--no-flip", dest="flip", action="store_false")

    # Sleep moi vong lap de giam CPU.
    p.add_argument("--sleep", type=float, default=0.005)

    args = p.parse_args()

    t = threading.Thread(
        target=inference_loop,
        args=(args,),
        daemon=True,
    )
    t.start()

    print(f"[WEB] Open: http://<PI_IP>:{args.port}")

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
