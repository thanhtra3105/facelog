#!/usr/bin/env python3
"""
stream_light.py

Ban stream nhe cho Raspberry Pi Zero 2W:
- Stream MJPEG qua Flask
- Detect mat bang YuNet
- Align crop 112x112 bang 5 landmarks
- Trich embedding bang output_model.tflite
- So sanh face_database.json
- Khong GPIO, khong cam bien, khong UI phuc tap

Chay Pi Camera:
    python3 stream_light.py --width 320 --height 240 --fps 8 --infer-every 12 --jpeg-quality 40

Chay USB camera:
    python3 stream_light.py --use-usb-camera --camera 0 --width 320 --height 240 --fps 8 --infer-every 12 --jpeg-quality 40

Mo browser:
    http://<PI_IP>:5000
"""

import argparse
import json
import threading
import time
import urllib.request
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

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"

EXPECTED_EMB_DIM = 128
DEFAULT_THRESHOLD = 0.83


# =========================
# Download YuNet
# =========================
def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return

    print(f"[INFO] Download YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] YuNet downloaded: {path.stat().st_size // 1024} KB")


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
# YuNet Detector
# =========================
class YuNetDetector:
    def __init__(self, score_threshold=0.75):
        ensure_yunet(YUNET_PATH)

        self.det = cv2.FaceDetectorYN_create(
            str(YUNET_PATH),
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )

        self.score_threshold = score_threshold

        print(f"[INFO] YuNet ready score_threshold={score_threshold}")

    def detect_largest(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self.det.setInputSize((w, h))

        try:
            _, faces = self.det.detect(frame_bgr)
        except cv2.error as e:
            print(f"[WARN] YuNet detect error: {e}")
            return None

        if faces is None or len(faces) == 0:
            return None

        best = None
        best_area = 0

        for f in faces:
            if not np.all(np.isfinite(f)):
                continue

            x, y, bw, bh = [int(v) for v in f[:4]]
            score = float(f[-1])

            if score < self.score_threshold:
                continue

            if bw <= 5 or bh <= 5:
                continue

            area = bw * bh

            if area > best_area:
                landmarks = f[4:14].reshape(5, 2).astype(np.float32)

                best = {
                    "box": (x, y, bw, bh),
                    "landmarks": landmarks,
                    "score": score,
                }

                best_area = area

        return best

    @staticmethod
    def align_crop(frame_bgr, landmarks, size=112):
        dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        try:
            M, _ = cv2.estimateAffinePartial2D(
                landmarks.astype(np.float32),
                dst,
                method=cv2.LMEDS,
            )

            if M is None:
                return None

            face = cv2.warpAffine(
                frame_bgr,
                M,
                (size, size),
                borderValue=0.0,
            )

            return face

        except cv2.error:
            return None

    @staticmethod
    def crop_fallback(frame_bgr, box, pad_ratio=0.18):
        x, y, w, h = box
        pad = int(max(w, h) * pad_ratio)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)

        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return cv2.resize(crop, (112, 112))

    def crop(self, frame_bgr, face):
        crop = self.align_crop(frame_bgr, face["landmarks"])

        if crop is not None:
            return crop

        return self.crop_fallback(frame_bgr, face["box"])


# =========================
# Camera
# =========================
def open_camera(args):
    if not args.use_usb_camera:
        try:
            from picamera2 import Picamera2

            class PiCamWrap:
                def __init__(self):
                    print("[INFO] Init PiCamera2...")
                    self.picam2 = Picamera2()

                    print("[INFO] Create camera config...")
                    cfg = self.picam2.create_video_configuration(
                        main={
                            "size": (args.width, args.height),
                            "format": "RGB888",
                        },
                        controls={
                            "FrameRate": args.fps,
                            "FrameDurationLimits": (
                                int(1000000 / args.fps),
                                int(1000000 / args.fps),
                            ),
                        },
                        buffer_count=2,
                    )

                    print("[INFO] Configure camera...")
                    self.picam2.configure(cfg)

                    print("[INFO] Start camera...")
                    self.picam2.start()

                    time.sleep(1.0)

                    print(f"[INFO] Pi Camera ready {args.width}x{args.height}@{args.fps}")

                def isOpened(self):
                    return True

                def read(self):
                    try:
                        rgb = self.picam2.capture_array("main")
                    except Exception as e:
                        print(f"[WARN] capture_array loi: {e}")
                        return False, None

                    if rgb is None:
                        return False, None

                    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    return True, rgb

                def release(self):
                    try:
                        self.picam2.stop()
                    except Exception:
                        pass

            return PiCamWrap()

        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")
            print("[WARN] Chuyen sang USB camera/OpenCV")

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
    "face_score": 0.0,
}


# =========================
# Inference loop
# =========================
def inference_loop(args):
    global _latest_jpeg, _status

    print("[INFO] Opening camera...")
    cap = open_camera(args)
    print("[INFO] Camera opened")

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera")
        with _lock:
            _status["state"] = "camera_error"
        return

    print("[INFO] Loading YuNet...")
    detector = YuNetDetector(score_threshold=args.score_threshold)

    print("[INFO] Loading MobileFaceNet...")
    embedder = MobileFaceNet(MODEL_PATH)

    print("[INFO] Loading database...")
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
    last_face_score = 0.0

    last_infer_time = 0.0

    print("[INFO] Stream light YuNet started")
    print(
        f"[INFO] threshold={args.threshold}, "
        f"score_threshold={args.score_threshold}, "
        f"infer_every={args.infer_every}"
    )

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

        do_infer = frame_id % args.infer_every == 0

        # Gioi han tan suat inference neu can
        if args.min_infer_interval > 0:
            if now - last_infer_time < args.min_infer_interval:
                do_infer = False

        if do_infer:
            last_infer_time = now

            face = detector.detect_largest(frame)

            if face is not None:
                crop = detector.crop(frame, face)

                if crop is not None:
                    emb = embedder.embed(crop)
                    name, sim = recognize(emb, db, args.threshold)

                    last_box = face["box"]
                    last_name = name
                    last_sim = sim
                    last_face_score = face["score"]
                    last_state = "ok" if name else "unknown"

                else:
                    last_box = None
                    last_name = None
                    last_sim = 0.0
                    last_face_score = 0.0
                    last_state = "crop_error"

            else:
                last_box = None
                last_name = None
                last_sim = 0.0
                last_face_score = 0.0
                last_state = "no_face"

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
                    "face_score": round(float(last_face_score), 4),
                }

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
    <title>Face Stream Light YuNet</title>
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
    p.add_argument("--fps", type=int, default=8)

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    # YuNet confidence threshold
    p.add_argument("--score-threshold", type=float, default=0.75)

    # N frame moi detect/recognize 1 lan.
    p.add_argument("--infer-every", type=int, default=12)

    # Them gioi han thoi gian giua 2 lan inference, 0 la tat.
    p.add_argument("--min-infer-interval", type=float, default=0.0)

    # Chat luong JPEG cang thap cang nhe.
    p.add_argument("--jpeg-quality", type=int, default=40)

    p.add_argument("--draw", dest="draw", action="store_true", default=True)
    p.add_argument("--no-draw", dest="draw", action="store_false")

    p.add_argument("--flip", action="store_true", default=True)
    p.add_argument("--no-flip", dest="flip", action="store_false")

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
