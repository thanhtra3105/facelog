#!/usr/bin/env python3
"""
verify_web.py

Verify khuon mat + stream MJPEG len web browser qua Flask.
Ban nay da BO HOAN TOAN cam bien khoang cach VL53L0X.

Chay test khong GPIO:
    python3 verify_web.py --no-gpio

Chay voi USB camera:
    python3 verify_web.py --use-usb-camera --camera 0 --no-gpio

Chay voi Pi Camera CSI:
    python3 verify_web.py --no-gpio

Mo browser:
    http://<PI_IP>:5000
"""

import argparse
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify


os.environ["QT_QPA_PLATFORM"] = "xcb"


# =========================
# TFLite Interpreter import
# =========================
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


# =========================
# Paths / Config
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "output_model.tflite"
DB_PATH = BASE_DIR / "face_database.json"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"

DEFAULT_THRESHOLD = 0.55

PIN_RELAY = 23
PIN_LED_OK = 24
PIN_LED_FAIL = 25


# =========================
# Display check
# =========================
def has_display() -> bool:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test")
            return True
        except Exception:
            return False
    return False


USE_LOCAL_DISPLAY = has_display()


# =========================
# YuNet downloader
# =========================
def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return

    print(f"[INFO] Tai YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Xong: {path.stat().st_size // 1024} KB")


# =========================
# GPIO
# =========================
class MockOutput:
    def __init__(self, pin, name):
        self.pin = pin
        self.name = name
        print(f"[GPIO MOCK] {name} pin={pin}")

    def on(self):
        print(f"[GPIO MOCK] {self.name} ON")

    def off(self):
        print(f"[GPIO MOCK] {self.name} OFF")

    def close(self):
        pass


class DoorGPIO:
    def __init__(self, enabled=True):
        if not enabled:
            self.relay = MockOutput(PIN_RELAY, "RELAY")
            self.led_ok = MockOutput(PIN_LED_OK, "LED_OK")
            self.led_fail = MockOutput(PIN_LED_FAIL, "LED_FAIL")
            return

        try:
            from gpiozero import OutputDevice, LED

            self.relay = OutputDevice(PIN_RELAY, active_high=False, initial_value=False)
            self.led_ok = LED(PIN_LED_OK)
            self.led_fail = LED(PIN_LED_FAIL)

            self.relay.off()
            self.led_ok.off()
            self.led_fail.off()

            print(f"[INFO] GPIO ready relay={PIN_RELAY}")

        except Exception as e:
            print(f"[WARN] GPIO loi: {e}")
            self.relay = MockOutput(PIN_RELAY, "RELAY")
            self.led_ok = MockOutput(PIN_LED_OK, "LED_OK")
            self.led_fail = MockOutput(PIN_LED_FAIL, "LED_FAIL")

    def unlock(self, seconds):
        def run():
            self.relay.on()
            self.led_ok.on()
            time.sleep(seconds)
            self.relay.off()
            self.led_ok.off()

        threading.Thread(target=run, daemon=True).start()

    def deny(self):
        def run():
            for _ in range(3):
                self.led_fail.on()
                time.sleep(0.15)
                self.led_fail.off()
                time.sleep(0.15)

        threading.Thread(target=run, daemon=True).start()

    def close(self):
        for x in (self.relay, self.led_ok, self.led_fail):
            try:
                x.off()
                x.close()
            except Exception:
                pass


# =========================
# MobileFaceNet
# =========================
class MobileFaceNet:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

        self.interp = Interpreter(model_path=str(MODEL_PATH))
        self.interp.allocate_tensors()

        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]

        self.input_shape = self.inp["shape"]
        self.input_dtype = self.inp["dtype"]
        self.output_shape = self.outp["shape"]
        self.output_dtype = self.outp["dtype"]

        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])

        print("[INFO] MobileFaceNet ready")
        print("[INFO] Model path:", MODEL_PATH)
        print("[INFO] Input :", self.input_shape, self.input_dtype)
        print("[INFO] Output:", self.output_shape, self.output_dtype)

    def embed(self, face_bgr):
        """
        output_model.tflite:
            input  = [1, 112, 112, 3], float32
            output = [1, 128], float32
        """

        face = cv2.resize(face_bgr, (self.input_w, self.input_h))

        # OpenCV la BGR, model thuong train voi RGB
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
# YuNet Detector
# =========================
class YuNetDetector:
    def __init__(self, score_threshold=0.82):
        ensure_yunet(YUNET_PATH)

        self.det = cv2.FaceDetectorYN_create(
            str(YUNET_PATH),
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )

        print("[INFO] YuNet ready")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))

        _, faces = self.det.detect(frame)

        if faces is None:
            return []

        return [
            {
                "box": tuple(int(x) for x in f[:4]),
                "landmarks": f[4:14].reshape(5, 2).astype(np.float32),
                "score": float(f[-1]),
            }
            for f in faces
        ]

    @staticmethod
    def largest(faces):
        if not faces:
            return None

        return max(faces, key=lambda f: f["box"][2] * f["box"][3])

    @staticmethod
    def align_crop(frame, landmarks, size=112):
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

            return cv2.warpAffine(frame, M, (size, size), borderValue=0.0)

        except cv2.error:
            return None

    @staticmethod
    def crop_fallback(frame, face, pad_ratio=0.18):
        x, y, w, h = face["box"]
        pad = int(pad_ratio * max(w, h))

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return cv2.resize(crop, (112, 112))

    def crop(self, frame, face):
        c = self.align_crop(frame, face["landmarks"])
        return c if c is not None else self.crop_fallback(frame, face)


# =========================
# Database / Recognition
# =========================
def load_database():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay: {DB_PATH}\n"
            "Ban can tao face_database.json bang cung model output_model.tflite."
        )

    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    db = {}

    for name, value in raw.items():
        if isinstance(value, list):
            db[name] = [np.array(e, dtype=np.float32) for e in value]

        elif isinstance(value, dict) and "embeddings" in value:
            db[name] = [
                np.array(e, dtype=np.float32)
                for e in value["embeddings"]
            ]

    # Normalize database
    for name in db:
        normalized = []

        for e in db[name]:
            e = e.astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-8)
            normalized.append(e)

        db[name] = normalized

    total = sum(len(v) for v in db.values())
    print(f"[INFO] DB: {len(db)} nguoi, {total} embeddings")

    return db


def recognize(embedding, db, threshold):
    best_name = None
    best_sim = -1.0

    for name, embs in db.items():
        if not embs:
            continue

        sim = max(float(np.dot(embedding, e)) for e in embs)

        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_sim >= threshold:
        return best_name, best_sim

    return None, best_sim


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

                    # Doi RGB sang BGR cho OpenCV
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    return True, bgr

                def release(self):
                    self.picam2.stop()

            return PiCamWrap()

        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")
            print("[WARN] Thu chuyen sang USB camera/OpenCV")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print(f"[INFO] OpenCV camera id={args.camera}")

    return cap


# =========================
# Overlay
# =========================
def draw_overlay(frame, bbox, name, sim, state, fps=0.0):
    color = {
        "idle": (180, 180, 180),
        "detecting": (0, 200, 255),
        "ok": (0, 220, 120),
        "deny": (40, 40, 220),
    }.get(state, (180, 180, 180))

    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)

    if state == "ok":
        text = f"OK  {name}  sim={sim:.3f}"
    elif state == "deny":
        text = f"TU CHOI  sim={sim:.3f}"
    elif state == "detecting":
        text = "DANG NHAN DIEN..."
    else:
        text = "NHIN VAO CAMERA"

    cv2.putText(
        frame,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

    cv2.putText(
        frame,
        f"FPS:{fps:.1f}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )


def draw_local_ui(frame, bbox, name, sim, state, fps):
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    vid_h = int(h * 0.72)
    vid_w = w

    resized = cv2.resize(frame, (vid_w, vid_h))
    canvas[:vid_h, :vid_w] = resized

    if bbox:
        x, y, bw, bh = bbox
        sx = vid_w / frame.shape[1]
        sy = vid_h / frame.shape[0]

        color = {
            "ok": (0, 220, 120),
            "deny": (40, 40, 220),
            "detecting": (0, 200, 255),
        }.get(state, (180, 180, 180))

        cv2.rectangle(
            canvas,
            (int(x * sx), int(y * sy)),
            (int((x + bw) * sx), int((y + bh) * sy)),
            color,
            2,
        )

    color = {
        "ok": (0, 220, 120),
        "deny": (40, 40, 220),
        "detecting": (0, 200, 255),
        "idle": (180, 180, 180),
    }.get(state, (180, 180, 180))

    cv2.rectangle(canvas, (0, 0), (w, 44), (20, 20, 20), -1)

    if state == "ok":
        label = f"OK  {name}  sim={sim:.3f}"
    elif state == "deny":
        label = f"TU CHOI  sim={sim:.3f}"
    elif state == "detecting":
        label = "DANG NHAN DIEN..."
    else:
        label = "NHIN VAO CAMERA"

    cv2.putText(
        canvas,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )

    clock = time.strftime("%H:%M:%S")
    cv2.putText(
        canvas,
        clock,
        (w - 130, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (160, 160, 160),
        1,
    )

    cv2.putText(
        canvas,
        f"FPS:{fps:.1f}",
        (10, vid_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (120, 120, 120),
        1,
    )

    panel_y = vid_h
    panel_h = h - vid_h

    cv2.rectangle(canvas, (0, panel_y), (w, h), (17, 17, 17), -1)
    cv2.line(canvas, (0, panel_y), (w, panel_y), (40, 40, 40), 1)

    def draw_card(x, y, cw, ch, label, value, val_color=(220, 220, 220)):
        cv2.rectangle(canvas, (x + 3, y + 3), (x + cw - 3, y + ch - 3), (28, 28, 28), -1)
        cv2.rectangle(canvas, (x + 3, y + 3), (x + cw - 3, y + ch - 3), (45, 45, 45), 1)

        cv2.putText(
            canvas,
            label,
            (x + 10, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (100, 100, 100),
            1,
        )

        cv2.putText(
            canvas,
            value,
            (x + 10, y + ch - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            val_color,
            1,
        )

    cw = w // 3
    ch = panel_h

    state_color = {
        "ok": (0, 220, 120),
        "deny": (40, 40, 220),
        "detecting": (0, 200, 255),
        "idle": (100, 100, 100),
    }.get(state, (180, 180, 180))

    draw_card(0, panel_y, cw, ch, "TRANG THAI", state.upper(), state_color)
    draw_card(cw, panel_y, cw, ch, "TEN NHAN DIEN", name or "--")

    sim_str = f"{sim:.3f}" if sim > 0 else "--"
    draw_card(cw * 2, panel_y, cw, ch, "SIMILARITY", sim_str)

    return canvas


# =========================
# Shared state
# =========================
_lock = threading.Lock()
_jpeg_frame = b""

_shared = {
    "state": "idle",
    "name": "",
    "sim": 0.0,
    "fps": 0.0,
    "last_log": None,
}


# =========================
# Inference thread
# =========================
def inference_loop(args):
    global _jpeg_frame, _shared

    gpio = DoorGPIO(enabled=not args.no_gpio)
    detector = YuNetDetector(score_threshold=args.score_threshold)
    embedder = MobileFaceNet()
    db = load_database()
    cap = open_camera(args)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera")
        return

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    frame_count = 0
    infer_every = args.infer_every
    last_verify = 0.0

    state = "idle"
    show_until = 0.0
    show_name = None
    show_sim = 0.0
    show_bbox = None

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    print(f"[INFO] Inference loop start. infer_every={infer_every}")

    if USE_LOCAL_DISPLAY:
        cv2.namedWindow("FaceLog", cv2.WINDOW_NORMAL)
        cv2.moveWindow("FaceLog", 0, 0)
        cv2.setWindowProperty("FaceLog", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("[INFO] Local display fullscreen")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)

        frame_count += 1
        now = time.time()

        fps_count += 1

        if fps_count >= 15:
            fps_display = fps_count / max(now - fps_t0, 1e-6)
            fps_t0 = now
            fps_count = 0

        if now > show_until and state in ("ok", "deny"):
            state = "idle"
            show_name = None
            show_bbox = None
            show_sim = 0.0

        new_log = None

        if frame_count % infer_every == 0 and now - last_verify >= args.cooldown:
            state = "detecting"

            faces = detector.detect(frame)
            face = detector.largest(faces)

            if face is not None:
                crop = detector.crop(frame, face)

                if crop is not None:
                    emb = embedder.embed(crop)
                    name, sim = recognize(emb, db, args.threshold)

                    show_bbox = face["box"]
                    show_sim = sim

                    if name:
                        msg = f"[OK] {time.strftime('%H:%M:%S')} {name} sim={sim:.3f}"
                        print(msg)

                        new_log = msg
                        state = "ok"
                        show_name = name
                        show_until = now + args.unlock_duration + 0.5

                        gpio.unlock(args.unlock_duration)

                    else:
                        msg = f"[DENY] {time.strftime('%H:%M:%S')} sim={sim:.3f}"
                        print(msg)

                        new_log = msg
                        state = "deny"
                        show_name = None
                        show_until = now + 1.5

                        gpio.deny()

                    last_verify = now

            else:
                state = "idle"
                show_bbox = None

        if USE_LOCAL_DISPLAY:
            ui = draw_local_ui(
                frame,
                show_bbox,
                show_name,
                show_sim,
                state,
                fps_display,
            )

            cv2.imshow("FaceLog", ui)

            if cv2.waitKey(1) == 27:
                break

        else:
            draw_overlay(
                frame,
                show_bbox,
                show_name,
                show_sim,
                state,
                fps_display,
            )

            ok, buf = cv2.imencode(".jpg", frame, encode_param)

            if ok:
                with _lock:
                    _jpeg_frame = buf.tobytes()

                    _shared.update(
                        {
                            "state": state,
                            "name": show_name or "",
                            "sim": round(show_sim, 4),
                            "fps": round(fps_display, 1),
                            "last_log": new_log,
                        }
                    )

    cap.release()
    gpio.close()
    cv2.destroyAllWindows()


# =========================
# Flask
# =========================
app = Flask(__name__)


@app.route("/")
def index():
    html = """
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Face Verify</title>
        <style>
            body {
                margin: 0;
                background: #111;
                color: #eee;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h2 {
                margin: 16px 0 8px;
            }
            img {
                width: 96vw;
                max-width: 900px;
                border: 2px solid #333;
                border-radius: 12px;
            }
            pre {
                display: inline-block;
                text-align: left;
                background: #222;
                padding: 12px;
                border-radius: 8px;
                min-width: 300px;
            }
        </style>
    </head>
    <body>
        <h2>MobileFaceNet Verify</h2>
        <img src="/video_feed">
        <br>
        <pre id="status">Loading...</pre>

        <script>
            async function updateStatus() {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    document.getElementById('status').textContent =
                        JSON.stringify(data, null, 2);
                } catch (e) {
                    document.getElementById('status').textContent = e;
                }
            }

            setInterval(updateStatus, 500);
            updateStatus();
        </script>
    </body>
    </html>
    """
    return html


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _lock:
                frame = _jpeg_frame

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
        s = dict(_shared)

    return jsonify(s)


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--use-usb-camera", action="store_true")

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    p.add_argument("--no-gpio", action="store_true")

    p.add_argument("--unlock-duration", type=float, default=5.0)
    p.add_argument("--cooldown", type=float, default=2.0)

    p.add_argument("--score-threshold", type=float, default=0.82)
    p.add_argument("--infer-every", type=int, default=6)

    p.add_argument("--jpeg-quality", type=int, default=70)

    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="0.0.0.0")

    args = p.parse_args()

    if USE_LOCAL_DISPLAY:
        print("[INFO] Co man hinh -> Local display mode")
        inference_loop(args)
    else:
        print(f"[WEB] Mo browser: http://<PI_IP>:{args.port}")

        ti = threading.Thread(
            target=inference_loop,
            args=(args,),
            daemon=True,
        )
        ti.start()

        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            use_reloader=False,
        )


if __name__ == "__main__":
    main()
