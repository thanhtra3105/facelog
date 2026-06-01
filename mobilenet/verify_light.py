#!/usr/bin/env python3
"""
verify_light.py

Ban verify toi uu nhe cho Raspberry Pi Zero 2W.
Dung MODEL CU: mobilefacenet.tflite

Chuc nang:
- Flask MJPEG stream nhe
- VL53L0X pause/resume khi khong co nguoi
- GPIO relay/LED
- YuNet detect + align 112x112
- MobileFaceNet model cu
- So sanh face_database.json

Chay test khong GPIO:
    python3 verify_light.py --no-gpio

Chay khong sensor:
    python3 verify_light.py --no-gpio --no-sensor

Chay USB camera:
    python3 verify_light.py --use-usb-camera --camera 0 --no-gpio --no-sensor

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
from flask import Flask, Response, jsonify, render_template

# =========================
# TFLite Interpreter
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

# MODEL CU
MODEL_PATH = BASE_DIR / "mobilefacenet.tflite"

DB_PATH = BASE_DIR / "face_database.json"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"

# Tu dong lay dim theo model cu, khong ep 128
EXPECTED_EMB_DIM = None

# Model cu cua ban truoc do dang dung threshold 0.85
DEFAULT_THRESHOLD = 0.85

PIN_RELAY = 23
PIN_LED_OK = 24
PIN_LED_FAIL = 25


# =========================
# Shared state
# =========================
app = Flask(__name__)

_lock = threading.Lock()

_latest_jpeg = b""

_shared = {
    "paused": True,
    "state": "idle",
    "name": "",
    "sim": 0.0,
    "fps": 0.0,
    "distance_mm": -1,
    "sensor_present": False,
    "sensor_ok": False,
    "last_log": None,
}


# =========================
# Utility
# =========================
def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return

    print(f"[INFO] Download YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] YuNet downloaded: {path.stat().st_size // 1024} KB")


def create_pause_jpeg(width: int, height: int, text: str = "WAITING") -> bytes:
    img = np.full((height, width, 3), 25, dtype=np.uint8)

    cv2.putText(
        img,
        text,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return buf.tobytes() if ok else b""


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
            for _ in range(2):
                self.led_fail.on()
                time.sleep(0.12)
                self.led_fail.off()
                time.sleep(0.12)

        threading.Thread(target=run, daemon=True).start()

    def close(self):
        for item in (self.relay, self.led_ok, self.led_fail):
            try:
                item.off()
                item.close()
            except Exception:
                pass


# =========================
# VL53L0X Sensor
# =========================
class VL53L0XSensor:
    def __init__(self, detect_distance_mm: int = 500):
        self.detect_distance_mm = detect_distance_mm
        self._sensor = None
        self._ok = False

        try:
            import board
            import busio
            import adafruit_vl53l0x

            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self._sensor.measurement_timing_budget = 200_000
            self._ok = True

            print(f"[INFO] VL53L0X ready, threshold={detect_distance_mm}mm")

        except Exception as e:
            print(f"[WARN] VL53L0X khoi dong loi: {e}")
            print("[WARN] Neu muon test khong sensor, chay them --no-sensor")

    @property
    def ok(self):
        return self._ok

    def read_mm(self):
        if not self._ok:
            return -1

        try:
            return int(self._sensor.range)
        except Exception as e:
            print(f"[WARN] VL53L0X read loi: {e}")
            return -1

    def person_present(self):
        mm = self.read_mm()

        if mm < 0:
            return False, -1

        return mm <= self.detect_distance_mm, mm


def sensor_loop(sensor, no_sensor: bool, hold_time: float, poll_interval: float):
    global _shared

    last_present_time = 0.0

    print(f"[SENSOR] start no_sensor={no_sensor}, hold={hold_time}, poll={poll_interval}")

    while True:
        now = time.time()

        if no_sensor or sensor is None:
            present, mm = True, -1
        else:
            present, mm = sensor.person_present()

        if present:
            last_present_time = now

        effective_present = (now - last_present_time) < hold_time

        with _lock:
            old_paused = _shared["paused"]

            _shared["paused"] = not effective_present
            _shared["sensor_present"] = present
            _shared["sensor_ok"] = no_sensor or (sensor is not None and sensor.ok)
            _shared["distance_mm"] = mm

            if old_paused and effective_present:
                _shared["last_log"] = f"[SENSOR] RESUME {mm}mm"
                print(_shared["last_log"])

            elif not old_paused and not effective_present:
                _shared["last_log"] = "[SENSOR] PAUSE"
                print(_shared["last_log"])

        time.sleep(poll_interval)


# =========================
# MobileFaceNet
# =========================
class MobileFaceNet:
    def __init__(self, model_path: Path, num_threads: int = 2):
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay model: {model_path}")

        try:
            self.interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
        except TypeError:
            self.interp = Interpreter(model_path=str(model_path))

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
        print("[INFO] Embedding dim:", self.embedding_dim)

        if EXPECTED_EMB_DIM is not None and self.embedding_dim != EXPECTED_EMB_DIM:
            print(f"[WARN] Model output dim={self.embedding_dim}, expected={EXPECTED_EMB_DIM}")

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
                best_area = area
                landmarks = f[4:14].reshape(5, 2).astype(np.float32)

                best = {
                    "box": (x, y, bw, bh),
                    "landmarks": landmarks,
                    "score": score,
                }

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

            return cv2.warpAffine(frame_bgr, M, (size, size), borderValue=0.0)

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
# Database
# =========================
def load_database(db_path: Path, expected_dim=None):
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

            if arr.ndim != 1:
                print(f"[WARN] Skip {name} embedding shape={arr.shape}")
                continue

            if expected_dim is not None and arr.shape[0] != expected_dim:
                print(
                    f"[WARN] Skip {name} embedding shape={arr.shape}, expected={expected_dim}"
                )
                continue

            arr = arr / (np.linalg.norm(arr) + 1e-8)
            valid.append(arr)

        if valid:
            db[name] = valid

    total = sum(len(v) for v in db.values())
    print(f"[INFO] DB: {len(db)} nguoi, {total} embeddings")

    if total == 0:
        raise RuntimeError("Database rong hoac khong co embedding hop le.")

    dims = sorted(set(int(e.shape[0]) for embs in db.values() for e in embs))
    print(f"[INFO] DB embedding dims: {dims}")

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

                    self.picam2.configure(cfg)
                    self.picam2.start()
                    time.sleep(0.8)

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

                    return True, rgb

                def release(self):
                    try:
                        self.picam2.stop()
                    except Exception:
                        pass

            return PiCamWrap()

        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")
            print("[WARN] Chuyen sang USB/OpenCV camera")

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
def draw_overlay(frame, bbox, name, sim, state, fps, dist_mm):
    h, w = frame.shape[:2]

    if state == "ok":
        color = (0, 220, 120)
        text = f"OK {name} {sim:.2f}"
    elif state == "deny":
        color = (40, 40, 220)
        text = f"DENY {sim:.2f}"
    elif state == "detecting":
        color = (0, 200, 255)
        text = "DETECTING"
    else:
        color = (180, 180, 180)
        text = "IDLE"

    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

    cv2.putText(
        frame,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    dist_str = f"{dist_mm}mm" if dist_mm > 0 else "--"

    cv2.putText(
        frame,
        f"FPS:{fps:.1f} D:{dist_str}",
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


# =========================
# Inference loop
# =========================
def inference_loop(args):
    global _latest_jpeg, _shared

    pause_frame = create_pause_jpeg(args.width, args.height, "NO PERSON")

    gpio = DoorGPIO(enabled=not args.no_gpio)

    print("[INFO] Loading YuNet...")
    detector = YuNetDetector(score_threshold=args.score_threshold)

    print("[INFO] Loading MobileFaceNet...")
    embedder = MobileFaceNet(MODEL_PATH, num_threads=args.tflite_threads)

    print("[INFO] Loading database...")
    db = load_database(Path(args.db), expected_dim=embedder.embedding_dim)

    print("[INFO] Opening camera...")
    cap = open_camera(args)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera")
        with _lock:
            _shared["state"] = "camera_error"
        return

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    frame_count = 0
    fps_count = 0
    fps_t0 = time.time()
    fps_display = 0.0

    last_verify = 0.0
    state = "idle"
    show_until = 0.0
    show_name = None
    show_sim = 0.0
    show_bbox = None

    print("[INFO] Verify light started")
    print(
        f"[INFO] model={MODEL_PATH.name}, "
        f"{args.width}x{args.height}@{args.fps}, "
        f"infer_every={args.infer_every}, threshold={args.threshold}"
    )

    while True:
        with _lock:
            paused = _shared["paused"]
            dist_mm = _shared["distance_mm"]

        if paused:
            with _lock:
                _latest_jpeg = pause_frame
                _shared["state"] = "idle"
                _shared["name"] = ""
                _shared["sim"] = 0.0
                _shared["fps"] = 0.0

            state = "idle"
            show_name = None
            show_bbox = None
            show_sim = 0.0
            last_verify = 0.0
            fps_t0 = time.time()
            fps_count = 0

            time.sleep(args.pause_sleep)
            continue

        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.02)
            continue

        if args.flip:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        fps_count += 1
        now = time.time()

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

        do_infer = frame_count % args.infer_every == 0

        if now - last_verify < args.cooldown:
            do_infer = False

        if do_infer:
            state = "detecting"

            face = detector.detect_largest(frame)

            if face is not None:
                crop = detector.crop(frame, face)

                if crop is not None:
                    emb = embedder.embed(crop)
                    name, sim = recognize(emb, db, args.threshold)

                    show_bbox = face["box"]
                    show_sim = sim

                    if name:
                        state = "ok"
                        show_name = name
                        show_until = now + args.unlock_duration + 0.5
                        last_verify = now

                        msg = f"[OK] {time.strftime('%H:%M:%S')} {name} sim={sim:.3f}"
                        print(msg)
                        new_log = msg

                        gpio.unlock(args.unlock_duration)

                    else:
                        state = "deny"
                        show_name = None
                        show_until = now + args.deny_show_time
                        last_verify = now

                        msg = f"[DENY] {time.strftime('%H:%M:%S')} sim={sim:.3f}"
                        print(msg)
                        new_log = msg

                        if args.blink_deny:
                            gpio.deny()

            else:
                state = "idle"
                show_bbox = None

        if args.draw:
            draw_overlay(
                frame,
                show_bbox,
                show_name,
                show_sim,
                state,
                fps_display,
                dist_mm,
            )

        ok, buf = cv2.imencode(".jpg", frame, encode_param)

        if ok:
            with _lock:
                _latest_jpeg = buf.tobytes()
                _shared.update(
                    {
                        "state": state,
                        "name": show_name or "",
                        "sim": round(float(show_sim), 4),
                        "fps": round(float(fps_display), 1),
                        "last_log": new_log,
                    }
                )

        if args.sleep > 0:
            time.sleep(args.sleep)


# =========================
# Flask
# =========================

@app.route("/")
def index():
    return render_template("ui.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _lock:
                frame = _latest_jpeg

            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with _lock:
        return jsonify(dict(_shared))


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
    p.add_argument("--score-threshold", type=float, default=0.75)

    p.add_argument("--db", default=str(DB_PATH))

    p.add_argument("--no-gpio", action="store_true")
    p.add_argument("--no-sensor", action="store_true")

    p.add_argument("--detect-distance", type=int, default=500)
    p.add_argument("--sensor-hold", type=float, default=5.0)
    p.add_argument("--sensor-poll", type=float, default=0.15)

    p.add_argument("--unlock-duration", type=float, default=5.0)
    p.add_argument("--deny-show-time", type=float, default=1.2)

    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--infer-every", type=int, default=12)

    p.add_argument("--jpeg-quality", type=int, default=40)

    p.add_argument("--draw", dest="draw", action="store_true", default=True)
    p.add_argument("--no-draw", dest="draw", action="store_false")

    p.add_argument("--flip", action="store_true", default=True)
    p.add_argument("--no-flip", dest="flip", action="store_false")

    p.add_argument("--blink-deny", action="store_true", default=False)

    p.add_argument("--sleep", type=float, default=0.005)
    p.add_argument("--pause-sleep", type=float, default=0.25)

    p.add_argument("--tflite-threads", type=int, default=2)

    args = p.parse_args()

    if args.no_sensor:
        sensor = None
        print("[INFO] --no-sensor: chay lien tuc")
    else:
        sensor = VL53L0XSensor(detect_distance_mm=args.detect_distance)

    threading.Thread(
        target=sensor_loop,
        args=(sensor, args.no_sensor, args.sensor_hold, args.sensor_poll),
        daemon=True,
    ).start()

    threading.Thread(
        target=inference_loop,
        args=(args,),
        daemon=True,
    ).start()

    print(f"[WEB] Open: http://<PI_IP>:{args.port}")

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
