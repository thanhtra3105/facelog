#!/usr/bin/env python3
"""
verify_web.py
Verify khuon mat + stream MJPEG len web browser qua Flask.
Tich hop VL53L0X de tu dong pause/resume khi khong co nguoi.

Ban nay chi thay model sang output_model.tflite.
Giao dien, pause frame, sensor, Flask template giu nhu code goc.

Chay:
    python3 verify_web.py --no-gpio
    python3 verify_web.py --use-usb-camera --camera 0 --no-gpio
    python3 verify_web.py --no-gpio --no-sensor   # test khong co sensor

Mo browser:
    http://<PI_IP>:5000
"""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path
import os

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

os.environ["QT_QPA_PLATFORM"] = "xcb"

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

# =========================
# MODEL MOI
# =========================
MODEL_PATH = BASE_DIR / "output_model.tflite"

DB_PATH = BASE_DIR / "face_database.json"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"

IMG_SIZE = 112

# Model moi nen test tu 0.55 - 0.70.
# Neu muon giu 0.85 thi chay: python3 verify_web.py --threshold 0.85
DEFAULT_THRESHOLD = 0.85

EXPECTED_EMB_DIM = 128

PIN_RELAY = 23
PIN_LED_OK = 24
PIN_LED_FAIL = 25

_PAUSE_IMG_BASE = None


def draw_local_ui(frame, bbox, name, sim, state, fps, dist_mm, sensor_present):
    """UI day du cho local display."""
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

    badge_text = "? Co nguoi" if sensor_present else "? Khong co nguoi"
    badge_color = (0, 200, 100) if sensor_present else (100, 100, 100)
    cv2.putText(
        canvas,
        badge_text,
        (w - 220, vid_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        badge_color,
        1,
    )

    dist_str = f"{dist_mm}mm" if dist_mm > 0 else "--"
    cv2.putText(
        canvas,
        f"FPS:{fps:.1f}  dist:{dist_str}",
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

    cw = w // 4
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

    draw_card(cw * 3, panel_y, cw, ch, "KHOANG CACH", dist_str)

    return canvas


def has_display() -> bool:
    import os

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test")
            return True
        except Exception:
            return False

    return False


USE_LOCAL_DISPLAY = has_display()


def ensure_yunet(path):
    if path.exists() and path.stat().st_size > 100_000:
        return

    print(f"[INFO] Tai YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Xong: {path.stat().st_size // 1024} KB")


# ── GPIO ──────────────────────────────────────
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


# ── VL53L0X sensor ────────────────────────────
class VL53L0XSensor:
    def __init__(self, detect_distance_mm: int = 500, i2c_bus: int = 1):
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

            print(f"[INFO] VL53L0X ready, nguong={detect_distance_mm}mm")

        except Exception as e:
            print(f"[WARN] VL53L0X khong khoi dong duoc: {e}")
            print("[WARN] Sensor loi -> se PAUSE (khong detect) cho den khi fix sensor")

    @property
    def ok(self) -> bool:
        return self._ok

    def read_mm(self) -> int:
        if not self._ok:
            return -1

        try:
            return self._sensor.range
        except Exception as e:
            print(f"[WARN] VL53L0X doc loi: {e}")
            return -1

    def person_present(self) -> tuple[bool, int]:
        mm = self.read_mm()

        if mm < 0:
            return False, -1

        return mm <= self.detect_distance_mm, mm


# ── ML models ─────────────────────────────────
class MobileFaceNet:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Khong tim thay {MODEL_PATH}")

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
        self.embedding_dim = int(self.output_shape[-1])

        print("[INFO] MobileFaceNet ready")
        print("[INFO] Model path:", MODEL_PATH)
        print("[INFO] Input :", self.input_shape, self.input_dtype)
        print("[INFO] Output:", self.output_shape, self.output_dtype)

        if self.embedding_dim != EXPECTED_EMB_DIM:
            print(f"[WARN] Model output dim = {self.embedding_dim}, expected {EXPECTED_EMB_DIM}")

    def embed(self, face_bgr):
        face = cv2.resize(face_bgr, (self.input_w, self.input_h))

        # OpenCV BGR -> model RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.float32:
            face = face.astype(np.float32)
            face = (face - 127.5) / 128.0
        else:
            face = face.astype(self.input_dtype)

        self.interp.set_tensor(self.inp["index"], face[np.newaxis])
        self.interp.invoke()

        emb = self.interp.get_tensor(self.outp["index"])[0]
        emb = emb.astype(np.float32)

        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb


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
        return max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None

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

        return cv2.resize(crop, (112, 112)) if crop.size else None

    def crop(self, frame, face):
        c = self.align_crop(frame, face["landmarks"])
        return c if c is not None else self.crop_fallback(frame, face)


def load_database():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay: {DB_PATH}")

    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    db = {}

    for name, value in raw.items():
        if isinstance(value, list):
            embs = [np.array(e, dtype=np.float32) for e in value]

        elif isinstance(value, dict) and "embeddings" in value:
            embs = [np.array(e, dtype=np.float32) for e in value["embeddings"]]

        else:
            continue

        valid_embs = []

        for e in embs:
            if len(e.shape) != 1:
                print(f"[WARN] Bo qua embedding loi cua {name}, shape={e.shape}")
                continue

            if e.shape[0] != EXPECTED_EMB_DIM:
                print(
                    f"[WARN] Bo qua embedding cu cua {name}, "
                    f"shape={e.shape}, khong phai {EXPECTED_EMB_DIM}"
                )
                continue

            e = e / (np.linalg.norm(e) + 1e-8)
            valid_embs.append(e.astype(np.float32))

        if valid_embs:
            db[name] = valid_embs

    total = sum(len(v) for v in db.values())
    print(f"[INFO] DB: {len(db)} nguoi, {total} embeddings")

    if total == 0:
        raise RuntimeError(
            "face_database.json khong co embedding 128 chieu hop le. "
            "Hay dang ky lai khuon mat bang model output_model.tflite."
        )

    return db


def recognize(embedding, db, threshold):
    best_name, best_sim = None, -1.0

    for name, embs in db.items():
        if not embs:
            continue

        sims = []

        for e in embs:
            if embedding.shape != e.shape:
                print(f"[WARN] Skip {name}: embedding {embedding.shape} != db {e.shape}")
                continue

            sim = float(np.dot(embedding, e))
            sims.append(sim)

        if not sims:
            continue

        sim = max(sims)

        if sim > best_sim:
            best_sim, best_name = sim, name

    return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)


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

                    # Picamera2 RGB888 -> OpenCV BGR
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    return True, bgr

                def release(self):
                    self.picam2.stop()

            return PiCamWrap()

        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print(f"[INFO] OpenCV camera id={args.camera}")

    return cap


def draw_overlay(frame, bbox, name, sim, state, fps=0.0, dist_mm=-1):
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

    dist_str = f"{dist_mm}mm" if dist_mm > 0 else "—"

    cv2.putText(
        frame,
        f"FPS:{fps:.1f}  dist:{dist_str}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )


# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
_lock = threading.Lock()
_jpeg_frame = b""

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

_PAUSE_FRAME: bytes = b""


def _make_pause_frame(w: int, h: int, image_path: str = "image.png") -> bytes:
    global _PAUSE_IMG_BASE

    if image_path:
        img_orig = cv2.imread(image_path)

        if img_orig is not None:
            logo_h = int(h * 0.55)
            logo_w = int(logo_h * img_orig.shape[1] / img_orig.shape[0])
            logo = cv2.resize(img_orig, (logo_w, logo_h))

            # Giu nguyen giao dien pause nen trang nhu code goc
            canvas = np.full((h, w, 3), 240, dtype=np.uint8)

            y0 = int(h * 0.04)
            x0 = (w - logo_w) // 2

            canvas[y0:y0 + logo_h, x0:x0 + logo_w] = logo
            _PAUSE_IMG_BASE = canvas.copy()

            _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()

    # Giu nguyen fallback nen trang nhu code goc
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    _PAUSE_IMG_BASE = img.copy()

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return buf.tobytes()


def _make_pause_frame_with_clock(w: int, h: int) -> np.ndarray:
    global _PAUSE_IMG_BASE

    if _PAUSE_IMG_BASE is not None:
        img = _PAUSE_IMG_BASE.copy()
    else:
        img = np.zeros((h, w, 3), dtype=np.uint8)

    clock = time.strftime("%H:%M:%S")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Satuday", "Sunday"]

    t = time.localtime()
    date = f"{day_names[t.tm_wday]},  {t.tm_mday:02d}/{t.tm_mon:02d}/{t.tm_year}"

    font = cv2.FONT_HERSHEY_SIMPLEX

    logo_zone_bottom = int(h * 0.62)

    clock_scale = w / 640 * 2.2
    date_scale = w / 640 * 0.75

    (cw, ch), _ = cv2.getTextSize(clock, font, clock_scale, 3)
    (dw, dh), _ = cv2.getTextSize(date, font, date_scale, 2)

    cx_clock = (w - cw) // 2
    cy_clock = logo_zone_bottom + ch + int(h * 0.04)

    cx_date = (w - dw) // 2
    cy_date = cy_clock + int(h * 0.07)

    cv2.putText(img, clock, (cx_clock, cy_clock), font, clock_scale, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(img, clock, (cx_clock, cy_clock), font, clock_scale, (30, 30, 30), 3, cv2.LINE_AA)

    cv2.putText(img, date, (cx_date, cy_date), font, date_scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, date, (cx_date, cy_date), font, date_scale, (100, 100, 200), 2, cv2.LINE_AA)

    return img


# ──────────────────────────────────────────────
# Sensor polling thread
# ──────────────────────────────────────────────
def sensor_loop(sensor, no_sensor: bool, hold_time: float, poll_interval: float):
    global _shared

    last_present_time = 0.0

    print(f"[SENSOR] Loop start. no_sensor={no_sensor} hold={hold_time}s poll={poll_interval}s")

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
            was_paused = _shared["paused"]
            _shared["paused"] = not effective_present
            _shared["sensor_present"] = present
            _shared["sensor_ok"] = (sensor is not None and sensor.ok) or no_sensor
            _shared["distance_mm"] = mm

        if was_paused and effective_present:
            print(f"[SENSOR] Co nguoi ({mm}mm) -> RESUME")
            with _lock:
                _shared["last_log"] = f"[SENSOR] Phat hien nguoi {mm}mm -> resume"

        elif not was_paused and not effective_present:
            print("[SENSOR] Khong co nguoi -> PAUSE")
            with _lock:
                _shared["last_log"] = "[SENSOR] Khong co nguoi -> pause"

        time.sleep(poll_interval)


# ──────────────────────────────────────────────
# Inference / camera loop
# ──────────────────────────────────────────────
def inference_loop(args):
    global _jpeg_frame, _shared, _PAUSE_FRAME

    _PAUSE_FRAME = _make_pause_frame(args.width, args.height, args.pause_image)

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

    yunet_times = []
    embed_times = []

    while True:
        with _lock:
            paused = _shared["paused"]
            dist_mm = _shared["distance_mm"]

        if paused:
            if USE_LOCAL_DISPLAY:
                pause_img = _make_pause_frame_with_clock(args.width, args.height)
                cv2.imshow("FaceLog", pause_img)
                cv2.waitKey(200)
            else:
                with _lock:
                    _jpeg_frame = _PAUSE_FRAME
                    _shared["fps"] = 0.0
                    _shared["state"] = "idle"
                    _shared["name"] = ""
                    _shared["sim"] = 0.0

                time.sleep(0.2)

            state = "idle"
            show_name = None
            show_bbox = None
            show_sim = 0.0
            last_verify = 0.0
            fps_t0 = time.time()
            fps_count = 0

            continue

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

            t0 = time.perf_counter()
            faces = detector.detect(frame)
            yunet_ms = (time.perf_counter() - t0) * 1000
            yunet_times.append(yunet_ms)

            face = detector.largest(faces)

            if face is not None:
                crop = detector.crop(frame, face)

                if crop is not None:
                    t1 = time.perf_counter()
                    emb = embedder.embed(crop)
                    embed_ms = (time.perf_counter() - t1) * 1000
                    embed_times.append(embed_ms)

                    if len(embed_times) % 20 == 0:
                        print(
                            f"[TIMING] YuNet avg={sum(yunet_times) / len(yunet_times):.1f}ms  "
                            f"MobileFaceNet avg={sum(embed_times) / len(embed_times):.1f}ms  "
                            f"(n={len(embed_times)})"
                        )

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
            with _lock:
                sensor_present = _shared["sensor_present"]

            ui = draw_local_ui(
                frame,
                show_bbox,
                show_name,
                show_sim,
                state,
                fps_display,
                dist_mm,
                sensor_present,
            )

            cv2.imshow("FaceLog", ui)

            if cv2.waitKey(1) == 27:
                break

        else:
            draw_overlay(frame, show_bbox, show_name, show_sim, state, fps_display, dist_mm)

            ok, buf = cv2.imencode(".jpg", frame, encode_param)

            if ok:
                with _lock:
                    if not _shared["paused"]:
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


# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


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


# ──────────────────────────────────────────────
def main():
    global DB_PATH

    p = argparse.ArgumentParser()

    p.add_argument("--camera", type=int, default=9)
    p.add_argument("--use-usb-camera", action="store_true")

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--db", default=str(DB_PATH))

    p.add_argument("--no-gpio", action="store_true")

    p.add_argument(
        "--no-sensor",
        action="store_true",
        help="Tat VL53L0X, chay lien tuc de test",
    )

    p.add_argument("--unlock-duration", type=float, default=5.0)
    p.add_argument("--cooldown", type=float, default=2.0)

    p.add_argument("--score-threshold", type=float, default=0.82)
    p.add_argument("--infer-every", type=int, default=6)

    p.add_argument("--jpeg-quality", type=int, default=70)

    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="0.0.0.0")

    p.add_argument(
        "--detect-distance",
        type=int,
        default=1000,
        help="Nguong khoang cach co nguoi, mm",
    )

    p.add_argument(
        "--sensor-hold",
        type=float,
        default=5.0,
        help="Giu trang thai co nguoi them N giay sau khi roi khoi nguong",
    )

    p.add_argument(
        "--sensor-poll",
        type=float,
        default=0.15,
        help="Chu ky polling sensor, giay",
    )

    p.add_argument(
        "--pause-image",
        default="image.png",
        help="Duong dan anh hien thi khi pause, jpg/png",
    )

    args = p.parse_args()

    DB_PATH = Path(args.db)

    if args.no_sensor:
        sensor = None
        print("[INFO] --no-sensor: bo qua VL53L0X, chay lien tuc")
    else:
        sensor = VL53L0XSensor(detect_distance_mm=args.detect_distance)

    ts = threading.Thread(
        target=sensor_loop,
        args=(sensor, args.no_sensor, args.sensor_hold, args.sensor_poll),
        daemon=True,
    )
    ts.start()

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
