#!/usr/bin/env python3
"""
verify_web.py
Verify khuon mat + stream MJPEG len web browser qua Flask.
Khong can man hinh, khong can VNC.

Chay:
    python verify_web.py --no-gpio
    python verify_web.py --use-usb-camera --camera 0 --no-gpio

Roi mo browser: http://<PI_IP>:5000
"""

import argparse
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

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
MODEL_PATH = BASE_DIR / "mobilefacenet.tflite"
DB_PATH = BASE_DIR / "face_database.json"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
IMG_SIZE = 112
DEFAULT_THRESHOLD = 0.9
PIN_RELAY = 23
PIN_LED_OK = 24
PIN_LED_FAIL = 25

# ──────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Verify — Pi</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: monospace; display: flex;
         flex-direction: column; align-items: center; min-height: 100vh; padding: 16px; }
  h1 { font-size: 1rem; color: #aaa; margin-bottom: 12px; letter-spacing: 2px; }
  #stream { width: 100%; max-width: 640px; border: 1px solid #333; border-radius: 6px; }
  #status { margin-top: 12px; width: 100%; max-width: 640px; }
  .row { display: flex; justify-content: space-between; padding: 6px 10px;
         border-bottom: 1px solid #222; font-size: 0.82rem; }
  .label { color: #888; }
  .val { color: #eee; }
  .val.ok   { color: #3ddc84; font-weight: bold; }
  .val.deny { color: #ff5555; font-weight: bold; }
  .val.detecting { color: #ffb86c; }
  #log { margin-top: 12px; width: 100%; max-width: 640px; background: #1a1a1a;
         border: 1px solid #333; border-radius: 6px; padding: 8px;
         font-size: 0.78rem; height: 120px; overflow-y: auto; color: #aaa; }
</style>
</head>
<body>
<h1>FACE VERIFY — RASPBERRY PI</h1>
<img id="stream" src="/video_feed" alt="stream">
<div id="status">
  <div class="row"><span class="label">Trạng thái</span><span class="val" id="s-state">—</span></div>
  <div class="row"><span class="label">Tên</span><span class="val" id="s-name">—</span></div>
  <div class="row"><span class="label">Similarity</span><span class="val" id="s-sim">—</span></div>
  <div class="row"><span class="label">FPS stream</span><span class="val" id="s-fps">—</span></div>
</div>
<div id="log"></div>
<script>
  function poll() {
    fetch('/status').then(r => r.json()).then(d => {
      const el = document.getElementById('s-state');
      el.textContent = d.state;
      el.className = 'val ' + d.state;
      document.getElementById('s-name').textContent = d.name || '—';
      document.getElementById('s-sim').textContent = d.sim > 0 ? d.sim.toFixed(3) : '—';
      document.getElementById('s-fps').textContent = d.fps.toFixed(1);
      if (d.last_log) {
        const log = document.getElementById('log');
        const line = document.createElement('div');
        line.textContent = d.last_log;
        log.appendChild(line);
        log.scrollTop = log.scrollHeight;
      }
    }).catch(() => {});
    setTimeout(poll, 400);
  }
  poll();
</script>
</body>
</html>"""
# ──────────────────────────────────────────────


def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return
    print(f"[INFO] Dang tai YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Da tai: {path.stat().st_size // 1024} KB")


class MockOutput:
    def __init__(self, pin, name):
        self.pin = pin; self.name = name
        print(f"[GPIO MOCK] {name} pin={pin}")
    def on(self):  print(f"[GPIO MOCK] {self.name} ON")
    def off(self): print(f"[GPIO MOCK] {self.name} OFF")
    def close(self): pass


class DoorGPIO:
    def __init__(self, enabled=True):
        if not enabled:
            self.relay   = MockOutput(PIN_RELAY,   "RELAY")
            self.led_ok  = MockOutput(PIN_LED_OK,  "LED_OK")
            self.led_fail= MockOutput(PIN_LED_FAIL,"LED_FAIL")
            return
        try:
            from gpiozero import OutputDevice, LED
            self.relay    = OutputDevice(PIN_RELAY, active_high=False, initial_value=False)
            self.led_ok   = LED(PIN_LED_OK)
            self.led_fail = LED(PIN_LED_FAIL)
            self.relay.off(); self.led_ok.off(); self.led_fail.off()
            print(f"[INFO] GPIO ready relay={PIN_RELAY}")
        except Exception as e:
            print(f"[WARN] GPIO loi: {e}. Dung mock.")
            self.relay   = MockOutput(PIN_RELAY,   "RELAY")
            self.led_ok  = MockOutput(PIN_LED_OK,  "LED_OK")
            self.led_fail= MockOutput(PIN_LED_FAIL,"LED_FAIL")

    def unlock(self, seconds):
        def run():
            self.relay.on(); self.led_ok.on()
            time.sleep(seconds)
            self.relay.off(); self.led_ok.off()
        threading.Thread(target=run, daemon=True).start()

    def deny(self):
        def run():
            for _ in range(3):
                self.led_fail.on();  time.sleep(0.15)
                self.led_fail.off(); time.sleep(0.15)
        threading.Thread(target=run, daemon=True).start()

    def close(self):
        for x in (self.relay, self.led_ok, self.led_fail):
            try: x.off(); x.close()
            except Exception: pass


class MobileFaceNet:
    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay {model_path}")
        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp  = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        print(f"[INFO] MobileFaceNet ready")

    def embed(self, face_bgr):
        face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 128.0
        self.interp.set_tensor(self.inp["index"], face[np.newaxis])
        self.interp.invoke()
        emb = self.interp.get_tensor(self.outp["index"])[0]
        return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)


class YuNetDetector:
    def __init__(self, model_path=YUNET_PATH, score_threshold=0.82):
        ensure_yunet(model_path)
        self.detector = cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
        print(f"[INFO] YuNet ready")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None: return []
        out = []
        for f in faces:
            out.append({"box": tuple(int(x) for x in f[:4]),
                        "landmarks": f[4:14].reshape(5,2).astype(np.float32),
                        "score": float(f[-1])})
        return out

    @staticmethod
    def largest(faces):
        return max(faces, key=lambda f: f["box"][2]*f["box"][3]) if faces else None

    @staticmethod
    def align_crop(frame, landmarks, size=112):
        dst = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                        [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)
        try:
            M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
            if M is None: return None
            return cv2.warpAffine(frame, M, (size, size), borderValue=0.0)
        except cv2.error: return None

    @staticmethod
    def crop_fallback(frame, face, pad_ratio=0.18):
        x, y, w, h = face["box"]
        pad = int(pad_ratio * max(w, h))
        x1=max(0,x-pad); y1=max(0,y-pad)
        x2=min(frame.shape[1],x+w+pad); y2=min(frame.shape[0],y+h+pad)
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (112,112)) if crop.size else None

    def crop(self, frame, face):
        c = self.align_crop(frame, face["landmarks"])
        return c if c is not None else self.crop_fallback(frame, face)


def load_database(path=DB_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    db = {}
    for name, value in raw.items():
        if isinstance(value, list):
            db[name] = [np.array(e, dtype=np.float32) for e in value]
        elif isinstance(value, dict) and "embeddings" in value:
            db[name] = [np.array(e, dtype=np.float32) for e in value["embeddings"]]
    total = sum(len(v) for v in db.values())
    print(f"[INFO] DB: {len(db)} nguoi, {total} embeddings")
    return db


def recognize(embedding, db, threshold):
    best_name, best_sim = None, -1.0
    for name, embs in db.items():
        if not embs: continue
        sim = max(float(np.dot(embedding, e/(np.linalg.norm(e)+1e-8))) for e in embs)
        if sim > best_sim:
            best_sim, best_name = sim, name
    return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)


def draw_overlay(frame, bbox, name, sim, state, fps=0.0):
    color = {"idle":(180,180,180),"detecting":(0,200,255),
             "ok":(0,220,120),"deny":(40,40,220)}.get(state,(180,180,180))
    h, w = frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,42),(20,20,20),-1)
    if state=="ok":        text = f"OK  {name}  sim={sim:.3f}"
    elif state=="deny":    text = f"TU CHOI  sim={sim:.3f}"
    elif state=="detecting": text = "DANG NHAN DIEN..."
    else:                  text = "NHIN VAO CAMERA"
    cv2.putText(frame,text,(12,29),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)
    if bbox:
        x,y,bw,bh = bbox
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),color,2)
    cv2.putText(frame,f"FPS:{fps:.1f}",(w-90,h-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1)


def open_camera(camera, width, height, fps, use_usb_camera):
    if not use_usb_camera:
        try:
            from picamera2 import Picamera2
            class PiCamWrap:
                def __init__(self):
                    self.picam2 = Picamera2()
                    cfg = self.picam2.create_preview_configuration(
                        main={"size":(width,height),"format":"RGB888"},
                        controls={"FrameRate":fps})
                    self.picam2.configure(cfg); self.picam2.start(); time.sleep(0.5)
                    print(f"[INFO] Pi Camera {width}x{height}@{fps}")
                def isOpened(self): return True
                def read(self):
                    rgb = self.picam2.capture_array()
                    if rgb is None: return False, None
                    return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                def release(self): self.picam2.stop()
            return PiCamWrap()
        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}, fallback OpenCV")
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"[INFO] OpenCV camera id={camera}")
    return cap


# ──────────────────────────────────────────────
# Global shared state (giữa inference thread và Flask)
# ──────────────────────────────────────────────
_lock        = threading.Lock()
_jpeg_frame  = b""          # frame đã encode JPEG
_state_info  = {
    "state": "idle",
    "name": None,
    "sim": 0.0,
    "fps": 0.0,
    "last_log": None,
}
_log_sent = None            # tránh gửi log trùng


def inference_loop(args):
    global _jpeg_frame, _state_info, _log_sent

    gpio     = DoorGPIO(enabled=not args.no_gpio)
    detector = YuNetDetector(score_threshold=args.score_threshold)
    embedder = MobileFaceNet(MODEL_PATH)
    db       = load_database(Path(args.db))
    cap      = open_camera(args.camera, args.width, args.height, args.fps, args.use_usb_camera)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera"); return

    frame_count  = 0
    infer_every  = args.infer_every
    last_verify  = 0.0
    state        = "idle"
    show_until   = 0.0
    show_name    = None
    show_sim     = 0.0
    show_bbox    = None

    fps_t0       = time.time()
    fps_count    = 0
    fps_display  = 0.0

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    print(f"[INFO] Inference loop start. infer_every={infer_every} jpeg_quality={args.jpeg_quality}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02); continue

            frame = cv2.flip(frame, 1)
            frame_count += 1
            now = time.time()

            # FPS
            fps_count += 1
            if fps_count >= 15:
                fps_display = fps_count / (now - fps_t0)
                fps_t0 = now; fps_count = 0

            # Reset state sau timeout
            if now > show_until and state in ("ok","deny"):
                state="idle"; show_name=None; show_bbox=None; show_sim=0.0

            new_log = None

            # Inference mỗi N frame
            if frame_count % infer_every == 0 and now - last_verify >= args.cooldown:
                state = "detecting"
                face = detector.largest(detector.detect(frame))
                if face is not None:
                    crop = detector.crop(frame, face)
                    if crop is not None:
                        emb = embedder.embed(crop)
                        name, sim = recognize(emb, db, args.threshold)
                        show_bbox = face["box"]; show_sim = sim
                        if name:
                            msg = f"[OK] {time.strftime('%H:%M:%S')} {name} sim={sim:.3f}"
                            print(msg); new_log = msg
                            state="ok"; show_name=name
                            show_until = now + args.unlock_duration + 0.5
                            gpio.unlock(args.unlock_duration)
                        else:
                            msg = f"[DENY] {time.strftime('%H:%M:%S')} sim={sim:.3f}"
                            print(msg); new_log = msg
                            state="deny"; show_name=None
                            show_until = now + 1.5
                            gpio.deny()
                        last_verify = now
                else:
                    state="idle"; show_bbox=None

            # Vẽ overlay rồi encode JPEG
            draw_overlay(frame, show_bbox, show_name, show_sim, state, fps_display)
            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if ok:
                with _lock:
                    _jpeg_frame = buf.tobytes()
                    _state_info = {
                        "state": state,
                        "name":  show_name or "",
                        "sim":   round(show_sim, 4),
                        "fps":   round(fps_display, 1),
                        "last_log": new_log,
                    }

    except Exception as e:
        print(f"[ERROR] inference_loop: {e}")
    finally:
        cap.release()
        gpio.close()
        print("[INFO] Inference loop da dung")


# ──────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _lock:
                frame = _jpeg_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)   # ~30 fps max gửi
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with _lock:
        info = dict(_state_info)
    from flask import jsonify
    return jsonify(info)


# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",         type=int,   default=9)
    p.add_argument("--use-usb-camera", action="store_true")
    p.add_argument("--width",          type=int,   default=640)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--fps",            type=int,   default=15)
    p.add_argument("--threshold",      type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--db",             default=str(DB_PATH))
    p.add_argument("--no-gpio",        action="store_true")
    p.add_argument("--unlock-duration",type=float, default=3.0)
    p.add_argument("--cooldown",       type=float, default=2.0)
    p.add_argument("--score-threshold",type=float, default=0.82)
    p.add_argument("--infer-every",    type=int,   default=6,
                   help="Chi inference moi N frame (tang len neu Pi qua tai)")
    p.add_argument("--jpeg-quality",   type=int,   default=70,
                   help="JPEG quality 1-95, thap hon = nhe hon = it lag hon")
    p.add_argument("--port",           type=int,   default=5000)
    p.add_argument("--host",           default="0.0.0.0")
    args = p.parse_args()

    # Chạy inference trong thread riêng
    t = threading.Thread(target=inference_loop, args=(args,), daemon=True)
    t.start()

    print(f"\n[WEB] Mo browser: http://<PI_IP>:{args.port}\n")
    # use_reloader=False quan trọng — tránh Flask spawn thread thứ 2
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
