#!/usr/bin/env python3
"""
verify_web.py
Verify khuon mat + stream MJPEG len web browser qua Flask.
Tich hop VL53L0X de tu dong pause/resume khi khong co nguoi.

Chay:
    python verify_web.py --no-gpio
    python verify_web.py --use-usb-camera --camera 0 --no-gpio
    python verify_web.py --no-gpio --no-sensor   # test khong co sensor

Mo browser: http://<PI_IP>:5000
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
import os
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

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mobilefacenet.tflite"
DB_PATH    = BASE_DIR / "face_database.json"
YUNET_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
IMG_SIZE   = 112
DEFAULT_THRESHOLD = 0.85
PIN_RELAY    = 23
PIN_LED_OK   = 24
PIN_LED_FAIL = 25

def has_display() -> bool:
    """Kiem tra co man hinh khong."""
    import os
    # Linux: kiem tra DISPLAY env var
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test")
            return True
        except:
            return False
    return False

USE_LOCAL_DISPLAY = has_display()

def ensure_yunet(path):
    if path.exists() and path.stat().st_size > 100_000: return
    print(f"[INFO] Tai YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Xong: {path.stat().st_size//1024} KB")


# ── GPIO ──────────────────────────────────────
class MockOutput:
    def __init__(self, pin, name):
        self.pin=pin; self.name=name
        print(f"[GPIO MOCK] {name} pin={pin}")
    def on(self):  print(f"[GPIO MOCK] {self.name} ON")
    def off(self): print(f"[GPIO MOCK] {self.name} OFF")
    def close(self): pass

class DoorGPIO:
    def __init__(self, enabled=True):
        if not enabled:
            self.relay=MockOutput(PIN_RELAY,"RELAY")
            self.led_ok=MockOutput(PIN_LED_OK,"LED_OK")
            self.led_fail=MockOutput(PIN_LED_FAIL,"LED_FAIL")
            return
        try:
            from gpiozero import OutputDevice, LED
            self.relay=OutputDevice(PIN_RELAY,active_high=False,initial_value=False)
            self.led_ok=LED(PIN_LED_OK); self.led_fail=LED(PIN_LED_FAIL)
            self.relay.off(); self.led_ok.off(); self.led_fail.off()
            print(f"[INFO] GPIO ready relay={PIN_RELAY}")
        except Exception as e:
            print(f"[WARN] GPIO loi: {e}")
            self.relay=MockOutput(PIN_RELAY,"RELAY")
            self.led_ok=MockOutput(PIN_LED_OK,"LED_OK")
            self.led_fail=MockOutput(PIN_LED_FAIL,"LED_FAIL")
    def unlock(self, seconds):
        def run():
            self.relay.on(); self.led_ok.on()
            time.sleep(seconds)
            self.relay.off(); self.led_ok.off()
        threading.Thread(target=run,daemon=True).start()
    def deny(self):
        def run():
            for _ in range(3):
                self.led_fail.on(); time.sleep(0.15)
                self.led_fail.off(); time.sleep(0.15)
        threading.Thread(target=run,daemon=True).start()
    def close(self):
        for x in (self.relay,self.led_ok,self.led_fail):
            try: x.off(); x.close()
            except: pass


# ── VL53L0X sensor ────────────────────────────
class VL53L0XSensor:
    def __init__(self, detect_distance_mm: int = 500, i2c_bus: int = 1):
        self.detect_distance_mm = detect_distance_mm
        self._sensor = None
        self._ok = False
        try:
            import board, busio
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
        """Tra ve (co_nguoi, distance_mm).
        FIX: sensor loi -> False (pause camera), khong phai True.
        """
        mm = self.read_mm()
        if mm < 0:
            # ✅ FIX 1: sensor lỗi/không kết nối → KHÔNG coi là có người
            # Camera sẽ tắt thay vì chạy liên tục
            return False, -1
        return mm <= self.detect_distance_mm, mm


# ── ML models ─────────────────────────────────
class MobileFaceNet:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Khong tim thay {MODEL_PATH}")
        self.interp = Interpreter(model_path=str(MODEL_PATH))
        self.interp.allocate_tensors()
        self.inp  = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        print("[INFO] MobileFaceNet ready")
    def embed(self, face_bgr):
        face = cv2.resize(face_bgr,(IMG_SIZE,IMG_SIZE))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32)-127.5)/128.0
        self.interp.set_tensor(self.inp["index"],face[np.newaxis])
        self.interp.invoke()
        emb = self.interp.get_tensor(self.outp["index"])[0]
        return (emb/(np.linalg.norm(emb)+1e-8)).astype(np.float32)


class YuNetDetector:
    def __init__(self, score_threshold=0.82):
        ensure_yunet(YUNET_PATH)
        self.det = cv2.FaceDetectorYN_create(
            str(YUNET_PATH),"",(320,320),score_threshold,0.3,5000)
        print("[INFO] YuNet ready")
    def detect(self, frame):
        h,w = frame.shape[:2]
        self.det.setInputSize((w,h))
        _,faces = self.det.detect(frame)
        if faces is None: return []
        return [{"box":tuple(int(x) for x in f[:4]),
                 "landmarks":f[4:14].reshape(5,2).astype(np.float32),
                 "score":float(f[-1])} for f in faces]
    @staticmethod
    def largest(faces):
        return max(faces,key=lambda f:f["box"][2]*f["box"][3]) if faces else None
    @staticmethod
    def align_crop(frame,landmarks,size=112):
        dst=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                      [41.5493,92.3655],[70.7299,92.2041]],dtype=np.float32)
        try:
            M,_=cv2.estimateAffinePartial2D(landmarks.astype(np.float32),dst,method=cv2.LMEDS)
            if M is None: return None
            return cv2.warpAffine(frame,M,(size,size),borderValue=0.0)
        except cv2.error: return None
    @staticmethod
    def crop_fallback(frame,face,pad_ratio=0.18):
        x,y,w,h=face["box"]; pad=int(pad_ratio*max(w,h))
        x1=max(0,x-pad);y1=max(0,y-pad)
        x2=min(frame.shape[1],x+w+pad);y2=min(frame.shape[0],y+h+pad)
        crop=frame[y1:y2,x1:x2]
        return cv2.resize(crop,(112,112)) if crop.size else None
    def crop(self,frame,face):
        c=self.align_crop(frame,face["landmarks"])
        return c if c is not None else self.crop_fallback(frame,face)


def load_database():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay: {DB_PATH}")
    with open(DB_PATH,"r",encoding="utf-8") as f:
        raw=json.load(f)
    db={}
    for name,value in raw.items():
        if isinstance(value,list):
            db[name]=[np.array(e,dtype=np.float32) for e in value]
        elif isinstance(value,dict) and "embeddings" in value:
            db[name]=[np.array(e,dtype=np.float32) for e in value["embeddings"]]
    total=sum(len(v) for v in db.values())
    print(f"[INFO] DB: {len(db)} nguoi, {total} embeddings")
    return db

def recognize(embedding, db, threshold):
    best_name,best_sim=None,-1.0
    for name,embs in db.items():
        if not embs: continue
        sim=max(float(np.dot(embedding,e/(np.linalg.norm(e)+1e-8))) for e in embs)
        if sim>best_sim: best_sim,best_name=sim,name
    return (best_name,best_sim) if best_sim>=threshold else (None,best_sim)

def open_camera(args):
    if not args.use_usb_camera:
        try:
            from picamera2 import Picamera2
            class PiCamWrap:
                def __init__(self):
                    self.picam2=Picamera2()
                    cfg=self.picam2.create_preview_configuration(
                        main={"size":(args.width,args.height),"format":"RGB888"},
                        controls={"FrameRate":args.fps})
                    self.picam2.configure(cfg); self.picam2.start(); time.sleep(0.5)
                    print(f"[INFO] Pi Camera {args.width}x{args.height}@{args.fps}")
                def isOpened(self): return True
                def read(self):
                    rgb=self.picam2.capture_array()
                    return (True, rgb) if rgb is not None else (False, None)

                def release(self): self.picam2.stop()
            return PiCamWrap()
        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")
    cap=cv2.VideoCapture(args.camera,cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    cap.set(cv2.CAP_PROP_FPS,args.fps)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"[INFO] OpenCV camera id={args.camera}")
    return cap

def draw_overlay(frame, bbox, name, sim, state, fps=0.0, dist_mm=-1):
    color={"idle":(180,180,180),"detecting":(0,200,255),
           "ok":(0,220,120),"deny":(40,40,220)}.get(state,(180,180,180))
    h,w=frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,42),(20,20,20),-1)
    if state=="ok":          text=f"OK  {name}  sim={sim:.3f}"
    elif state=="deny":      text=f"TU CHOI  sim={sim:.3f}"
    elif state=="detecting": text="DANG NHAN DIEN..."
    else:                    text="NHIN VAO CAMERA"
    cv2.putText(frame,text,(12,29),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)
    if bbox:
        x,y,bw,bh=bbox
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),color,2)
    dist_str = f"{dist_mm}mm" if dist_mm>0 else "—"
    cv2.putText(frame,f"FPS:{fps:.1f}  dist:{dist_str}",
                (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(160,160,160),1)


# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
_lock       = threading.Lock()
_jpeg_frame = b""
_shared = {
    "paused":        True,   # mặc định pause, chờ sensor xác nhận
    "state":         "idle",
    "name":          "",
    "sim":           0.0,
    "fps":           0.0,
    "distance_mm":   -1,
    "sensor_present":False,
    "sensor_ok":     False,
    "last_log":      None,
}

_PAUSE_FRAME: bytes = b""

def _make_pause_frame(w: int, h: int, image_path: str = "image.png") -> bytes:
    # Thử load ảnh từ file nếu có
    if image_path:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, (w, h))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes()
        else:
            print(f"[WARN] Khong load duoc anh: {image_path}, dung nen den")

    # Fallback: nền đen như cũ
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img,"Waiting for person...",(w//2-160,h//2-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.75,(60,60,60),2)
    cv2.putText(img,"(VL53L0X paused)",(w//2-110,h//2+24),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,40,40),1)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return buf.tobytes()


# ──────────────────────────────────────────────
# Sensor polling thread
# ──────────────────────────────────────────────
def sensor_loop(sensor,           # VL53L0XSensor hoặc None khi --no-sensor
                no_sensor: bool,
                hold_time: float,
                poll_interval: float):
    """
    CHỦ DUY NHẤT của _shared["paused"].
    inference_loop chỉ được ĐỌC paused, không được ghi.
    """
    global _shared
    last_present_time = 0.0

    print(f"[SENSOR] Loop start. no_sensor={no_sensor} hold={hold_time}s poll={poll_interval}s")
    while True:
        now = time.time()

        if no_sensor or sensor is None:
            # ✅ FIX 2: --no-sensor → luôn coi là có người (dùng để test)
            present, mm = True, -1
        else:
            present, mm = sensor.person_present()

        if present:
            last_present_time = now

        effective_present = (now - last_present_time) < hold_time

        with _lock:
            was_paused = _shared["paused"]
            # ✅ sensor_loop là nơi DUY NHẤT set paused
            _shared["paused"]         = not effective_present
            _shared["sensor_present"] = present
            _shared["sensor_ok"]      = (sensor is not None and sensor.ok) or no_sensor
            _shared["distance_mm"]    = mm

        if was_paused and effective_present:
            print(f"[SENSOR] Co nguoi ({mm}mm) -> RESUME")
            with _lock:
                _shared["last_log"] = f"[SENSOR] Phat hien nguoi {mm}mm -> resume"
        elif not was_paused and not effective_present:
            print(f"[SENSOR] Khong co nguoi -> PAUSE")
            with _lock:
                _shared["last_log"] = "[SENSOR] Khong co nguoi -> pause"

        time.sleep(poll_interval)


# ──────────────────────────────────────────────
# Inference / camera loop
# ──────────────────────────────────────────────
def inference_loop(args):
    global _jpeg_frame, _shared, _PAUSE_FRAME

    _PAUSE_FRAME = _make_pause_frame(args.width, args.height, args.pause_image)

    gpio     = DoorGPIO(enabled=not args.no_gpio)
    detector = YuNetDetector(score_threshold=args.score_threshold)
    embedder = MobileFaceNet()
    db       = load_database()
    cap      = open_camera(args)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera"); return

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    frame_count  = 0
    infer_every  = args.infer_every
    last_verify  = 0.0
    state        = "idle"
    show_until   = 0.0
    show_name    = None
    show_sim     = 0.0
    show_bbox    = None

    fps_t0      = time.time()
    fps_count   = 0
    fps_display = 0.0

    print(f"[INFO] Inference loop start. infer_every={infer_every}")
    if USE_LOCAL_DISPLAY:
        cv2.namedWindow("FaceLog", cv2.WINDOW_NORMAL)
        cv2.moveWindow("FaceLog", 0, 0)
        cv2.setWindowProperty("FaceLog", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("[INFO] Local display fullscreen")
    while True:
        # ── Đọc trạng thái pause từ sensor_loop ──
        with _lock:
            paused  = _shared["paused"]
            dist_mm = _shared["distance_mm"]

        # M?I
        if paused:
            if USE_LOCAL_DISPLAY:
                pause_img = cv2.imdecode(
                    np.frombuffer(_PAUSE_FRAME, np.uint8), cv2.IMREAD_COLOR)
                if pause_img is not None:
                    cv2.imshow("FaceLog", pause_img)
                    cv2.waitKey(200)
            else:
                with _lock:
                    _jpeg_frame      = _PAUSE_FRAME
                    _shared["fps"]   = 0.0
                    _shared["state"] = "idle"
                    _shared["name"]  = ""
                    _shared["sim"]   = 0.0
                time.sleep(0.2)
            # Reset chung 
            state = "idle"; show_name = None; show_bbox = None
            show_sim = 0.0; last_verify = 0.0
            fps_t0 = time.time(); fps_count = 0
            continue   

        # ── Active: đọc frame ───────────────────
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02); continue

        frame = cv2.flip(frame, 1)
        frame_count += 1
        now = time.time()

        fps_count += 1
        if fps_count >= 15:
            fps_display = fps_count / (now - fps_t0)
            fps_t0 = now; fps_count = 0

        if now > show_until and state in ("ok","deny"):
            state="idle"; show_name=None; show_bbox=None; show_sim=0.0

        new_log = None

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

        draw_overlay(frame, show_bbox, show_name, show_sim, state, fps_display, dist_mm)

        if USE_LOCAL_DISPLAY:
            cv2.imshow("FaceLog", frame)
            if cv2.waitKey(1) == 27:  # ESC thoat
                break
        else:
            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if ok:
                with _lock:
                    if not _shared["paused"]:
                        _jpeg_frame = buf.tobytes()
                        _shared.update({
                            "state":    state,
                            "name":     show_name or "",
                            "sim":      round(show_sim, 4),
                            "fps":      round(fps_display, 1),
                            "last_log": new_log,
                        })


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
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with _lock:
        s = dict(_shared)
    return jsonify(s)


# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",           type=int,   default=9)
    p.add_argument("--use-usb-camera",   action="store_true")
    p.add_argument("--width",            type=int,   default=640)
    p.add_argument("--height",           type=int,   default=480)
    p.add_argument("--fps",              type=int,   default=15)
    p.add_argument("--threshold",        type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--db",               default=str(DB_PATH))
    p.add_argument("--no-gpio",          action="store_true")
    p.add_argument("--no-sensor",        action="store_true",
                   help="Tat VL53L0X, chay lien tuc (de test)")
    p.add_argument("--unlock-duration",  type=float, default=3.0)
    p.add_argument("--cooldown",         type=float, default=2.0)
    p.add_argument("--score-threshold",  type=float, default=0.82)
    p.add_argument("--infer-every",      type=int,   default=6)
    p.add_argument("--jpeg-quality",     type=int,   default=70)
    p.add_argument("--port",             type=int,   default=5000)
    p.add_argument("--host",             default="0.0.0.0")
    p.add_argument("--detect-distance",  type=int,   default=500,
                   help="Nguong khoang cach co nguoi (mm), mac dinh 500mm = 50cm")
    p.add_argument("--sensor-hold",      type=float, default=5.0,
                   help="Giu trang thai co nguoi them N giay sau khi roi khoi nguong")
    p.add_argument("--sensor-poll",      type=float, default=0.15,
                   help="Chu ky polling sensor (giay)")
    p.add_argument("--pause-image", default="image.png",
               help="Duong dan anh hien thi khi pause (jpg/png)")
    args = p.parse_args()

    # ✅ FIX 2: Khởi tạo sensor sạch, không dùng __new__ hack
    if args.no_sensor:
        sensor = None
        print("[INFO] --no-sensor: bo qua VL53L0X, chay lien tuc")
    else:
        sensor = VL53L0XSensor(detect_distance_mm=args.detect_distance)

    ts = threading.Thread(
        target=sensor_loop,
        args=(sensor, args.no_sensor, args.sensor_hold, args.sensor_poll),
        daemon=True)
    ts.start()

    if USE_LOCAL_DISPLAY:
        print("[INFO] Co man hinh -> Local display mode")
        inference_loop(args)          # chay main thread (b?t bu?c v?i imshow)
    else:
        print(f"[WEB] Mo browser: http://<PI_IP>:{args.port}")
        ti = threading.Thread(target=inference_loop, args=(args,), daemon=True)
        ti.start()
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
