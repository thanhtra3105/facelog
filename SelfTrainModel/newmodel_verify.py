#!/usr/bin/env python3
"""
verify_yunet_mobilefacenet.py
Nhan dien khuon mat de mo cua bang YuNet + MobileFaceNet TFLite.

- Khong dung mediapipe.
- Mac dinh dung Pi Camera qua Picamera2.
- Co the dung USB camera bang --use-usb-camera --camera 9.
- Doc database cu face_database.json tao boi register_yunet_mobilefacenet.py.

Chay test khong GPIO:
    python verify_yunet_mobilefacenet.py --no-gpio

Chay USB cam test:
    python verify_yunet_mobilefacenet.py --use-usb-camera --camera 9 --no-gpio

    
======================================================================================
Chạy code của Trí ở đây 
Chay model moi
    python newmodel_verify.py --use-usb-camera --camera 0 --no-gpio
"""

import argparse
import json
import os
import sys
import time
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

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
MODEL_PATH = BASE_DIR / "mobilefacenet_integer_quant.tflite"
DB_PATH = BASE_DIR / "face_database.json"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
IMG_SIZE = 112
DEFAULT_THRESHOLD = 0.9
PIN_RELAY = 17
PIN_LED_OK = 27
PIN_LED_FAIL = 22


def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return
    print(f"[INFO] Dang tai YuNet model -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Da tai YuNet: {path.stat().st_size // 1024} KB")


class MockOutput:
    def __init__(self, pin, name, active_low=False):
        self.pin = pin
        self.name = name
        self.active_low = active_low
        print(f"[GPIO MOCK] {name} pin={pin} active_low={active_low}")
    def on(self):
        print(f"[GPIO MOCK] {self.name} ON")
    def off(self):
        print(f"[GPIO MOCK] {self.name} OFF")
    def close(self):
        pass


class DoorGPIO:
    def __init__(self, enabled=True):
        if not enabled:
            self.relay = MockOutput(PIN_RELAY, "RELAY", active_low=True)
            self.led_ok = MockOutput(PIN_LED_OK, "LED_OK")
            self.led_fail = MockOutput(PIN_LED_FAIL, "LED_FAIL")
            print("[INFO] --no-gpio: chi in log")
            return
        try:
            from gpiozero import OutputDevice, LED
            self.relay = OutputDevice(PIN_RELAY, active_high=False, initial_value=False)
            self.led_ok = LED(PIN_LED_OK)
            self.led_fail = LED(PIN_LED_FAIL)
            self.relay.off(); self.led_ok.off(); self.led_fail.off()
            print(f"[INFO] GPIO ready relay={PIN_RELAY}, led_ok={PIN_LED_OK}, led_fail={PIN_LED_FAIL}")
        except Exception as e:
            print(f"[WARN] GPIO loi: {e}. Chuyen sang mock.")
            self.relay = MockOutput(PIN_RELAY, "RELAY", active_low=True)
            self.led_ok = MockOutput(PIN_LED_OK, "LED_OK")
            self.led_fail = MockOutput(PIN_LED_FAIL, "LED_FAIL")
    def unlock(self, seconds):
        def run():
            print(f"[DOOR] MO KHOA {seconds:.1f}s")
            self.relay.on(); self.led_ok.on()
            time.sleep(seconds)
            self.relay.off(); self.led_ok.off()
            print("[DOOR] DA KHOA LAI")
        threading.Thread(target=run, daemon=True).start()
    def deny(self):
        def run():
            for _ in range(3):
                self.led_fail.on(); time.sleep(0.15)
                self.led_fail.off(); time.sleep(0.15)
        threading.Thread(target=run, daemon=True).start()
    def close(self):
        for x in (self.relay, self.led_ok, self.led_fail):
            try:
                x.off(); x.close()
            except Exception:
                pass


class MobileFaceNet:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay {model_path}")
        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        print(f"[INFO] MobileFaceNet ready: {model_path}")
    def embed(self, face_bgr):
        face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # --------------------------------------------------------
        # SỬA TẠI ĐÂY: ĐỒNG BỘ CHUẨN HÓA IMAGENET GIỐNG LÚC TRAIN
        # --------------------------------------------------------
        face = face.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face = (face - mean) / std
        # --------------------------------------------------------
        
        self.interp.set_tensor(self.inp["index"], face[np.newaxis])
        self.interp.invoke()
        emb = self.interp.get_tensor(self.outp["index"])[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)


class YuNetDetector:
    def __init__(self, model_path: Path = YUNET_PATH, score_threshold: float = 0.82):
        ensure_yunet(model_path)
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError("OpenCV khong co FaceDetectorYN_create")
        self.detector = cv2.FaceDetectorYN_create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
        print(f"[INFO] YuNet ready: {model_path}")
    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, bw, bh = f[:4]
            out.append({"box": (int(x), int(y), int(bw), int(bh)), "landmarks": f[4:14].reshape(5, 2).astype(np.float32), "score": float(f[-1])})
        return out
    @staticmethod
    def largest(faces):
        return max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None
    @staticmethod
    def align_crop(frame, landmarks, size=112):
        dst = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)
        if size != 112:
            dst *= size / 112.0
        try:
            M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
            if M is None:
                return None
            return cv2.warpAffine(frame, M, (size, size), borderValue=0.0)
        except cv2.error:
            return None
    @staticmethod
    def crop_fallback(frame, face, pad_ratio=0.18):
        x, y, w, h = face["box"]
        pad = int(pad_ratio * max(w, h))
        x1=max(0,x-pad); y1=max(0,y-pad); x2=min(frame.shape[1],x+w+pad); y2=min(frame.shape[0],y+h+pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112,112))
    def crop(self, frame, face):
        crop = self.align_crop(frame, face["landmarks"], 112)
        return crop if crop is not None else self.crop_fallback(frame, face)


def open_camera(camera, width, height, fps, use_usb_camera):
    if not use_usb_camera:
        try:
            from picamera2 import Picamera2
            class PiCamWrap:
                def __init__(self):
                    self.picam2 = Picamera2()
                    cfg = self.picam2.create_preview_configuration(main={"size": (width,height), "format":"RGB888"}, controls={"FrameRate": fps})
                    self.picam2.configure(cfg); self.picam2.start(); time.sleep(0.5)
                    print(f"[INFO] Dung Pi Camera: {width}x{height}@{fps}")
                def isOpened(self): return True
                def read(self):
                    rgb = self.picam2.capture_array()
                    if rgb is None: return False, None
                    return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                def release(self): self.picam2.stop()
            return PiCamWrap()
        except Exception as e:
            print(f"[WARN] Khong mo duoc Pi Camera: {e}")
            print("[WARN] Fallback sang OpenCV camera")
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); cap.set(cv2.CAP_PROP_FPS,fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"[INFO] Dung OpenCV camera id={camera}: {width}x{height}@{fps}")
    return cap


def load_database(path: Path = DB_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay database: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    db = {}
    for name, value in raw.items():
        if isinstance(value, list):
            db[name] = [np.array(e, dtype=np.float32) for e in value]
        elif isinstance(value, dict) and "embeddings" in value:
            db[name] = [np.array(e, dtype=np.float32) for e in value["embeddings"]]
    total = sum(len(v) for v in db.values())
    print(f"[INFO] Database: {len(db)} nguoi, {total} embeddings")
    return db


def recognize(embedding, db, threshold):
    best_name = None; best_sim = -1.0
    for name, embs in db.items():
        if not embs: continue
        sim = max(float(np.dot(embedding, e / (np.linalg.norm(e)+1e-8))) for e in embs)
        if sim > best_sim:
            best_sim = sim; best_name = name
    return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)


def draw_overlay(frame, bbox, name, sim, state):
    color = {"idle":(180,180,180), "detecting":(0,200,255), "ok":(0,220,120), "deny":(40,40,220)}.get(state,(180,180,180))
    h,w = frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,42),(20,20,20),-1)
    if state == "ok": text = f"OK {name} sim={sim:.3f} - MO KHOA"
    elif state == "deny": text = f"TU CHOI sim={sim:.3f}"
    elif state == "detecting": text = "DANG NHAN DIEN..."
    else: text = "NHIN VAO CAMERA"
    cv2.putText(frame,text,(12,29),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2,cv2.LINE_AA)
    if bbox is not None:
        x,y,bw,bh = bbox
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),color,2)
    cv2.putText(frame,"q/ESC de thoat",(12,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.55,(220,220,220),1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=9)
    p.add_argument("--use-usb-camera", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--no-gpio", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--unlock-duration", type=float, default=3.0)
    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--score-threshold", type=float, default=0.82)
    args = p.parse_args()

    gpio = DoorGPIO(enabled=not args.no_gpio)
    detector = YuNetDetector(score_threshold=args.score_threshold)
    embedder = MobileFaceNet(MODEL_PATH)
    db = load_database(Path(args.db))
    cap = open_camera(args.camera, args.width, args.height, args.fps, args.use_usb_camera)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc camera")

    if not args.no_preview:
        cv2.namedWindow("Door Verification YuNet MobileFaceNet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Door Verification YuNet MobileFaceNet", args.width, args.height)

    print("\n[START] Verification dang chay. q/ESC de thoat.\n")
    frame_count = 0; infer_every = 6; last_verify = 0.0
    state="idle"; show_until=0.0; show_name=None; show_sim=0.0; show_bbox=None
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03); continue
            frame = cv2.flip(frame, 1)
            frame_count += 1; now = time.time()
            if now > show_until and state in ("ok","deny"):
                state="idle"; show_name=None; show_bbox=None; show_sim=0.0
            if frame_count % infer_every == 0 and now - last_verify >= args.cooldown:
                state="detecting"
                face = detector.largest(detector.detect(frame))
                if face is not None:
                    crop = detector.crop(frame, face)
                    if crop is not None:
                        emb = embedder.embed(crop)
                        name, sim = recognize(emb, db, args.threshold)
                        show_bbox = face["box"]; show_sim = sim
                        if name:
                            print(f"[OK] {time.strftime('%H:%M:%S')} name={name} sim={sim:.3f} -> MO KHOA")
                            state="ok"; show_name=name; show_until=now+args.unlock_duration+0.5; gpio.unlock(args.unlock_duration)
                        else:
                            print(f"[DENY] {time.strftime('%H:%M:%S')} sim={sim:.3f}")
                            state="deny"; show_name=None; show_until=now+1.5; gpio.deny()
                        last_verify = now
                else:
                    state="idle"; show_bbox=None
            if not args.no_preview:
                vis = frame.copy(); draw_overlay(vis, show_bbox, show_name, show_sim, state)
                cv2.imshow("Door Verification YuNet MobileFaceNet", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')): break
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        gpio.close()
        print("[INFO] Da giai phong tai nguyen")


if __name__ == "__main__":
    main()
