#!/usr/bin/env python3
"""
register_yunet_mobilefacenet.py
Dang ky khuon mat bang YuNet + MobileFaceNet TFLite.

- Khong dung mediapipe.
- Mac dinh dung Pi Camera qua Picamera2.
- Co the dung USB camera bang --use-usb-camera --camera 9.
- Luu database dung format cu: face_database.json = {"name": [[emb], [emb], ...]}

Can co file mobilefacenet.tflite cung thu muc voi script.
YuNet model face_detection_yunet_2023mar.onnx se tu tai neu thieu.

Chay Pi Camera:
    python register_yunet_mobilefacenet.py --name "tra"

Chay USB camera:
    python register_yunet_mobilefacenet.py --name "tra" --use-usb-camera --camera 9
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
MODEL_PATH = BASE_DIR / "mobilefacenet.tflite"
DB_PATH = BASE_DIR / "face_database.json"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
IMG_SIZE = 112

SCENARIOS = [
    ("THANG_VUA", "Nhin thang vao camera, giu mat o khoang cach vua", (0.14, 0.35), 2.5),
    ("NGHIENG_TRAI", "Quay mat sang TRAI nhe", (0.12, 0.35), 2.5),
    ("NGHIENG_PHAI", "Quay mat sang PHAI nhe", (0.12, 0.35), 2.5),
    ("NGAU_LEN", "Ngua dau len nhe", (0.12, 0.35), 2.5),
    ("CUI_XUONG", "Cui dau xuong nhe", (0.12, 0.35), 2.5),
]


def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return
    print(f"[INFO] Dang tai YuNet model -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Da tai YuNet: {path.stat().st_size // 1024} KB")


class MobileFaceNet:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay {model_path}. Hay dat mobilefacenet.tflite cung thu muc script.")
        self.interp = Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        print(f"[INFO] MobileFaceNet ready: {model_path}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 128.0
        self.interp.set_tensor(self.inp["index"], face[np.newaxis])
        self.interp.invoke()
        emb = self.interp.get_tensor(self.outp["index"])[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)


class YuNetDetector:
    def __init__(self, model_path: Path = YUNET_PATH, score_threshold: float = 0.82):
        ensure_yunet(model_path)
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError("OpenCV cua ban khong co FaceDetectorYN_create. Hay cai python3-opencv ban moi hoac opencv-python moi.")
        self.detector = cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )
        print(f"[INFO] YuNet ready: {model_path}")

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None:
            return []
        results = []
        for f in faces:
            x, y, bw, bh = f[:4]
            landmarks = f[4:14].reshape(5, 2).astype(np.float32)
            score = float(f[-1])
            results.append({
                "box": (int(x), int(y), int(bw), int(bh)),
                "landmarks": landmarks,
                "score": score,
            })
        return results

    @staticmethod
    def largest(faces: List[Dict]) -> Optional[Dict]:
        if not faces:
            return None
        return max(faces, key=lambda f: f["box"][2] * f["box"][3])

    @staticmethod
    def face_ratio(face: Dict, frame: np.ndarray) -> float:
        _, _, w, h = face["box"]
        return max(w, h) / max(1, min(frame.shape[:2]))

    @staticmethod
    def align_crop(frame_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> Optional[np.ndarray]:
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        if size != 112:
            dst *= size / 112.0
        try:
            M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
            if M is None:
                return None
            return cv2.warpAffine(frame_bgr, M, (size, size), borderValue=0.0)
        except cv2.error:
            return None

    @staticmethod
    def crop_fallback(frame_bgr: np.ndarray, face: Dict, pad_ratio: float = 0.18) -> Optional[np.ndarray]:
        x, y, w, h = face["box"]
        pad = int(pad_ratio * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (112, 112))

    def crop(self, frame_bgr: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        crop = self.align_crop(frame_bgr, face["landmarks"], size=112)
        if crop is not None:
            return crop
        return self.crop_fallback(frame_bgr, face)


def blur_score(face_bgr: Optional[np.ndarray]) -> float:
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_camera(camera: int, width: int, height: int, fps: int, use_usb_camera: bool):
    if not use_usb_camera:
        try:
            from picamera2 import Picamera2
            class PiCamWrap:
                def __init__(self):
                    self.picam2 = Picamera2()
                    config = self.picam2.create_preview_configuration(
                        main={"size": (width, height), "format": "RGB888"},
                        controls={"FrameRate": fps},
                    )
                    self.picam2.configure(config)
                    self.picam2.start()
                    time.sleep(0.5)
                    print(f"[INFO] Dung Pi Camera: {width}x{height}@{fps}")
                def isOpened(self):
                    return True
                def read(self):
                    rgb = self.picam2.capture_array()
                    if rgb is None:
                        return False, None
                    return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                def release(self):
                    self.picam2.stop()
            return PiCamWrap()
        except Exception as e:
            print(f"[WARN] Khong mo duoc Pi Camera: {e}")
            print("[WARN] Fallback sang USB/OpenCV camera")

    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"[INFO] Dung OpenCV camera id={camera}: {width}x{height}@{fps}")
    return cap


def load_db(path: Path = DB_PATH) -> Dict[str, List[List[float]]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_db(db: Dict[str, List[List[float]]], path: Path = DB_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def draw_ui(frame, scene, instruction, msg, collected, total, ratio, blur, bbox=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"REGISTER [{collected}/{total}]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 150), 2)
    cv2.putText(frame, scene, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
    cv2.putText(frame, f"ratio={ratio:.2f} blur={blur:.0f}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, instruction, (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1)
    color = (0, 255, 150) if msg == "OK" else (0, 160, 255)
    cv2.putText(frame, msg, (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 255), 2)
    bar = int(w * collected / max(1, total))
    cv2.rectangle(frame, (0, h - 7), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 7), (bar, h), (0, 200, 100), -1)


def enroll(args):
    detector = YuNetDetector(score_threshold=args.score_threshold)
    embedder = MobileFaceNet(MODEL_PATH)
    cap = open_camera(args.camera, args.width, args.height, args.fps, args.use_usb_camera)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc camera")

    db = load_db(DB_PATH)
    all_embeddings: List[List[float]] = []

    if not args.no_preview:
        cv2.namedWindow("Register YuNet MobileFaceNet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Register YuNet MobileFaceNet", args.width, args.height)

    print(f"\n[START] Dang ky: {args.name}")
    print("Khong can bam chup. He thong tu thu mau khi mat dat yeu cau. ESC/q de thoat.\n")

    try:
        for idx, (scene, instruction, expected_range, duration) in enumerate(SCENARIOS):
            print(f"-> {idx+1}/{len(SCENARIOS)}: {scene} - {instruction}")
            scene_embs = []
            start_good = None
            collecting = False
            collect_start = None
            last_msg = "DANG TIM MAT"
            last_ratio = 0.0
            last_blur = 0.0
            last_bbox = None

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                faces = detector.detect(frame)
                face = detector.largest(faces)
                good = False
                crop = None

                if face is not None:
                    last_bbox = face["box"]
                    last_ratio = detector.face_ratio(face, frame)
                    low, high = expected_range
                    crop = detector.crop(frame, face)
                    last_blur = blur_score(crop)

                    if last_ratio < low:
                        last_msg = "TIEN LAI GAN CAMERA"
                    elif last_ratio > high:
                        last_msg = "LUI RA XA CAMERA"
                    elif last_blur < args.min_blur:
                        last_msg = "ANH MO - GIU YEN/TANG SANG"
                    else:
                        last_msg = "OK"
                        good = True
                else:
                    last_bbox = None
                    last_ratio = 0.0
                    last_blur = 0.0
                    last_msg = "KHONG TIM THAY MAT"

                now = time.time()
                if good:
                    if start_good is None:
                        start_good = now
                    if not collecting and now - start_good >= args.stable_time:
                        collecting = True
                        collect_start = now
                        print("   [REC] Tu dong thu mau...")
                else:
                    start_good = None
                    if not collecting:
                        collect_start = None

                if collecting and crop is not None and good:
                    emb = embedder.embed(crop)
                    scene_embs.append(emb.tolist())
                    if len(scene_embs) >= args.target_frames or (collect_start and now - collect_start >= duration):
                        print(f"   OK: thu {len(scene_embs)} embeddings")
                        break

                if not args.no_preview:
                    vis = frame.copy()
                    draw_ui(vis, scene, instruction, last_msg, idx, len(SCENARIOS), last_ratio, last_blur, last_bbox)
                    cv2.imshow("Register YuNet MobileFaceNet", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        raise KeyboardInterrupt
                else:
                    time.sleep(0.02)

            if scene_embs:
                arr = np.array(scene_embs, dtype=np.float32)
                centroid = arr.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                sims = arr @ centroid
                top_k = max(3, int(len(arr) * 0.6))
                top_idx = np.argsort(sims)[-top_k:]
                for i in top_idx:
                    all_embeddings.append(arr[i].tolist())

    except KeyboardInterrupt:
        print("\n[STOP] Da huy dang ky")
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if not all_embeddings:
        print("[ERROR] Khong thu duoc embedding nao")
        return 1

    db[args.name] = all_embeddings
    save_db(db, DB_PATH)
    print(f"\n[DONE] Da dang ky: {args.name}")
    print(f"       So embedding luu: {len(all_embeddings)}")
    print(f"       Database: {DB_PATH}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--camera", type=int, default=9)
    p.add_argument("--use-usb-camera", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--target-frames", type=int, default=12)
    p.add_argument("--min-blur", type=float, default=25.0)
    p.add_argument("--stable-time", type=float, default=0.6)
    p.add_argument("--score-threshold", type=float, default=0.82)
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()
    raise SystemExit(enroll(args))


if __name__ == "__main__":
    main()
