"""
verification_ncnn.py — Xác thực khuôn mặt realtime bằng YuNet + NCNN ArcFace.
Tốc độ cực cao, tối ưu cho Raspberry Pi / Orange Pi.
"""

import argparse
import json
import os
import time
from collections import Counter, deque
import cv2
import numpy as np
import ncnn
#import serial # Thư viện giao tiếp UART với phần cứng

# Import từ file enrollment (đảm bảo file enrollment_yunet.py nằm cùng thư mục)
from enrollment_yunet2_onnx import (
    DB_PATH, FaceDetector, blur_score, face_ratio, largest_face, setup_camera
)

# ── CẤU HÌNH NGƯỠNG & PHẦN CỨNG ──────────
THRESHOLD_OPEN    = 0.80
THRESHOLD_UNSURE  = 0.70
SMOOTH_WINDOW     = 8
REQUIRED_SAME_NAME = 6
LOCK_COOLDOWN     = 3.0

# Cấu hình cổng UART kết nối với mạch điều khiển khóa
# Trên Pi thường là /dev/ttyS0, /dev/ttyUSB0 hoặc /dev/ttyAMA0
UART_PORT = "/dev/ttyUSB0" 
BAUD_RATE = 115200

# Khởi tạo kết nối Serial (Bỏ comment khi chạy trên Pi có cắm mạch)
# try:
#     uart = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
#     print(f"[INFO] Đã kết nối UART tại {UART_PORT}")
# except Exception as e:
#     print(f"[WARN] Không thể mở UART: {e}")
#     uart = None

def load_db():
    if not os.path.exists(DB_PATH): return {}
    with open(DB_PATH, encoding="utf-8") as f: return json.load(f)

def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def identify(embedding, db, unsure_threshold):
    if not db: return "UNKNOWN", 0.0
    best_name, best_score = "UNKNOWN", 0.0
    for name, data in db.items():
        refs = data.get("embeddings", [])
        if not refs: continue
        scores = sorted([cosine_similarity(embedding, ref) for ref in refs], reverse=True)
        score = float(np.mean(scores[:min(5, len(scores))]))
        if score > best_score:
            best_score = score
            best_name  = name
    return best_name if best_score >= unsure_threshold else "UNKNOWN", best_score

def init_ncnn_model(param_path, bin_path):
    net = ncnn.Net()
    net.opt.use_fp16_arithmetic = True # Kích hoạt tính toán FP16
    net.load_param(param_path)
    net.load_model(bin_path)
    print(f"[INFO] Đã tải mô hình NCNN: {param_path}")
    return net

def extract_embedding_ncnn(net, face_bgr):
    face_resized = cv2.resize(face_bgr, (112, 112))
    mat_in = ncnn.Mat.from_pixels(face_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 112, 112)
    
    # Chuẩn hóa ảnh: (pixel - 127.5) * 0.0078125
    mat_in.substract_mean_normalize([127.5]*3, [0.0078125]*3)
    
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")
    
    emb = np.array(mat_out)
    return (emb / (np.linalg.norm(emb) + 1e-8)).tolist()

def run_verification(param_path, bin_path, yunet_path, camera_id=0):
    net = init_ncnn_model(param_path, bin_path)
    detector = FaceDetector(yunet_path)
    db = load_db()
    
    cap = setup_camera(camera_id, 1280, 720, 30)
    score_history, name_history = deque(maxlen=SMOOTH_WINDOW), deque(maxlen=SMOOTH_WINDOW)
    
    door_open = False
    last_open_time = 0.0
    
    print("\n[RUN] Hệ thống Xác thực NCNN Đang Chạy...\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        now = time.time()

        face = largest_face(detector.detect(frame))
        
        if face is not None:
            x, y, fw, fh = face["box"]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            
            face_crop = detector.crop(frame, face, align=True)
            if blur_score(face_crop) >= 28.0:
                # 1. Trích xuất đặc trưng bằng NCNN
                emb = extract_embedding_ncnn(net, face_crop)
                
                # 2. Định danh
                raw_name, raw_score = identify(emb, db, THRESHOLD_UNSURE)
                score_history.append(raw_score)
                name_history.append(raw_name)

                smooth_score = float(np.median(score_history))
                most_common, same_count = Counter(name_history).most_common(1)[0]

                # 3. Kích hoạt mở cửa
                if smooth_score >= THRESHOLD_OPEN and most_common != "UNKNOWN" and same_count >= REQUIRED_SAME_NAME:
                    if not door_open and (now - last_open_time >= LOCK_COOLDOWN):
                        door_open = True
                        last_open_time = now
                        print(f" 🔓 MỞ CỬA — {most_common} (Score: {smooth_score:.3f})")
                        
                        # ---> GỬI LỆNH UART XUỐNG MẠCH <---
                        # if uart:
                        #     uart.write(b"OPEN\n")
        else:
            score_history.clear(); name_history.clear()

        if door_open and (now - last_open_time > 2.0): door_open = False
        
        # Hiển thị text lên màn hình
        status_text = f"MỞ CỬA ({most_common})" if door_open else "DANG QUET..."
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if door_open else (0, 165, 255), 2)
        cv2.imshow("Logface NCNN", frame)

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="E:/HK8/TTNT/QuangDaAI/my_model.ncnn.param")
    parser.add_argument("--bin",   default="E:/HK8/TTNT/QuangDaAI/my_model.ncnn.bin")
    parser.add_argument("--yunet", default="E:/HK8/TTNT/QuangDaAI/facelog/models/face_detection_yunet_2023mar.onnx")
    args = parser.parse_args()
    
    run_verification(args.param, args.bin, args.yunet)