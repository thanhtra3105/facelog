"""
enrollment_ncnn.py — Đăng ký khuôn mặt bằng YuNet + NCNN ArcFace.
Chạy siêu nhẹ, thiết kế riêng cho Raspberry Pi / Orange Pi.
"""

import argparse
import json
import os
import time
from datetime import datetime
import cv2
import numpy as np
import ncnn

DB_PATH = "E:/HK8/TTNT/QuangDaAI/facelog/src/face_db.json"
# Kế thừa các hàm cơ sở từ file enrollment ONNX để code gọn hơn
from enrollment_yunet import (
     FaceDetector, blur_score, face_ratio, largest_face, check_distance, setup_camera, SCENARIOS, draw_ui
)

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# ── NCNN Core Functions ─────────────────────────
def init_ncnn_model(param_path, bin_path):
    net = ncnn.Net()
    net.opt.use_fp16_arithmetic = True # Kích hoạt tính toán FP16
    
    if not os.path.exists(param_path) or not os.path.exists(bin_path):
        print(f"[ERROR] Không tìm thấy file mô hình NCNN!")
        return None
        
    net.load_param(param_path)
    net.load_model(bin_path)
    print(f"[INFO] Khởi tạo thành công NCNN: {param_path}")
    return net

def extract_embedding_ncnn(net, face_bgr):
    face_resized = cv2.resize(face_bgr, (112, 112))
    mat_in = ncnn.Mat.from_pixels(face_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 112, 112)
    
    # Chuẩn hóa ảnh cho ArcFace (mean=127.5, scale=1/127.5)
    mat_in.substract_mean_normalize([127.5]*3, [0.0078125]*3)
    
    ex = net.create_extractor()
    ex.input("in0", mat_in)
    ret, mat_out = ex.extract("out0")
    
    emb = np.array(mat_out)
    # L2 Normalize
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb.tolist()


def enroll(name, param_path, bin_path, yunet_path, camera_id=0):
    net = init_ncnn_model(param_path, bin_path)
    if net is None: return

    detector = FaceDetector(yunet_path)
    db = load_db()
    cap = setup_camera(camera_id, 1280, 720, 30)

    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return

    all_embeddings = []
    scenario_meta = []

    print(f"\n[START] Bắt đầu đăng ký khuôn mặt cho: {name}")
    print("Nhấn SPACE để chụp từng cảnh; ESC để thoát.\n")

    for idx, (scene_name, instruction, expected_range, duration) in enumerate(SCENARIOS):
        print(f"  -> {idx + 1}/{len(SCENARIOS)}: {scene_name} - {instruction}")
        embeddings_this_scene = []
        start_time = None
        collecting = False
        collected_frames = 0
        last_msg, last_ratio, last_blur = "DANG TIM KHUON MAT", 0.0, 0.0
        TARGET_FRAMES = 18

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            face = largest_face(detector.detect(frame))
            has_good_face = False

            if face is not None:
                x, y, fw, fh = face["box"]
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 220, 255), 2)

                last_ratio = face_ratio(face, frame)
                ok_dist, dist_msg = check_distance(last_ratio, expected_range)
                face_crop = detector.crop(frame, face, align=True)
                last_blur = blur_score(face_crop)
                ok_blur = last_blur >= 35.0

                if not ok_dist: last_msg = dist_msg
                elif not ok_blur: last_msg = "ANH BI MO - GIU YEN / TANG SANG"
                else:
                    last_msg = "OK"
                    has_good_face = True

            if collecting and start_time:
                elapsed = time.time() - start_time
                countdown = max(0.0, duration - elapsed)

                if has_good_face and elapsed < duration:
                    # ---> TRÍCH XUẤT ĐẶC TRƯNG BẰNG NCNN <---
                    emb = extract_embedding_ncnn(net, face_crop)
                    embeddings_this_scene.append(emb)
                    collected_frames += 1

                if elapsed >= duration or collected_frames >= TARGET_FRAMES:
                    print(f"     v Thu thập {len(embeddings_this_scene)} embeddings")
                    break
            else:
                countdown = 0.0

            frame = draw_ui(
                frame, scene_name, instruction, countdown, idx, len(SCENARIOS),
                last_msg if collecting or face is not None else "Nhan SPACE khi da dung vi tri",
                last_ratio, last_blur
            )

            if not collecting:
                prompt = "Nhan SPACE de bat dau" if has_good_face else "Can chinh theo huong dan"
                cv2.putText(frame, prompt, (15, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.62, 
                            (0, 255, 150) if has_good_face else (0, 140, 255), 2)

            cv2.imshow("FaceLog - Enrollment NCNN", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC
                cap.release(); cv2.destroyAllWindows(); return
            if key == 32 and has_good_face and not collecting: # SPACE
                collecting = True; start_time = time.time()
                print("     [REC] Đang thu thập...")

        if embeddings_this_scene:
            arr = np.array(embeddings_this_scene)
            centroid = arr.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            sims = arr @ centroid
            top_k = max(3, int(len(arr) * 0.6))
            for i in np.argsort(sims)[-top_k:]:
                all_embeddings.append(arr[i].tolist())
            scenario_meta.append({
                "name": scene_name, "instruction": instruction,
                "num_frames": len(embeddings_this_scene),
                "expected_face_ratio": list(expected_range),
            })

    cap.release()
    cv2.destroyAllWindows()

    if not all_embeddings:
        print("[ERROR] Không thu thập được embedding nào!")
        return

    db[name] = {
        "embeddings": all_embeddings,
        "enrolled_at": datetime.now().isoformat(),
        "num_embeddings": len(all_embeddings),
        "num_scenarios": len(scenario_meta),
        "scenarios": scenario_meta,
        "detector": "yunet",
    }
    save_db(db)
    print(f"\n[THÀNH CÔNG] Đã đăng ký thành công: {name} (Lưu tại {DB_PATH})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  required=True, help="Tên người đăng ký")
    parser.add_argument("--param", default="E:/HK8/TTNT/QuangDaAI/my_model.ncnn.param")
    parser.add_argument("--bin",   default="E:/HK8/TTNT/QuangDaAI/my_model.ncnn.bin")
    parser.add_argument("--yunet", default="E:/HK8/TTNT/QuangDaAI/facelog/models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    enroll(args.name, args.param, args.bin, args.yunet, args.camera)