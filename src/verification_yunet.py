"""
verification_yunet.py — Xác thực khuôn mặt realtime bằng YuNet (đã tối ưu pipeline).

Thay đổi so với bản cũ:
  1. Tự động đọc EER threshold từ model thay vì hardcode 0.80
  2. identify() dùng top-5 mean thay vì max → ít bị outlier
  3. Smooth score dùng median thay vì mean → ổn định hơn

Chạy ví dụ:
  python verification_yunet.py --model models/arcface_vggface2.pth --yunet face_detection_yunet_2023mar.onnx
"""

import argparse
import json
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from enrollment_yunet import (
    DB_PATH,
    FaceDetector,
    FaceEmbedder,
    blur_score,
    face_ratio,
    largest_face,
    setup_camera,
    transform,
)


# ── Cấu hình mặc định (fallback nếu model không có EER threshold) ──────────
THRESHOLD_OPEN    = 0.80
THRESHOLD_UNSURE  = 0.70
SMOOTH_WINDOW     = 8
REQUIRED_SAME_NAME = 6
LOCK_COOLDOWN     = 3.0
MIN_FACE_RATIO_VERIFY = 0.075
MAX_FACE_RATIO_VERIFY = 0.45
MIN_BLUR_VERIFY   = 28.0


def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def identify(embedding, db, unsure_threshold: float):
    """
    So sánh embedding với toàn bộ DB.
    Dùng trung bình top-5 scores thay vì max để tránh bị outlier.
    Trả về (name, best_score).
    """
    if not db:
        return "UNKNOWN", 0.0

    best_name  = "UNKNOWN"
    best_score = 0.0

    for name, data in db.items():
        refs = data.get("embeddings", [])
        if not refs:
            continue
        scores = sorted(
            [cosine_similarity(embedding, ref) for ref in refs],
            reverse=True,
        )
        top_k = min(5, len(scores))
        score = float(np.mean(scores[:top_k]))   # top-5 mean, không dùng max
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < unsure_threshold:
        return "UNKNOWN", best_score
    return best_name, best_score


def load_threshold_from_model(model_path: str):
    """
    Đọc EER threshold đã tính sẵn trong notebook từ file .pth.
    Nếu không có thì fallback về giá trị mặc định.
    """
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            eer_th = ckpt.get("eer_threshold")
            if eer_th is not None:
                th_open   = float(eer_th)
                th_unsure = max(0.0, th_open - 0.08)
                print(f"[INFO] EER threshold từ model: open={th_open:.4f}, unsure={th_unsure:.4f}")
                return th_open, th_unsure
    except Exception as e:
        print(f"[WARN] Không đọc được threshold từ model: {e}")

    print(f"[INFO] Dùng threshold mặc định: open={THRESHOLD_OPEN}, unsure={THRESHOLD_UNSURE}")
    return THRESHOLD_OPEN, THRESHOLD_UNSURE


def draw_verification_ui(frame, name, score, status, door_open, fps, msg=""):
    h, w = frame.shape[:2]
    colors = {
        "OPEN":      (0, 230, 100),
        "UNSURE":    (0, 165, 255),
        "DENIED":    (0, 60, 220),
        "WAITING":   (180, 180, 180),
        "TOO_FAR":   (0, 180, 255),
        "TOO_CLOSE": (0, 180, 255),
        "BLUR":      (0, 180, 255),
    }
    color = colors.get(status, (180, 180, 180))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    display_name = name if name != "UNKNOWN" else "???"
    cv2.putText(frame, display_name, (15, 38),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame, f"Score: {score:.3f}",   (15, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1)
    if msg:
        cv2.putText(frame, msg, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    badge_text = {
        "OPEN":      "MO CUA",
        "UNSURE":    "NHAN DANG...",
        "DENIED":    "TU CHOI",
        "WAITING":   "DANG QUET...",
        "TOO_FAR":   "TIEN LAI GAN",
        "TOO_CLOSE": "LUI RA XA",
        "BLUR":      "ANH MO",
    }.get(status, "")

    badge_x = max(0, w - 220)
    cv2.rectangle(frame, (badge_x, 48), (w - 10, 85), color, -1)
    cv2.putText(frame, badge_text, (badge_x + 8, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2)

    bar_h  = 6
    bar_y  = h - bar_h - 2
    cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)
    filled = int(w * min(max(score, 0.0), 1.0))
    cv2.rectangle(frame, (0, bar_y), (filled, h), color, -1)

    if door_open:
        door_overlay = frame.copy()
        cv2.rectangle(door_overlay, (0, 0), (w, h), (0, 180, 80), -1)
        cv2.addWeighted(door_overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "WELCOME!", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 120), 3)

    return frame


def load_model(model_path: str, device):
    model = FaceEmbedder(embedding_dim=512, backbone="resnet50").to(device)
    if model_path and os.path.exists(model_path):
        checkpoint  = torch.load(model_path, map_location=device)
        state_dict  = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights: {model_path}")
        if isinstance(checkpoint, dict):
            print(f"[INFO] Best epoch: {checkpoint.get('best_epoch')}, "
                  f"AUC: {checkpoint.get('auc')}, "
                  f"EER: {checkpoint.get('eer')}, "
                  f"EER threshold: {checkpoint.get('eer_threshold')}")
    else:
        print("[WARN] Dùng random weights — chỉ để test UI.")
    model.eval()
    return model


def run_verification(
    model_path: str,
    yunet_path: str,
    camera_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps_cap: int = 30,
    threshold_open: float = THRESHOLD_OPEN,
    threshold_unsure: float = THRESHOLD_UNSURE,
):
    # ── Tự động lấy threshold từ model nếu người dùng không truyền tay ──
    # if threshold_open == THRESHOLD_OPEN and threshold_unsure == THRESHOLD_UNSURE:
    #     threshold_open, threshold_unsure = load_threshold_from_model(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model    = load_model(model_path, device)
    detector = FaceDetector(yunet_path)
    db       = load_db()
    print(f"[INFO] DB có {len(db)} người: {list(db.keys())}")
    print(f"[INFO] Threshold — open: {threshold_open:.4f}, unsure: {threshold_unsure:.4f}")

    cap = setup_camera(camera_id, width, height, fps_cap)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return

    score_history  = deque(maxlen=SMOOTH_WINDOW)
    name_history   = deque(maxlen=SMOOTH_WINDOW)
    door_open      = False
    door_open_time = 0.0
    last_open_time = 0.0
    last_fps_time  = time.time()
    fps            = 0.0

    print("\n[RUN] Xác thực — ESC: thoát, R: reload DB\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_fps_time, 1e-5))
        last_fps_time = now

        faces  = detector.detect(frame)
        face   = largest_face(faces)
        name, score, status = "UNKNOWN", 0.0, "WAITING"
        msg    = ""

        if face is not None:
            x, y, fw, fh = face["box"]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 200, 255), 2)
            r         = face_ratio(face, frame)
            face_crop = detector.crop(frame, face, align=True)
            bscore    = blur_score(face_crop)

            if r < MIN_FACE_RATIO_VERIFY:
                score_history.clear(); name_history.clear()
                status = "TOO_FAR"
                msg    = f"Mat qua nho: {r:.2f}. Hay tien lai gan."
            elif r > MAX_FACE_RATIO_VERIFY:
                score_history.clear(); name_history.clear()
                status = "TOO_CLOSE"
                msg    = f"Mat qua gan: {r:.2f}. Hay lui ra mot chut."
            elif bscore < MIN_BLUR_VERIFY:
                score_history.clear(); name_history.clear()
                status = "BLUR"
                msg    = f"Anh mo: {bscore:.0f}. Giu yen hoac tang anh sang."
            else:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                tensor   = transform(face_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor).cpu().numpy()[0]
                emb = emb / (np.linalg.norm(emb) + 1e-8)

                raw_name, raw_score = identify(emb, db, threshold_unsure)
                score_history.append(raw_score)
                name_history.append(raw_name)

                # Dùng median thay vì mean → ổn định hơn khi có frame nhiễu
                smooth_score             = float(np.median(score_history))
                most_common, same_count  = Counter(name_history).most_common(1)[0]

                if (
                    smooth_score >= threshold_open
                    and most_common != "UNKNOWN"
                    and same_count >= REQUIRED_SAME_NAME
                ):
                    status = "OPEN"
                    if not door_open and (now - last_open_time >= LOCK_COOLDOWN):
                        door_open      = True
                        door_open_time = now
                        last_open_time = now
                        print(f"  🔓 MỞ CỬA — {most_common} "
                              f"(score={smooth_score:.3f}, same={same_count}/{SMOOTH_WINDOW})")
                elif smooth_score >= threshold_unsure:
                    status = "UNSURE"
                else:
                    status = "DENIED"

                name  = most_common
                score = smooth_score
                msg   = (f"ratio={r:.2f} blur={bscore:.0f} "
                         f"same={same_count}/{SMOOTH_WINDOW} "
                         f"th={threshold_open:.3f}")
        else:
            score_history.clear(); name_history.clear()
            status = "WAITING"
            msg    = "Khong thay khuon mat"

        if door_open and (now - door_open_time > 2.0):
            door_open = False

        frame = draw_verification_ui(frame, name, score, status, door_open, fps, msg)
        cv2.imshow("FaceLog — Verification YuNet", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            db = load_db()
            print(f"[INFO] Reload DB: {list(db.keys())}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            default="E:/THANHTRA/KI_8/AI/facelog/models/arcface_vggface2.pth")
    parser.add_argument("--yunet",            default="face_detection_yunet_2023mar.onnx")
    parser.add_argument("--camera",           type=int,   default=0)
    parser.add_argument("--width",            type=int,   default=1280)
    parser.add_argument("--height",           type=int,   default=720)
    parser.add_argument("--fps",              type=int,   default=30)
    parser.add_argument("--open-threshold",   type=float, default=THRESHOLD_OPEN)
    parser.add_argument("--unsure-threshold", type=float, default=THRESHOLD_UNSURE)
    args = parser.parse_args()

    run_verification(
        model_path       = args.model,
        yunet_path       = args.yunet,
        camera_id        = args.camera,
        width            = args.width,
        height           = args.height,
        fps_cap          = args.fps,
        threshold_open   = args.open_threshold,
        threshold_unsure = args.unsure_threshold,
    )