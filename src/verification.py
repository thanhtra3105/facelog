"""
verification.py — Xác thực khuôn mặt real-time qua webcam
Chạy: python verification.py
"""

import cv2
import torch
import numpy as np
import json
import os
import time
from collections import deque

# Import từ enrollment.py
import sys
sys.path.append(os.path.dirname(__file__))
from enrollment import FaceEmbedder, FaceDetector, transform, DB_PATH

# ── Cấu hình ──────────────────────────────────────────────────────────────
THRESHOLD_OPEN   = 0.8   # similarity > này → MỞ CỬA
THRESHOLD_UNSURE = 0.7   # similarity > này → NHẬN DẠNG ĐƯỢC nhưng chưa đủ
SMOOTH_WINDOW    = 8      # làm mượt qua N frames
LOCK_COOLDOWN    = 3.0    # giây trước khi có thể mở lại


# ── Database helper ────────────────────────────────────────────────────────
def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH) as f:
        return json.load(f)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def identify(embedding, db):
    """
    So sánh embedding với toàn bộ DB
    Trả về (name, best_score) hoặc ("UNKNOWN", score)
    """
    if not db:
        return "UNKNOWN", 0.0

    best_name  = "UNKNOWN"
    best_score = 0.0

    for name, data in db.items():
        # So sánh với tất cả embedding đã lưu (các hướng)
        scores = [cosine_similarity(embedding, ref)
                  for ref in data["embeddings"]]
        score = max(scores)   # lấy điểm cao nhất

        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < THRESHOLD_UNSURE:
        return "UNKNOWN", best_score
    return best_name, best_score


# ── Liveness check đơn giản (Eye Aspect Ratio — blink detection) ──────────
def eye_aspect_ratio(eye_points):
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C + 1e-6)


# ── UI Drawing ─────────────────────────────────────────────────────────────
def draw_verification_ui(frame, name, score, status, door_open, fps):
    h, w = frame.shape[:2]

    # Màu theo trạng thái
    colors = {
        "OPEN":    (0, 230, 100),
        "UNSURE":  (0, 165, 255),
        "DENIED":  (0, 60, 220),
        "WAITING": (180, 180, 180),
    }
    color = colors.get(status, (180, 180, 180))

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Tên + score
    display_name = name if name != "UNKNOWN" else "???"
    cv2.putText(frame, display_name, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame, f"Score: {score:.3f}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # FPS
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Status badge
    badge_text = {
        "OPEN":    "MO CUA",
        "UNSURE":  "NHAN DANG...",
        "DENIED":  "TU CHOI",
        "WAITING": "DANG QUET...",
    }.get(status, "")

    badge_x = w - 200
    cv2.rectangle(frame, (badge_x, 45), (w - 10, 78), color, -1)
    cv2.putText(frame, badge_text, (badge_x + 8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2)

    # Score bar
    bar_h = 6
    bar_y = h - bar_h - 2
    cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)
    filled = int(w * min(score, 1.0))
    cv2.rectangle(frame, (0, bar_y), (filled, h), color, -1)
    # Threshold line
    thresh_x = int(w * THRESHOLD_OPEN)
    cv2.line(frame, (thresh_x, bar_y - 4), (thresh_x, h), (255, 255, 0), 2)

    # Door animation
    if door_open:
        door_overlay = frame.copy()
        cv2.rectangle(door_overlay, (0, 0), (w, h), (0, 180, 80), -1)
        cv2.addWeighted(door_overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "WELCOME!", (w//2 - 90, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 120), 3)

    return frame


# ── Main verification loop ─────────────────────────────────────────────────
def run_verification(model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = FaceEmbedder(embedding_dim=512, backbone='resnet50').to(device)
    if model_path and os.path.exists(model_path):
        # Thành:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights: {model_path}")
        print(f"[INFO] Best epoch: {checkpoint.get('best_epoch')}, AUC: {checkpoint.get('auc')}")
        print(f"[INFO] Loaded: {model_path}")
    else:
        print("[WARN] Dùng random weights — chỉ để test UI")
    model.eval()

    detector  = FaceDetector()
    db        = load_db()
    print(f"[INFO] DB có {len(db)} người: {list(db.keys())}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!"); return

    # State
    score_history = deque(maxlen=SMOOTH_WINDOW)
    name_history  = deque(maxlen=SMOOTH_WINDOW)
    door_open     = False
    door_open_time = 0
    last_fps_time  = time.time()
    fps = 0

    print("\n[RUN] Bắt đầu xác thực — nhấn ESC để thoát, R để reload DB\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_fps_time, 1e-5))
        last_fps_time = now

        faces = detector.detect(frame)
        name, score, status = "UNKNOWN", 0.0, "WAITING"

        if len(faces) > 0:
            # Lấy khuôn mặt lớn nhất
            biggest = max(faces, key=lambda f: f[2]*f[3])
            x, y, w_f, h_f = biggest
            cv2.rectangle(frame, (x, y), (x+w_f, y+h_f), (0, 200, 255), 2)

            face_crop = detector.crop(frame, biggest)
            face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor    = transform(face_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor).cpu().numpy()[0]

            name, score = identify(emb, db)

            # Smooth qua nhiều frames
            score_history.append(score)
            name_history.append(name)
            smooth_score = np.mean(score_history)

            # Lấy tên xuất hiện nhiều nhất
            from collections import Counter
            most_common = Counter(name_history).most_common(1)[0][0]

            # Quyết định
            if smooth_score >= THRESHOLD_OPEN and most_common != "UNKNOWN":
                status = "OPEN"
                if not door_open:
                    door_open = True
                    door_open_time = now
                    print(f"  🔓 MỞ CỬA — {most_common} (score={smooth_score:.3f})")
            elif smooth_score >= THRESHOLD_UNSURE:
                status = "UNSURE"
            else:
                status = "DENIED"

            score = smooth_score
            name  = most_common
        else:
            score_history.clear()
            name_history.clear()
            status = "WAITING"

        # Tắt hiệu ứng mở cửa sau 2 giây
        if door_open and (now - door_open_time > 2.0):
            door_open = False

        frame = draw_verification_ui(frame, name, score, status, door_open, fps)
        cv2.imshow("FaceLog — Verification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   break        # ESC
        if key == ord('r'):
            db = load_db()
            print(f"[INFO] Reload DB: {list(db.keys())}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:/THANHTRA/KI_8/AI/project/models/arcface_vggface2.pth")
    args = parser.parse_args()
    run_verification(args.model)
