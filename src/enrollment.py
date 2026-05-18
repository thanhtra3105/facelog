"""
enrollment.py — Thu thập khuôn mặt từ 4 hướng + trích xuất embedding
Chạy: python enrollment.py --name "Nguyen Van A"
"""

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import torchvision.models as models
import torch.nn as nn

class FaceEmbedder(nn.Module):
    """ResNet-50 backbone — khớp với notebook training"""
    def __init__(self, embedding_dim=512, backbone='resnet50'):
        super().__init__()
        if backbone == 'resnet50':
            base = models.resnet50(weights=None)
            in_features = 2048
        else:
            base = models.resnet34(weights=None)
            in_features = 512

        self.features = nn.Sequential(*list(base.children())[:-2])
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.bn1  = nn.BatchNorm2d(in_features)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(in_features, embedding_dim, bias=False)
        self.bn2  = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.bn1(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        x = self.bn2(x)
        return nn.functional.normalize(x, p=2, dim=1)

# ── Face Detector (dùng Haar Cascade built-in OpenCV) ─────────────────────
class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        return faces  # list of (x, y, w, h)

    def crop(self, frame, box, size=112):
        x, y, w, h = box
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, (size, size))


# ── Transform pipeline ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


# ── Enrollment App ─────────────────────────────────────────────────────────
DIRECTIONS = [
    ("THẲNG",   "Nhìn thẳng vào camera",         3.0),
    ("TRÁI",    "Quay mặt sang TRÁI nhẹ (~30°)",  3.0),
    ("PHẢI",    "Quay mặt sang PHẢI nhẹ (~30°)",  3.0),
    ("LÊN",     "Ngẩng đầu lên nhẹ",              3.0),
    ("XUỐNG",   "Cúi đầu xuống nhẹ",              3.0),
]

DB_PATH = "face_db.json"


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            return json.load(f)
    return {}


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def extract_embedding(model, face_bgr, device):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy()[0]
    return emb.tolist()


def draw_ui(frame, direction_name, instruction, countdown, collected, total_dirs):
    h, w = frame.shape[:2]

    # Overlay tối
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h-70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Tiêu đề
    cv2.putText(frame, f"ENROLLMENT  [{collected}/{total_dirs} huong]",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)

    # Hướng + hướng dẫn
    cv2.putText(frame, direction_name, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    cv2.putText(frame, instruction, (15, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Countdown
    if countdown > 0:
        cv2.putText(frame, f"{countdown:.1f}s", (w - 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    # Progress bar
    bar_w = int(w * collected / total_dirs)
    cv2.rectangle(frame, (0, h - 8), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 200, 100), -1)

    return frame


def enroll(name: str, model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    model = FaceEmbedder(embedding_dim=512, backbone='resnet50').to(device)
    if model_path and os.path.exists(model_path):
        # Thành:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights: {model_path}")
        print(f"[INFO] Best epoch: {checkpoint.get('best_epoch')}, AUC: {checkpoint.get('auc')}")
        print(f"[INFO] Loaded weights: {model_path}")
    else:
        print("[WARN] Không tìm thấy weights → dùng random (chỉ để test UI)")
    model.eval()

    detector = FaceDetector()
    db = load_db()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return

    all_embeddings = []
    print(f"\n[START] Đăng ký khuôn mặt cho: {name}")
    print("Nhấn SPACE để chụp từng hướng, ESC để thoát\n")

    for idx, (dir_name, instruction, duration) in enumerate(DIRECTIONS):
        print(f"  → Hướng {idx+1}/{len(DIRECTIONS)}: {dir_name} — {instruction}")
        embeddings_this_dir = []
        start_time = None
        collecting = False
        collected_frames = 0
        TARGET_FRAMES = 15  # lấy 15 frames/hướng

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror

            faces = detector.detect(frame)
            has_face = len(faces) > 0

            # Vẽ bounding box
            for (x, y, w, h) in faces:
                color = (0, 255, 100) if has_face else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            countdown = 0.0
            if collecting and start_time:
                elapsed = time.time() - start_time
                countdown = max(0, duration - elapsed)

                if has_face and elapsed < duration:
                    face_crop = detector.crop(frame, faces[0])
                    emb = extract_embedding(model, face_crop, device)
                    embeddings_this_dir.append(emb)
                    collected_frames += 1

                if elapsed >= duration or collected_frames >= TARGET_FRAMES:
                    print(f"     ✓ Thu thập {len(embeddings_this_dir)} embeddings")
                    break

            frame = draw_ui(frame, dir_name, instruction,
                            countdown, idx, len(DIRECTIONS))

            # Hướng dẫn bấm phím
            if not collecting:
                msg = "Nhan SPACE de bat dau chup" if has_face else "Khong tim thay khuon mat!"
                color = (0, 255, 200) if has_face else (0, 100, 255)
                cv2.putText(frame, msg, (15, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            cv2.imshow("FaceLog — Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cap.release(); cv2.destroyAllWindows(); return
            if key == 32 and has_face and not collecting:  # SPACE
                collecting = True
                start_time = time.time()
                print(f"     [REC] Đang thu thập...")

        # Tính embedding trung bình cho hướng này
        if embeddings_this_dir:
            mean_emb = np.mean(embeddings_this_dir, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)  # normalize lại
            all_embeddings.append(mean_emb.tolist())

    cap.release()
    cv2.destroyAllWindows()

    if not all_embeddings:
        print("[ERROR] Không thu thập được embedding nào!")
        return

    # Lưu vào DB
    db[name] = {
        "embeddings": all_embeddings,          # 5 embedding (1/hướng)
        "enrolled_at": datetime.now().isoformat(),
        "num_directions": len(all_embeddings),
    }
    save_db(db)
    print(f"\n✅ Đã đăng ký thành công: {name}")
    print(f"   → {len(all_embeddings)} hướng, lưu tại {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Tên người đăng ký")
    parser.add_argument("--model", default="E:/HK8/TTNT/QuangDaAI/facelog/models/arcface_vggface2.pth")
    args = parser.parse_args()
    enroll(args.name, args.model)
