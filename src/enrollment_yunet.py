"""
enrollment_yunet.py — Đăng ký khuôn mặt bằng YuNet + nhiều khoảng cách/góc nhìn.

Chuẩn bị:
  1) Cài OpenCV mới: pip install -U opencv-python
  2) Tải model YuNet: face_detection_yunet_2023mar.onnx
     Đặt file .onnx cùng thư mục với script này hoặc truyền --yunet đường_dẫn.onnx

Chạy ví dụ:
  python enrollment_yunet.py --name "Le Thanh Tra" --model "E:/THANHTRA/KI_8/AI/facelog/models/arcface_vggface2.pth" --yunet face_detection_yunet_2023mar.onnx
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort  # <-- THÊM MỚI: Dùng ONNX Runtime thay cho PyTorch

# XÓA HOÀN TOÀN CÁC DÒNG LIÊN QUAN ĐẾN TORCH DƯỚI ĐÂY:
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms

# ── THÊM MỚI: Hàm tiền xử lý ảnh bằng Numpy (Thay cho transforms của PyTorch) ──
def preprocess_face_numpy(face_rgb):
    # Resize về chuẩn 112x112 của ArcFace
    resized = cv2.resize(face_rgb, (112, 112))
    # Chuẩn hóa về dải [-1, 1]
    img_float = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    # Đổi trục từ [H, W, C] sang [C, H, W]
    img_chw = np.transpose(img_float, (2, 0, 1))
    # Thêm batch_size ở đầu -> shape: [1, 3, 112, 112]
    return np.expand_dims(img_chw, axis=0)
DB_PATH = "face_db.json"


# ── Model embedding: giữ nguyên kiến trúc của bạn ─────────────────────────
# ── THAY THẾ: Sử dụng ONNX Runtime và Numpy (Sạch bóng PyTorch) ──────────────
import onnxruntime as ort

class MockTensor:
    """
    Lớp giả lập Tensor của PyTorch.
    Giúp 'đánh lừa' các đoạn code phía dưới của enrollment.py để không bị lỗi 
    khi gọi .unsqueeze(), .cpu(), .numpy(), .detach()
    """
    def __init__(self, array):
        self.array = array
    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self.array, axis=dim))
    def to(self, *args, **kwargs):
        return self
    def cpu(self):
        return self
    def detach(self):
        return self
    def numpy(self):
        return self.array
    def __getitem__(self, idx):
        return self.array[idx]

class FaceEmbedder:
    """Bộ trích xuất đặc trưng dùng ONNX Runtime (Giao diện giả lập PyTorch)"""
    def __init__(self, embedding_dim: int = 512, backbone: str = "resnet50", model_path: str = "models/arcface_vggface2.onnx"):
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
            
        print(f"[INFO] Enrollment Model (ONNX) đang nạp: {model_path}")
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def forward(self, mock_tensor):
        input_data = mock_tensor.array
        ort_outputs = self.ort_session.run(None, {'input': input_data})
        return MockTensor(ort_outputs[0])

    def __call__(self, mock_tensor):
        return self.forward(mock_tensor)
    
    # ── THÊM 2 HÀM NÀY ĐỂ TRÁNH LỖI PHÍA DƯỚI ────────────────────────
    def to(self, *args, **kwargs):
        return self  # Bỏ qua lệnh .to(device) của PyTorch
        
    def eval(self):
        return self


def transform(face_img):
    """Hàm tiền xử lý ảnh bằng Numpy thay thế hoàn toàn cho torchvision.transforms"""
    # Nếu code phía dưới truyền vào ảnh dạng PIL Image, chuyển nó về Numpy
    if not isinstance(face_img, np.ndarray):
        face_img = np.array(face_img)
        
    # Chuẩn hóa kích thước về 112x112 chuẩn ArcFace
    resized = cv2.resize(face_img, (112, 112))
    
    # Chuẩn hóa pixel về dải [-1, 1] (giống mean=[0.5], std=[0.5] của PyTorch)
    img_float = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    # Đổi trục từ [H, W, C] sang [C, H, W] để khớp định dạng mạng nơ-ron
    img_chw = np.transpose(img_float, (2, 0, 1))
    
    return MockTensor(img_chw)


# ── Face detector: ưu tiên YuNet, fallback Haar nếu thiếu model ───────────
def _script_dir() -> Path:
    return Path(__file__).resolve().parent


class FaceDetector:
    """
    Trả về mỗi face dưới dạng dict:
      {
        "box": (x, y, w, h),
        "score": confidence,
        "landmarks": ndarray shape (5,2) hoặc None
      }

    YuNet có landmark 5 điểm, dùng để align mặt về 112x112.
    """

    def __init__(
        self,
        model_path: str = "face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.82,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ):
        self.name = "haar"
        self.use_yunet = False
        self.detector = None

        yunet_path = Path(model_path)
        if not yunet_path.is_absolute():
            yunet_path = _script_dir() / yunet_path

        if hasattr(cv2, "FaceDetectorYN_create") and yunet_path.exists():
            self.detector = cv2.FaceDetectorYN_create(
                str(yunet_path),
                "",
                (320, 320),
                score_threshold,
                nms_threshold,
                top_k,
            )
            self.name = "yunet"
            self.use_yunet = True
            print(f"[INFO] Dùng YuNet detector: {yunet_path}")
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            print("[WARN] Không dùng được YuNet. Fallback sang Haar Cascade.")
            print("       Kiểm tra: file face_detection_yunet_2023mar.onnx và phiên bản OpenCV.")

    def detect(self, frame) -> List[Dict]:
        h, w = frame.shape[:2]

        if self.use_yunet:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None:
                return []

            results = []
            for f in faces:
                x, y, bw, bh = f[:4]
                score = float(f[-1])
                landmarks = f[4:14].reshape(5, 2).astype(np.float32)
                results.append(
                    {
                        "box": (int(x), int(y), int(bw), int(bh)),
                        "score": score,
                        "landmarks": landmarks,
                    }
                )
            return results

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
        )
        return [
            {"box": tuple(map(int, f)), "score": 1.0, "landmarks": None}
            for f in faces
        ]

    @staticmethod
    def _clip_box(frame, box, pad_ratio: float = 0.18):
        x, y, w, h = box
        pad = int(pad_ratio * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        return x1, y1, x2, y2

    def crop(self, frame, face: Dict, size: int = 112, align: bool = True):
        """Crop mặt. Nếu YuNet có landmark thì align về template ArcFace 112x112."""
        if align and face.get("landmarks") is not None:
            aligned = self.align_crop(frame, face["landmarks"], size=size)
            if aligned is not None:
                return aligned

        x1, y1, x2, y2 = self._clip_box(frame, face["box"])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return cv2.resize(face_crop, (size, size))

    @staticmethod
    def align_crop(frame, landmarks: np.ndarray, size: int = 112):
        # YuNet trả 5 điểm theo thứ tự:
        # right_eye, left_eye, nose, right_mouth, left_mouth.
        # Template dưới đây tương ứng vị trí 5 điểm trên ảnh 112x112.
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
        if size != 112:
            dst = dst * (size / 112.0)

        try:
            M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
            if M is None:
                return None
            return cv2.warpAffine(frame, M, (size, size), borderValue=0.0)
        except cv2.error:
            return None


def largest_face(faces: List[Dict]) -> Optional[Dict]:
    if not faces:
        return None
    return max(faces, key=lambda f: f["box"][2] * f["box"][3])


def face_ratio(face: Dict, frame) -> float:
    """Tỉ lệ kích thước mặt so với cạnh ngắn của frame; dùng để ước lượng gần/xa."""
    _, _, w, h = face["box"]
    return max(w, h) / max(1, min(frame.shape[:2]))


def blur_score(face_bgr) -> float:
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_distance(ratio: float, expected_range: Tuple[float, float]) -> Tuple[bool, str]:
    low, high = expected_range
    if ratio < low:
        return False, "TIEN LAI GAN CAMERA"
    if ratio > high:
        return False, "LUI RA XA CAMERA"
    return True, "OK"


def setup_camera(camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


# ── Các kịch bản đăng ký: thêm gần/vừa/xa + góc mặt ───────────────────────
# face_ratio dùng trên camera 1280x720. Nếu camera khác, vẫn ổn vì tính theo tỉ lệ.
SCENARIOS = [
    ("THANG_GAN", "Nhin thang - dua mat GAN camera", (0.26, 0.42), 3.0),
    ("THANG_VUA", "Nhin thang - khoang cach BINH THUONG", (0.16, 0.30), 3.0),
    ("THANG_XA", "Nhin thang - LUI RA XA camera", (0.14, 0.18), 3.0),
    ("TRAI_VUA", "Quay mat sang TRAI nhe, giu khoang cach vua", (0.13, 0.30), 3.0),
    ("PHAI_VUA", "Quay mat sang PHAI nhe, giu khoang cach vua", (0.13, 0.30), 3.0),
    ("LEN_VUA", "Ngua dau len nhe, giu khoang cach vua", (0.13, 0.30), 3.0),
    ("XUONG_VUA", "Cui dau xuong nhe, giu khoang cach vua", (0.13, 0.30), 3.0),
    ("ANH_SANG_KHAC", "Doi anh sang/chech mat nhe de he thong on dinh hon", (0.12, 0.32), 3.0),
]


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def extract_embedding(model, face_crop, device):
    """Trích xuất vector đặc trưng từ khuôn mặt đã cắt bằng ONNX"""
    # 1. Chuyển đổi không gian màu từ BGR sang RGB
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    
    # 2. Tiền xử lý ảnh (Hàm transform này đã trả về MockTensor giả lập)
    tensor = transform(face_rgb).unsqueeze(0)
    
    # 3. Đưa qua model ONNX để lấy embedding (MockTensor xử lý hết các hàm .cpu().numpy())
    emb = model(tensor).cpu().numpy()[0]
    
    return emb


def draw_ui(frame, scenario_name, instruction, countdown, collected, total, msg, ratio, blur):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 92), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"ENROLLMENT [{collected}/{total}]", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 150), 2)
    cv2.putText(frame, scenario_name, (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    cv2.putText(frame, f"ratio={ratio:.2f}  blur={blur:.0f}", (15, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1)

    cv2.putText(frame, instruction, (15, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, msg, (15, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255) if msg == "OK" else (0, 120, 255), 2)

    if countdown > 0:
        cv2.putText(frame, f"{countdown:.1f}s", (w - 115, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    bar_w = int(w * collected / max(1, total))
    cv2.rectangle(frame, (0, h - 8), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 200, 100), -1)
    return frame


def load_model(model_path: str, device):
    model = FaceEmbedder(embedding_dim=512, backbone="resnet50").to(device)
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights: {model_path}")
        print(f"[INFO] Best epoch: {checkpoint.get('best_epoch') if isinstance(checkpoint, dict) else None}, "
              f"AUC: {checkpoint.get('auc') if isinstance(checkpoint, dict) else None}")
    else:
        print("[WARN] Không tìm thấy weights → dùng random weights, chỉ để test UI.")
    model.eval()
    return model


def enroll(
    name: str,
    model_path: str,
    yunet_path: str,
    camera_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    target_frames: int = 18,
    min_blur: float = 35.0,
):
    # THAY BẰNG:
    device = "cpu"
    print(f"[INFO] Device: {device}")

    model = load_model(model_path, device)
    detector = FaceDetector(yunet_path)
    db = load_db()

    cap = setup_camera(camera_id, width, height, fps)
    if not cap.isOpened():
        print("[ERROR] Không mở được webcam!")
        return

    all_embeddings = []
    scenario_meta = []

    print(f"\n[START] Đăng ký khuôn mặt cho: {name}")
    print("Nhấn SPACE để bắt đầu từng cảnh; ESC để thoát.\n")

    for idx, (scene_name, instruction, expected_range, duration) in enumerate(SCENARIOS):
        print(f"  → {idx + 1}/{len(SCENARIOS)}: {scene_name} — {instruction}")
        embeddings_this_scene = []
        start_time = None
        collecting = False
        collected_frames = 0
        last_msg = "DANG TIM KHUON MAT"
        last_ratio = 0.0
        last_blur = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)
            face = largest_face(faces)
            has_good_face = False
            face_crop = None
            countdown = 0.0

            if face is not None:
                x, y, fw, fh = face["box"]
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 220, 255), 2)

                last_ratio = face_ratio(face, frame)
                ok_dist, dist_msg = check_distance(last_ratio, expected_range)
                face_crop = detector.crop(frame, face, align=True)
                last_blur = blur_score(face_crop)
                ok_blur = last_blur >= min_blur

                if not ok_dist:
                    last_msg = dist_msg
                elif not ok_blur:
                    last_msg = "ANH BI MO - GIU YEN / TANG SANG"
                else:
                    last_msg = "OK"
                    has_good_face = True
            else:
                last_msg = "KHONG TIM THAY KHUON MAT"

            if collecting and start_time:
                elapsed = time.time() - start_time
                countdown = max(0.0, duration - elapsed)

                if has_good_face and elapsed < duration:
                    emb = extract_embedding(model, face_crop, device)
                    embeddings_this_scene.append(emb)
                    collected_frames += 1

                if elapsed >= duration or collected_frames >= target_frames:
                    print(f"     ✓ Thu thập {len(embeddings_this_scene)} embeddings")
                    break

            frame = draw_ui(
                frame,
                scene_name,
                instruction,
                countdown,
                idx,
                len(SCENARIOS),
                last_msg if collecting or face is not None else "Nhan SPACE khi da dung vi tri",
                last_ratio,
                last_blur,
            )

            if not collecting:
                prompt = "Nhan SPACE de bat dau" if has_good_face else "Can chinh theo huong dan truoc khi chup"
                cv2.putText(frame, prompt, (15, 128),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 150) if has_good_face else (0, 140, 255), 2)

            cv2.imshow("FaceLog — Enrollment YuNet", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == 32 and has_good_face and not collecting:
                collecting = True
                start_time = time.time()
                print("     [REC] Đang thu thập...")

        if embeddings_this_scene:
            arr = np.array(embeddings_this_scene)
            # Tính centroid, lấy top 60% frame gần centroid nhất
            centroid = arr.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            sims = arr @ centroid
            top_k = max(3, int(len(arr) * 0.6))
            top_indices = np.argsort(sims)[-top_k:]
            for i in top_indices:
                all_embeddings.append(arr[i].tolist())
            scenario_meta.append(
                {
                    "name": scene_name,
                    "instruction": instruction,
                    "num_frames": len(embeddings_this_scene),
                    "expected_face_ratio": list(expected_range),
                }
            )

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
        "detector": detector.name,
        "camera": {"width": width, "height": height, "fps": fps},
    }
    save_db(db)

    print(f"\n✅ Đã đăng ký thành công: {name}")
    print(f"   → {len(all_embeddings)} embeddings theo nhiều khoảng cách/góc nhìn")
    print(f"   → Lưu tại {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Tên người đăng ký")
    parser.add_argument("--model", default="E:/THANHTRA/KI_8/AI/facelog/models/arcface_vggface2.pth")
    parser.add_argument("--yunet", default="face_detection_yunet_2023mar.onnx")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--target-frames", type=int, default=18)
    parser.add_argument("--min-blur", type=float, default=35.0)
    args = parser.parse_args()

    enroll(
        name=args.name,
        model_path=args.model,
        yunet_path=args.yunet,
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        target_frames=args.target_frames,
        min_blur=args.min_blur,
    )
