import cv2
import torch
import numpy as np
import time
import os
import urllib.request
from datetime import datetime

# Tái sử dụng logic cũ từ dự án Logface của bạn
from enrollment import FaceEmbedder, extract_embedding, save_db, load_db, DB_PATH
from verification import identify

# Định nghĩa 7 bước tương tác động (Hướng mặt + Khoảng cách)
NEW_DIRECTIONS = [
    ("THẲNG",     "Nhìn thẳng vào camera ở khoảng cách bình thường", "straight"),
    ("VÀO GẦN",   "Di chuyển khuôn mặt lại GẦN camera hơn",          "close"),
    ("RA XA",     "Di chuyển khuôn mặt ra XA camera hơn",            "far"),
    ("QUAY TRÁI", "Quay mặt sang TRÁI một góc nhẹ (~30°)",           "left"),
    ("QUAY PHẢI", "Quay mặt sang PHẢI một góc nhẹ (~30°)",          "right"),
    ("NGẨNG LÊN", "Ngẩng đầu lên phía TRÊN nhẹ",                     "up"),
    ("CÚI XUỐNG", "Cúi đầu xuống phía DƯỚI nhẹ",                     "down")
]

YUNET_MODEL_FILE = "face_detection_yunet_2023mar.onnx"

class Pi5LogfaceEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        print("[INFO] Khởi tạo hệ thống Logface Thuần OpenCV (YuNet)...")
        
        # 1. Nạp Model ResNet-50 PyTorch
        self.model = FaceEmbedder(embedding_dim=512, backbone='resnet50').to(self.device)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("[INFO] Đã nạp thành công weights ResNet-50.")
        self.model.eval()

        # 2. Tự động tải file model YuNet từ OpenCV Zoo nếu chưa có
        if not os.path.exists(YUNET_MODEL_FILE):
            print("[INFO] Không tìm thấy YuNet model. Tiến hành tải tự động từ OpenCV Zoo...")
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            urllib.request.urlretrieve(url, YUNET_MODEL_FILE)
            print("[INFO] Tải thành công YuNet ONNX model.")

        # 3. Khởi tạo bộ phát hiện khuôn mặt YuNet tích hợp sẵn của OpenCV
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_FILE,
            config="",
            input_size=(640, 480),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=1
        )
        self.db = load_db()

    def estimate_pose_and_distance(self, face_data, frame_w, frame_h):
        """
        Thuật toán hình học dựa trên 5 điểm mốc của YuNet:
        Mắt phải (4,5), Mắt trái (6,7), Mũi (8,9), Mép miệng phải (10,11), Mép miệng trái (12,14)
        """
        # Trích xuất tọa độ các điểm mốc
        eye_right = np.array([face_data[4], face_data[5]])
        eye_left = np.array([face_data[6], face_data[7]])
        nose = np.array([face_data[8], face_data[9]])
        mouth_right = np.array([face_data[10], face_data[11]])
        mouth_left = np.array([face_data[12], face_data[13]])

        # 1. Tính toán Yaw (Xoay Ngang): Khoảng cách từ mũi tới 2 mắt
        d_le = np.linalg.norm(nose - eye_left)
        d_re = np.linalg.norm(nose - eye_right)
        yaw_ratio = d_le / (d_re + 1e-6)

        # 2. Tính toán Pitch (Gật Đầu): Vị trí tương đối của mũi giữa mắt và miệng
        mid_eyes_y = (eye_left[1] + eye_right[1]) / 2.0
        mid_mouth_y = (mouth_left[1] + mouth_right[1]) / 2.0
        d_up = nose[1] - mid_eyes_y
        d_down = mid_mouth_y - nose[1]
        pitch_ratio = d_up / (d_down + 1e-6)

        # 3. Tính toán Distance (Khoảng cách): Tỉ lệ chiều rộng bounding box so với frame
        face_w = face_data[2]
        distance_ratio = face_w / frame_w

        # Phân loại hướng mặt dựa trên ngưỡng tỉ lệ hình học
        pose = "straight"
        if yaw_ratio < 0.60:       pose = "left"    # Quay sang trái thì mũi lệch gần mắt trái (trên cam)
        elif yaw_ratio > 1.60:     pose = "right"
        elif pitch_ratio < 0.55:   pose = "up"      # Ngẩng lên thì mũi dịch gần sát đường mắt
        elif pitch_ratio > 1.55:   pose = "down"

        # Phân loại khoảng cách
        dist_status = "normal"
        if distance_ratio > 0.45:     dist_status = "close"  # Mặt chiếm hơn 45% màn hình -> Quá gần
        elif distance_ratio < 0.20:   dist_status = "far"    # Mặt chiếm dưới 20% màn hình -> Quá xa

        return pose, dist_status, (int(nose[0]), int(nose[1]))

    def process_enrollment(self, target_name):
        cap = cv2.VideoCapture(0)
        # Thiết lập kích thước input đồng bộ với detector
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.detector.setInputSize((640, 480))

        print(f"\n[GIAI ĐOẠN 1] KÍCH HOẠT PRE-CHECK CHỐNG TRÙNG CHO: {target_name}...")
        pre_check_frames = 0
        is_duplicate = False
        matched_name = ""

        while pre_check_frames < 5:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            face_crop = cv2.resize(frame, (112, 112))
            emb = extract_embedding(self.model, face_crop, self.device)
            name, score = identify(emb, self.db)
            
            if name != "UNKNOWN":
                is_duplicate = True
                matched_name = name
                break
            pre_check_frames += 1

        if is_duplicate:
            print(f"❌ [TỪ CHỐI ĐĂNG KÝ] Khuôn mặt đã tồn tại dưới tên: '{matched_name}'!")
            cap.release()
            return "DUPLICATE"

        print("=> [OK] Xác nhận khuôn mặt mới. Bắt đầu thu thập dữ liệu 7 bước.")

        all_embeddings = []
        
        for idx, (dir_name, instruction, required_state) in enumerate(NEW_DIRECTIONS):
            print(f"\n[👉 YÊU CẦU BƯỚC {idx+1}/7]: {dir_name} — {instruction}")
            embeddings_this_step = []
            TARGET_FRAMES = 15

            while len(embeddings_this_step) < TARGET_FRAMES:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Chạy phát hiện khuôn mặt bằng YuNet
                _, faces = self.detector.detect(frame)

                if faces is not None:
                    face = faces[0] # Lấy khuôn mặt đầu tiên phát hiện được
                    pose, dist_status, nose_pos = self.estimate_pose_and_distance(face, w, h)

                    # Kiểm tra logic khớp điều kiện kịch bản hướng/khoảng cách
                    is_matched = False
                    if required_state in ["straight", "left", "right", "up", "down"] and pose == required_state:
                        if dist_status != "far": is_matched = True
                    elif required_state == "close" and dist_status == "close":
                        is_matched = True
                    elif required_state == "far" and dist_status == "far":
                        is_matched = True

                    # Vẽ bounding box khuôn mặt
                    x, y, face_w, face_h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), (255, 176, 0), 2)

                    # Vẽ UI chỉ dẫn trạng thái động
                    color = (0, 255, 0) if is_matched else (0, 0, 255)
                    cv2.circle(frame, nose_pos, 6, color, -1)
                    cv2.putText(frame, f"Yeu cau: {dir_name} | Cam bat: P:{pose}, D:{dist_status}", 
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(frame, f"Tien do: {len(embeddings_this_step)}/{TARGET_FRAMES}", 
                                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                    if is_matched:
                        face_crop = cv2.resize(frame[max(0, y):y+face_h, max(0, x):x+face_w], (112, 112))
                        if face_crop.size > 0:
                            emb = extract_embedding(self.model, face_crop, self.device)
                            embeddings_this_step.append(emb)

                cv2.imshow("Logface Smart Engine — Pure OpenCV YuNet", frame)
                if cv2.waitKey(1) & 0xFF == 27: # Ấn ESC để hủy bỏ
                    cap.release()
                    cv2.destroyAllWindows()
                    return "CANCELLED"

            mean_emb = np.mean(embeddings_this_step, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            all_embeddings.append(mean_emb.tolist())
            print(f"   ✓ Hoàn thành bước: {dir_name}")

        # GIAI ĐOẠN 3: LƯU TRỮ CƠ SỞ DỮ LIỆU SẠCH
        self.db[target_name] = {
            "embeddings": all_embeddings,
            "enrolled_at": datetime.now().isoformat(),
            "num_directions": len(all_embeddings),
        }
        save_db(self.db)
        print(f"\n✅ ĐĂNG KÝ THÀNH CÔNG DỮ LIỆU ĐỘNG CỦA '{target_name}'.")
        
        cap.release()
        cv2.destroyAllWindows()
        return "SUCCESS"

if __name__ == "__main__":
    MODEL_PATH = "E:/HK8/TTNT/QuangDaAI/facelog/models/arcface_vggface2.pth"
    engine = Pi5LogfaceEngine(MODEL_PATH)
    engine.process_enrollment("Nguyen Van A")