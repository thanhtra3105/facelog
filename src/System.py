import sys
import os
import cv2
import torch
import numpy as np
import time
from datetime import datetime
from collections import deque, Counter

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QStackedWidget, QProgressBar, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

# Re-use your existing logic
from enrollment import FaceEmbedder, FaceDetector, transform, DB_PATH, load_db, save_db, extract_embedding, DIRECTIONS
from verification import identify, THRESHOLD_OPEN, THRESHOLD_UNSURE, SMOOTH_WINDOW

# 😎 Modern Dark-Mode Stylesheet (Cyberpunk Accent)
MODERN_STYLE = """
    QWidget {
        background-color: #121214;
        color: #E2E8F0;
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    QFrame#Sidebar {
        background-color: #1A1A1E;
        border-right: 1px solid #2D2D34;
    }
    QPushButton {
        background-color: #2D2D34;
        border: none;
        color: #94A3B8;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: bold;
        text-align: left;
    }
    QPushButton:hover {
        background-color: #3E3E4A;
        color: #F8FAFC;
    }
    QPushButton:checked {
        background-color: #00E676;
        color: #0A0A0C;
    }
    QPushButton#ActionBtn {
        background-color: #00B0FF;
        color: #0A0A0C;
        text-align: center;
    }
    QPushButton#ActionBtn:hover {
        background-color: #40C4FF;
    }
    QLineEdit {
        background-color: #1A1A1E;
        border: 2px solid #2D2D34;
        border-radius: 8px;
        padding: 10px;
        color: #F8FAFC;
        font-size: 14px;
    }
    QLineEdit:focus {
        border: 2px solid #00B0FF;
    }
    QProgressBar {
        border: 1px solid #2D2D34;
        border-radius: 4px;
        text-align: center;
        background-color: #1A1A1E;
    }
    QProgressBar::chunk {
        background-color: #00E676;
        border-radius: 4px;
    }
"""

class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray, dict)

    def __init__(self, mode="verify", target_name=None, model_path=None):
        super().__init__()
        self.mode = mode  # "verify" hoặc "enroll"
        self.target_name = target_name
        self.model_path = model_path
        self.running = True
        
        # Cấu hình AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceEmbedder(embedding_dim=512, backbone='resnet50').to(self.device)
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        self.detector = FaceDetector()
        self.db = load_db()

        # State cho Verification
        self.score_history = deque(maxlen=SMOOTH_WINDOW)
        self.name_history = deque(maxlen=SMOOTH_WINDOW)
        
        # State cho Enrollment
        self.current_dir_idx = 0
        self.collecting = False
        self.start_time = None
        self.embeddings_this_dir = []
        self.all_embeddings = []
        self.target_frames = 15

    def run(self):
        cap = cv2.VideoCapture(0)
        fps = 0
        last_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            # Tính toán FPS mượt
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-5))
            last_time = now

            faces = self.detector.detect(frame)
            info = {"fps": fps, "faces": faces, "status": "WAITING", "name": "UNKNOWN", "score": 0.0}

            if self.mode == "verify":
                self.process_verification(frame, faces, info)
            elif self.mode == "enroll" and self.collecting:
                self.process_enrollment(frame, faces, info, now)

            self.frame_signal.emit(frame, info)
            time.sleep(0.01)

        cap.release()

    def process_verification(self, frame, faces, info):
        if len(faces) > 0:
            biggest = max(faces, key=lambda f: f[2]*f[3])
            face_crop = self.detector.crop(frame, biggest)
            emb = extract_embedding(self.model, face_crop, self.device)
            
            name, score = identify(emb, self.db)
            self.score_history.append(score)
            self.name_history.append(name)
            
            smooth_score = np.mean(self.score_history)
            most_common_name = Counter(self.name_history).most_common(1)[0][0]
            
            if smooth_score >= THRESHOLD_OPEN and most_common_name != "UNKNOWN":
                info["status"] = "OPEN"
            elif smooth_score >= THRESHOLD_UNSURE:
                info["status"] = "UNSURE"
            else:
                info["status"] = "DENIED"
                
            info["name"] = most_common_name
            info["score"] = smooth_score
        else:
            self.score_history.clear()
            self.name_history.clear()

    def process_enrollment(self, frame, faces, info, now):
        dir_name, instruction, duration = DIRECTIONS[self.current_dir_idx]
        elapsed = now - self.start_time
        countdown = max(0, duration - elapsed)
        
        info.update({
            "mode": "enroll", "dir_name": dir_name, "instruction": instruction,
            "countdown": countdown, "progress": len(self.embeddings_this_dir)
        })

        if len(faces) > 0 and elapsed < duration:
            face_crop = self.detector.crop(frame, faces[0])
            emb = extract_embedding(self.model, face_crop, self.device)
            self.embeddings_this_dir.append(emb)

        if elapsed >= duration or len(self.embeddings_this_dir) >= self.target_frames:
            if self.embeddings_this_dir:
                mean_emb = np.mean(self.embeddings_this_dir, axis=0)
                mean_emb = mean_emb / np.linalg.norm(mean_emb)
                self.all_embeddings.append(mean_emb.tolist())
            
            self.embeddings_this_dir = []
            self.current_dir_idx += 1
            self.start_time = time.time()

            if self.current_dir_idx >= len(DIRECTIONS):
                self.collecting = False
                # Lưu Database
                self.db[self.target_name] = {
                    "embeddings": self.all_embeddings,
                    "enrolled_at": datetime.now().isoformat(),
                    "num_directions": len(self.all_embeddings),
                }
                save_db(self.db)
                info["status"] = "ENROLL_DONE"

    def start_direction_capture(self):
        self.start_time = time.time()
        self.collecting = True

    def stop(self):
        self.running = False
        self.wait()


class LogfaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Logface — Advanced Security Dashboard")
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(MODERN_STYLE)
        
        # Mặc định load model từ đường dẫn của bạn
        self.model_path = "E:/HK8/TTNT/QuangDaAI/facelog/models/arcface_vggface2.pth"
        
        self.init_ui()
        self.start_verification_mode()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── SIDEBAR ────────────────────────────────────────────────────────
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 30, 15, 30)
        sidebar_layout.setSpacing(10)

        logo_label = QLabel("LOGFACE AIoT")
        logo_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        logo_label.setStyleSheet("color: #00B0FF; margin-bottom: 20px;")
        sidebar_layout.addWidget(logo_label)

        self.btn_verify = QPushButton(" 🔓  Xác Thực Real-time")
        self.btn_verify.setCheckable(True)
        self.btn_verify.setChecked(True)
        self.btn_verify.clicked.connect(self.start_verification_mode)
        sidebar_layout.addWidget(self.btn_verify)

        self.btn_enroll = QPushButton(" 👤  Đăng Ký Khuôn Mặt")
        self.btn_enroll.setCheckable(True)
        self.btn_enroll.clicked.connect(self.start_enrollment_mode)
        sidebar_layout.addWidget(self.btn_enroll)

        sidebar_layout.addStretch()
        
        self.status_db_label = QLabel("Hệ thống: Sẵn sàng")
        self.status_db_label.setStyleSheet("color: #64748B; font-size: 12px;")
        sidebar_layout.addWidget(self.status_db_label)
        main_layout.addWidget(sidebar)

        # ── MAIN CONTENT (Stacked Widget) ──────────────────────────────────
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Page 1: Verification
        self.page_verify = QWidget()
        v_layout = QVBoxLayout(self.page_verify)
        self.camera_view_v = QLabel("Đang kết nối Camera...")
        self.camera_view_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view_v.setStyleSheet("background-color: #1A1A1E; border-radius: 12px;")
        v_layout.addWidget(self.camera_view_v, stretch=4)
        
        # Bottom Status Panel for Verification
        self.panel_v = QFrame()
        self.panel_v.setStyleSheet("background-color: #1A1A1E; border-radius: 12px; padding: 15px;")
        pv_layout = QHBoxLayout(self.panel_v)
        
        self.lbl_v_name = QLabel("ĐANG QUÉT...")
        self.lbl_v_name.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self.lbl_v_status = QPushButton("WAITING")
        self.lbl_v_status.setDisabled(True)
        self.lbl_v_status.setStyleSheet("background-color: #3E3E4A; color: white; border-radius: 6px; font-size: 14px;")
        
        pv_layout.addWidget(self.lbl_v_name)
        pv_layout.addStretch()
        pv_layout.addWidget(self.lbl_v_status)
        v_layout.addWidget(self.panel_v, stretch=1)
        
        self.stacked_widget.addWidget(self.page_verify)

        # Page 2: Enrollment
        self.page_enroll = QWidget()
        e_layout = QVBoxLayout(self.page_enroll)
        
        # Form input tên
        form_layout = QHBoxLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Nhập tên người đăng ký...")
        self.btn_start_enroll = QPushButton("Bắt Đầu Đăng Ký")
        self.btn_start_enroll.setObjectName("ActionBtn")
        self.btn_start_enroll.clicked.connect(self.trigger_enroll_process)
        form_layout.addWidget(self.txt_name, stretch=3)
        form_layout.addWidget(self.btn_start_enroll, stretch=1)
        e_layout.addLayout(form_layout)

        self.camera_view_e = QLabel("Nhập tên và ấn Bắt Đầu")
        self.camera_view_e.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view_e.setStyleSheet("background-color: #1A1A1E; border-radius: 12px;")
        e_layout.addWidget(self.camera_view_e, stretch=4)

        # UI chỉ dẫn hướng chụp khuôn mặt
        self.panel_e = QFrame()
        self.panel_e.setStyleSheet("background-color: #1A1A1E; border-radius: 12px; padding: 15px;")
        pe_layout = QVBoxLayout(self.panel_e)
        
        self.lbl_direction = QLabel("Hướng dẫn:")
        self.lbl_direction.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.lbl_instruction = QLabel("Hệ thống sẽ lấy dữ liệu từ 5 hướng để tối ưu độ chính xác.")
        self.lbl_instruction.setStyleSheet("color: #94A3B8;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(15)
        
        pe_layout.addWidget(self.lbl_direction)
        pe_layout.addWidget(self.lbl_instruction)
        pe_layout.addWidget(self.progress_bar)
        e_layout.addWidget(self.panel_e, stretch=1)

        self.stacked_widget.addWidget(self.page_enroll)

    def stop_current_thread(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()

    def start_verification_mode(self):
        self.btn_verify.setChecked(True)
        self.btn_enroll.setChecked(False)
        self.stacked_widget.setCurrentIndex(0)
        self.stop_current_thread()

        self.video_thread = VideoThread(mode="verify", model_path=self.model_path)
        self.video_thread.frame_signal.connect(self.update_verification_ui)
        self.video_thread.start()

    def start_enrollment_mode(self):
        self.btn_verify.setChecked(False)
        self.btn_enroll.setChecked(True)
        self.stacked_widget.setCurrentIndex(1)
        self.stop_current_thread()
        
        self.camera_view_e.setPixmap(QPixmap())
        self.camera_view_e.setText("Nhập tên và ấn Bắt đầu đăng ký")

    def trigger_enroll_process(self):
        name = self.txt_name.text().strip()
        if not name:
            self.lbl_direction.setText("⚠️ Lỗi: Vui lòng nhập tên!")
            return
        
        self.stop_current_thread()
        self.video_thread = VideoThread(mode="enroll", target_name=name, model_path=self.model_path)
        self.video_thread.frame_signal.connect(self.update_enrollment_ui)
        self.video_thread.start()
        self.video_thread.start_direction_capture()

    @pyqtSlot(np.ndarray, dict)
    def update_verification_ui(self, frame, info):
        # Vẽ bounding box lên frame trực tiếp bằng OpenCV trước khi đưa lên GUI
        for (x, y, w, h) in info["faces"]:
            color = (0, 230, 100) if info["status"] == "OPEN" else (0, 176, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
        # Convert BGR sang RGB và render ra QLabel
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.camera_view_v.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_view_v.width(), self.camera_view_v.height(), Qt.AspectRatioMode.KeepAspectRatio))

        # Cập nhật thông tin thẻ Trạng thái phía dưới
        status = info["status"]
        if status == "OPEN":
            self.lbl_v_name.setText(f"🔓 WELCOME, {info['name']}!")
            self.lbl_v_name.setStyleSheet("color: #00E676;")
            self.lbl_v_status.setText(f"MỞ CỬA ({info['score']:.2f})")
            self.lbl_v_status.setStyleSheet("background-color: #00E676; color: #0A0A0C; font-weight: bold; padding: 5px 15px;")
        elif status == "UNSURE":
            self.lbl_v_name.setText(f"🔍 {info['name']}?")
            self.lbl_v_name.setStyleSheet("color: #FFB300;")
            self.lbl_v_status.setText(f"CHECKING ({info['score']:.2f})")
            self.lbl_v_status.setStyleSheet("background-color: #FFB300; color: #0A0A0C; font-weight: bold; padding: 5px 15px;")
        elif status == "DENIED":
            self.lbl_v_name.setText("❌ UNKNOWN FACE")
            self.lbl_v_name.setStyleSheet("color: #FF1744;")
            self.lbl_v_status.setText("TỪ CHỐI")
            self.lbl_v_status.setStyleSheet("background-color: #FF1744; color: white; font-weight: bold; padding: 5px 15px;")
        else:
            self.lbl_v_name.setText("ĐANG ĐỢI QUÉT...")
            self.lbl_v_name.setStyleSheet("color: #94A3B8;")
            self.lbl_v_status.setText("SCANNING")
            self.lbl_v_status.setStyleSheet("background-color: #2D2D34; color: #94A3B8; padding: 5px 15px;")

    @pyqtSlot(np.ndarray, dict)
    def update_enrollment_ui(self, frame, info):
        if info.get("status") == "ENROLL_DONE":
            self.lbl_direction.setText("✅ ĐĂNG KÝ THÀNH CÔNG!")
            self.lbl_instruction.setText(f"Dữ liệu của {self.txt_name.text()} đã được cập nhật vào face_db.json.")
            self.progress_bar.setValue(15)
            self.stop_current_thread()
            return

        for (x, y, w, h) in info["faces"]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 176, 255), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.camera_view_e.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.camera_view_e.width(), self.camera_view_e.height(), Qt.AspectRatioMode.KeepAspectRatio))

        if "dir_name" in info:
            self.lbl_direction.setText(f"Hướng {self.video_thread.current_dir_idx + 1}/5: Nhìn qua hướng [{info['dir_name']}]")
            self.lbl_instruction.setText(f"Yêu cầu: {info['instruction']} (Còn lại: {info['countdown']:.1f}s)")
            self.progress_bar.setValue(info["progress"])

    def closeEvent(self, event):
        self.stop_current_thread()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LogfaceApp()
    window.show()
    sys.exit(app.exec())