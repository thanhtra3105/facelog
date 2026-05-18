import torch
import os
import sys

# Đưa thư mục src vào path để import được FaceEmbedder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from enrollment import FaceEmbedder

def export_onnx(pth_path, onnx_path):
    print("[INFO] Đang nạp model PyTorch...")
    device = torch.device("cpu") # Chuyển đổi trên CPU cho an toàn và tương thích
    
    # 1. Khởi tạo kiến trúc model giống hệt trong verification.py
    model = FaceEmbedder(embedding_dim=512, backbone='resnet50').to(device)
    
    # 2. Nạp trọng số (weights) từ file .pth
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval() # BẮT BUỘC: chuyển sang chế độ evaluation trước khi export
    print(f"[INFO] Đã nạp thành công trọng số từ: {pth_path}")

    # 3. Tạo một tensor giả (dummy input) có kích thước giống hệt ảnh đầu vào
    # ArcFace thường nhận ảnh RGB kích thước 112x112
    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    # 4. Thực hiện xuất ra file ONNX
    print("[INFO] Đang tiến hành convert sang ONNX...")
    torch.onnx.export(
        model,                      # Model cần convert
        dummy_input,                # Input mẫu để PyTorch trace các phép toán
        onnx_path,                  # Đường dẫn file đầu ra
        export_params=True,         # Lưu trữ trọng số (weights) vào file
        opset_version=11,           # Chuẩn ONNX (opset 11 rất ổn định cho ONNXRuntime)
        do_constant_folding=True,   # Tối ưu hóa (gộp các hằng số để chạy nhanh hơn)
        input_names=['input'],      # Đặt tên cho cổng vào
        output_names=['embedding'], # Đặt tên cho cổng ra
        dynamic_axes={              # (Tùy chọn) Cho phép model nhận nhiều ảnh cùng lúc
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    print(f"✅ HOÀN TẤT! File ONNX đã được lưu tại: {onnx_path}")

if __name__ == "__main__":
    # Đường dẫn file .pth hiện tại của bạn
    model_pth = "models/arcface_vggface2.pth" 
    # Đường dẫn file .onnx xuất ra
    model_onnx = "models/arcface_vggface2.onnx"
    
    export_onnx(model_pth, model_onnx)