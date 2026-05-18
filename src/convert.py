import torch
import torchvision.models as models
import torch.nn as nn

# 1. BẠN CẦN IMPORT CLASS MÔ HÌNH ARCFACE CỦA BẠN VÀO ĐÂY
# Ví dụ: from models.resnet import iResNet (tùy thuộc vào source code bạn dùng)
# model = iResNet(...) 
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

# Ở đây mình ví dụ bạn đã khởi tạo biến model thành công:
model = FaceEmbedder(embedding_dim=512)

# 2. Load trọng số
model_path = "E:/HK8/TTNT/QuangDaAI/facelog/models/arcface_vggface2.pth"

#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# 1. Load toàn bộ file checkpoint vào một biến
checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# 2. Chỉ trích xuất phần trọng số (model_state_dict) để nạp vào mô hình
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Dummy input cho ArcFace (Thường là ảnh RGB 112x112)
dummy_input = torch.randn(1, 3, 112, 112, device='cpu')

# 4. Xuất ra file ONNX
onnx_file_path = "arcface_vggface2.onnx"
torch.onnx.export(
    model,                      
    dummy_input,                
    onnx_file_path,             
    export_params=True,         
    opset_version=11,           
    do_constant_folding=True,   
    input_names=['input'],      
    output_names=['embedding'], # ArcFace trả về vector đặc trưng
    dynamic_axes={'input': {0: 'batch_size'},    
                  'embedding': {0: 'batch_size'}}
)

print(f"Xuất file ONNX thành công: {onnx_file_path}")