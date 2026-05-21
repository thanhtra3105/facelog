## Cai thu vien
`bash 
cd ~/thanhtra/facelog/mobilenet

deactivate 2>/dev/null

python3 -m venv --system-site-packages venv313
source venv313/bin/activate

python --version

pip install --upgrade pip setuptools wheel
pip install numpy onnxruntime gpiozero lgpio
`
* Nếu thiếu OpenCV hoặc Picamera2:
`bash 
sudo apt install -y python3-opencv python3-picamera2 python3-lgpio
`
* Cai tflite
pip install ai-edge-litert
