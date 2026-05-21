#!/usr/bin/env python3
"""
register_web_upload.py
Dang ky khuon mat qua web browser.
Ho tro 2 cach:
  1) Dang ky realtime bang Pi Camera / USB Camera
  2) Upload 1 anh co san -> YuNet detect/align -> augmentation -> MobileFaceNet -> face_database.json

Chay voi Pi Camera:
    python register_web_upload.py

Chay voi USB Camera:
    python register_web_upload.py --use-usb-camera --camera 9

Mo trinh duyet:
    http://<PI_IP>:5001

Can co cung thu muc:
    mobilefacenet.tflite

Can cai:
    pip install flask numpy
    pip install ai-edge-litert   # neu dung Python 3.13
    sudo apt install -y python3-opencv python3-picamera2
"""

import argparse
import json
import os
import threading
import time

# OpenCV FaceDetectorYN/DNN can crash if used from camera thread and upload route at the same time.
# This lock serializes all YuNet detect calls.
YUNET_DNN_LOCK = threading.Lock()
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

# TFLite / LiteRT fallback
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mobilefacenet.tflite"
DB_PATH = BASE_DIR / "face_database.json"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
IMG_SIZE = 112

SCENARIOS = [
    ("Straight", "Look straight at camera", (0.14, 0.35), 2.5),
    ("Turn left", "Turn face slightly LEFT", (0.12, 0.35), 2.5),
    ("Turn right", "Turn face slightly RIGHT", (0.12, 0.35), 2.5),
    ("Look up", "Tilt head UP slightly", (0.12, 0.35), 2.5),
    ("Look down", "Tilt head DOWN slightly", (0.12, 0.35), 2.5),
]

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Enrollment</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e8e8e8;font-family:Arial,Helvetica,sans-serif;display:flex;flex-direction:column;align-items:center;padding:16px;gap:12px}
h1{font-size:1.15rem;color:#f5f5f5;margin-bottom:4px}.sub{font-size:.82rem;color:#888;margin-bottom:6px}
.card{width:100%;max-width:720px;background:#171717;border:1px solid #2b2b2b;border-radius:10px;padding:14px}
.tabs{display:flex;gap:8px;width:100%;max-width:720px}.tab{flex:1;padding:10px;border:1px solid #333;border-radius:8px;background:#181818;color:#aaa;cursor:pointer;font-weight:bold}.tab.active{background:#2563eb;color:white;border-color:#2563eb}
.panel{display:none}.panel.active{display:block}
#wrap{position:relative;width:100%;max-width:640px;margin:auto}#stream{width:100%;border-radius:8px;display:block;background:#000}
#badge{position:absolute;top:10px;left:10px;padding:5px 10px;border-radius:6px;font-size:.82rem;font-weight:bold;background:#000a;color:#fff}.ok{background:#166534cc!important}.warn{background:#92400ecc!important}.err{background:#7f1d1dcc!important}
.row{display:flex;gap:8px;margin-top:10px}input[type=text]{flex:1;background:#222;border:1px solid #444;border-radius:8px;padding:10px;color:#eee;font-size:1rem}input[type=file]{width:100%;background:#222;border:1px solid #444;border-radius:8px;padding:10px;color:#ccc;margin-top:10px}
button{padding:10px 14px;border:0;border-radius:8px;background:#2563eb;color:white;cursor:pointer;font-weight:bold}button:disabled{background:#333;color:#777;cursor:not-allowed}.danger{background:#7f1d1d}.muted{background:#333}
#instruction{margin-top:10px;background:#101010;border-left:4px solid #2563eb;padding:10px;border-radius:6px;color:#d0d0d0;font-size:.92rem;min-height:42px}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.stat{background:#101010;border:1px solid #222;padding:7px 10px;border-radius:6px;font-size:.82rem;color:#aaa}.stat b{color:#eee}
#scenarios{display:flex;flex-direction:column;gap:6px;margin-top:10px}.sc{display:flex;align-items:center;gap:10px;background:#101010;border:1px solid #222;padding:8px 10px;border-radius:7px}.sc.active{border-color:#2563eb;background:#10203a}.sc.done{border-color:#16a34a;background:#0b2617}.dot{width:10px;height:10px;border-radius:50%;background:#555}.active .dot{background:#3b82f6}.done .dot{background:#22c55e}.sc span{flex:1;color:#ccc}.cnt{color:#888;font-size:.78rem}
#log{height:110px;overflow:auto;background:#101010;border:1px solid #222;border-radius:8px;padding:8px;font-family:monospace;font-size:.78rem;color:#888;margin-top:10px}.logok{color:#22c55e}.logerr{color:#f87171}
#upload-preview{width:100%;max-height:360px;object-fit:contain;background:#000;border-radius:8px;margin-top:10px;display:none}.result{margin-top:10px;padding:10px;background:#101010;border-radius:8px;color:#ccc;font-size:.9rem;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:.9rem}th,td{border-bottom:1px solid #292929;padding:8px;text-align:left}th{color:#888;font-weight:normal}
</style>
</head>
<body>
<h1>Face Enrollment Web</h1>
<div class="sub">YuNet detection + MobileFaceNet TFLite embedding + face_database.json</div>

<div class="tabs">
  <button class="tab active" onclick="showTab('cam')">Dang ky bang camera</button>
  <button class="tab" onclick="showTab('upload')">Dang ky bang anh upload</button>
  <button class="tab" onclick="showTab('db')">Database</button>
</div>

<div class="card panel active" id="panel-cam">
  <div id="wrap"><img id="stream" src="/video_feed"><div id="badge" class="warn">Idle</div></div>
  <div class="row"><input id="cam-name" type="text" placeholder="Nhap ten"><button id="btn-start" onclick="startCam()">Bat dau dang ky</button></div>
  <div id="instruction">Nhap ten va bam bat dau dang ky.</div>
  <div class="stats">
    <div class="stat">FPS: <b id="s-fps">-</b></div><div class="stat">Ratio: <b id="s-ratio">-</b></div><div class="stat">Blur: <b id="s-blur">-</b></div><div class="stat">Collected: <b id="s-collected">0</b></div>
  </div>
  <div id="scenarios"></div>
  <div id="log"></div>
</div>

<div class="card panel" id="panel-upload">
  <p style="color:#aaa;font-size:.9rem">Upload mot anh khuon mat. He thong se detect/align bang YuNet, tao anh tang cuong, trich xuat nhieu embeddings bang MobileFaceNet va luu vao database.</p>
  <div class="row"><input id="up-name" type="text" placeholder="Nhap ten"></div>
  <input id="up-file" type="file" accept="image/*" onchange="previewUpload()">
  <img id="upload-preview">
  <div class="row"><button onclick="uploadRegister()">Upload va dang ky</button><button class="muted" onclick="document.getElementById('up-file').value='';document.getElementById('upload-preview').style.display='none'">Xoa anh</button></div>
  <div class="result" id="upload-result">Chua upload anh.</div>
</div>

<div class="card panel" id="panel-db">
  <div class="row"><button onclick="loadDB()">Refresh database</button></div>
  <table><thead><tr><th>Name</th><th>Embeddings</th><th>Action</th></tr></thead><tbody id="db-body"></tbody></table>
</div>

<script>
const SCENARIOS = ["Straight","Turn left","Turn right","Look up","Look down"];
function showTab(id){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));event.target.classList.add('active');document.getElementById('panel-'+id).classList.add('active');if(id==='db')loadDB();}
function buildScenes(){let w=document.getElementById('scenarios');w.innerHTML='';SCENARIOS.forEach((s,i)=>{let d=document.createElement('div');d.className='sc';d.id='sc'+i;d.innerHTML=`<div class="dot"></div><span>${i+1}. ${s}</span><div class="cnt" id="cnt${i}">0 emb</div>`;w.appendChild(d);});} buildScenes();
function addLog(msg,cls=''){let log=document.getElementById('log');let d=document.createElement('div');d.className=cls;d.textContent=new Date().toLocaleTimeString()+'  '+msg;log.appendChild(d);log.scrollTop=log.scrollHeight;}
function startCam(){let name=document.getElementById('cam-name').value.trim();if(!name){alert('Nhap ten truoc');return;}fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})}).then(r=>r.json()).then(d=>{if(d.ok){document.getElementById('btn-start').disabled=true;addLog('[START] '+name);poll();}else alert(d.msg);});}
function poll(){fetch('/status').then(r=>r.json()).then(d=>{document.getElementById('s-fps').textContent=d.fps.toFixed(1);document.getElementById('s-ratio').textContent=d.ratio.toFixed(2);document.getElementById('s-blur').textContent=d.blur.toFixed(0);document.getElementById('s-collected').textContent=d.total_collected;document.getElementById('instruction').innerHTML=d.instruction||'';let b=document.getElementById('badge');b.textContent=d.msg;b.className=d.msg==='OK'?'ok':(d.phase==='done'?'ok':'warn');SCENARIOS.forEach((_,i)=>{let row=document.getElementById('sc'+i);row.className='sc';if(i===d.scene_idx&&d.phase==='enrolling')row.className+=' active';if((d.scene_counts[i]||0)>=d.target)row.className+=' done';document.getElementById('cnt'+i).textContent=(d.scene_counts[i]||0)+' emb';});if(d.last_log)addLog(d.last_log);if(d.phase==='done'){document.getElementById('btn-start').disabled=false;addLog('[DONE] '+d.name+' '+d.total_collected+' emb','logok');loadDB();return;}if(d.phase==='enrolling')setTimeout(poll,350);else document.getElementById('btn-start').disabled=false;});}
function previewUpload(){let f=document.getElementById('up-file').files[0];let img=document.getElementById('upload-preview');if(!f){img.style.display='none';return;}img.src=URL.createObjectURL(f);img.style.display='block';}
function uploadRegister(){let name=document.getElementById('up-name').value.trim();let f=document.getElementById('up-file').files[0];if(!name){alert('Nhap ten');return;}if(!f){alert('Chon anh');return;}let fd=new FormData();fd.append('name',name);fd.append('image',f);document.getElementById('upload-result').textContent='Dang xu ly...';fetch('/upload_register',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{document.getElementById('upload-result').textContent=JSON.stringify(d,null,2);if(d.ok)loadDB();}).catch(e=>{document.getElementById('upload-result').textContent='Loi: '+e;});}
function loadDB(){fetch('/db').then(r=>r.json()).then(db=>{let body=document.getElementById('db-body');body.innerHTML='';Object.entries(db).forEach(([name,c])=>{let tr=document.createElement('tr');tr.innerHTML=`<td>${name}</td><td>${c}</td><td><button class="danger" onclick="delPerson('${name}')">Delete</button></td>`;body.appendChild(tr);});});}
function delPerson(name){if(!confirm('Xoa '+name+'?'))return;fetch('/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})}).then(r=>r.json()).then(d=>{if(d.ok)loadDB();else alert(d.msg);});}
loadDB();
</script>
</body>
</html>"""


def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return
    print(f"[INFO] Download YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] YuNet size: {path.stat().st_size // 1024} KB")


class MobileFaceNet:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
        self.interp = Interpreter(model_path=str(MODEL_PATH))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        print(f"[INFO] MobileFaceNet ready: {MODEL_PATH}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 128.0
        self.interp.set_tensor(self.inp["index"], face[np.newaxis])
        self.interp.invoke()
        emb = self.interp.get_tensor(self.outp["index"])[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)


class YuNetDetector:
    def __init__(self, score_threshold: float = 0.82):
        ensure_yunet(YUNET_PATH)
        self.det = cv2.FaceDetectorYN_create(str(YUNET_PATH), "", (320, 320), score_threshold, 0.3, 5000)
        print(f"[INFO] YuNet ready: {YUNET_PATH}")

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        try:
            # FaceDetectorYN uses OpenCV DNN internally. On Raspberry Pi it can throw
            # releaseReference/mapIt errors when called concurrently or on unstable frames.
            # The lock prevents concurrent detect() calls from preview and upload threads.
            with YUNET_DNN_LOCK:
                self.det.setInputSize((w, h))
                _, faces = self.det.detect(frame)
        except cv2.error as e:
            print(f"[WARN] YuNet detect skipped bad frame: {e}")
            return []

        if faces is None:
            return []

        results = []
        for f in faces:
            # YuNet sometimes returns invalid values such as inf/nan on PiCamera
            # startup or bad frames. Skip those detections instead of crashing.
            if f is None or len(f) < 15:
                continue
            if not np.all(np.isfinite(f)):
                continue

            x, y, bw, bh = f[:4]
            score = float(f[-1])
            if bw <= 0 or bh <= 0 or score < 0:
                continue

            x = int(max(0, min(float(x), w - 1)))
            y = int(max(0, min(float(y), h - 1)))
            bw = int(max(1, min(float(bw), w - x)))
            bh = int(max(1, min(float(bh), h - y)))

            landmarks = f[4:14].reshape(5, 2).astype(np.float32)
            results.append({
                "box": (x, y, bw, bh),
                "landmarks": landmarks,
                "score": score,
            })
        return results

    @staticmethod
    def largest(faces):
        return max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None

    @staticmethod
    def face_ratio(face, frame):
        _, _, w, h = face["box"]
        return max(w, h) / max(1, min(frame.shape[:2]))

    @staticmethod
    def align_crop(frame: np.ndarray, landmarks: np.ndarray, size: int = 112):
        dst = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32,
        )
        try:
            M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
            if M is None:
                return None
            return cv2.warpAffine(frame, M, (size, size), borderValue=0.0)
        except cv2.error:
            return None

    @staticmethod
    def crop_fallback(frame, face, pad_ratio: float = 0.18):
        x, y, w, h = face["box"]
        pad = int(pad_ratio * max(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (112, 112)) if crop.size else None

    def crop(self, frame, face):
        crop = self.align_crop(frame, face["landmarks"])
        return crop if crop is not None else self.crop_fallback(frame, face)


def blur_score(img):
    if img is None or img.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def load_db():
    if not DB_PATH.exists():
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def shift_image(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img, sigma=4.0):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def augment_face(face_bgr):
    """Tao cac bien the nhe, khong lam bien dang khuon mat qua manh."""
    variants = []
    base = cv2.resize(face_bgr, (112, 112))
    variants.append(("original", base))
    variants.append(("bright_plus", adjust_brightness_contrast(base, 1.00, 18)))
    variants.append(("bright_minus", adjust_brightness_contrast(base, 1.00, -18)))
    variants.append(("contrast_plus", adjust_brightness_contrast(base, 1.12, 0)))
    variants.append(("contrast_minus", adjust_brightness_contrast(base, 0.88, 0)))
    variants.append(("rotate_left", rotate_image(base, -6)))
    variants.append(("rotate_right", rotate_image(base, 6)))
    variants.append(("shift_left", shift_image(base, -4, 0)))
    variants.append(("shift_right", shift_image(base, 4, 0)))
    variants.append(("shift_up", shift_image(base, 0, -4)))
    variants.append(("shift_down", shift_image(base, 0, 4)))
    variants.append(("blur_light", cv2.GaussianBlur(base, (3, 3), 0)))
    variants.append(("noise_light", add_noise(base, 3.5)))
    variants.append(("flip", cv2.flip(base, 1)))
    return variants


def enroll_from_image_bytes(name: str, image_bytes: bytes, detector: YuNetDetector, embedder: MobileFaceNet, append: bool = False):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"ok": False, "msg": "Khong doc duoc file anh"}

    # Try original image first. If no face is found, try common rotations.
    # This helps with phone photos because cv2.imdecode may ignore EXIF orientation.
    candidates = [
        ("original", frame),
        ("rot90", cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)),
        ("rot180", cv2.rotate(frame, cv2.ROTATE_180)),
        ("rot270", cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    face = None
    used_frame = None
    used_orientation = "original"
    for label, img in candidates:
        faces = detector.detect(img)
        face = detector.largest(faces)
        if face is not None:
            used_frame = img
            used_orientation = label
            break

    if face is None or used_frame is None:
        return {"ok": False, "msg": "Khong tim thay khuon mat trong anh. Hay dung anh ro mat, mat lon hon, nhin thang hon hoac giam nguong score."}

    crop = detector.crop(used_frame, face)
    if crop is None:
        return {"ok": False, "msg": "Khong crop/align duoc khuon mat"}

    bscore = blur_score(crop)
    variants = augment_face(crop)
    embeddings = []
    used = []
    for label, img in variants:
        emb = embedder.embed(img)
        embeddings.append(emb.tolist())
        used.append(label)

    db = load_db()
    old_count = len(db.get(name, []))
    if append and name in db:
        db[name].extend(embeddings)
    else:
        db[name] = embeddings
    save_db(db)

    x, y, w, h = face["box"]
    return {
        "ok": True,
        "name": name,
        "mode": "append" if append else "overwrite",
        "old_embeddings": old_count,
        "new_embeddings": len(embeddings),
        "total_embeddings": len(db[name]),
        "face_box": [x, y, w, h],
        "face_score": round(float(face["score"]), 4),
        "blur": round(float(bscore), 2),
        "augmentations": used,
        "db": str(DB_PATH),
    }


def open_camera(args):
    if not args.use_usb_camera:
        try:
            from picamera2 import Picamera2

            class PiCamWrap:
                def __init__(self):
                    self.picam2 = Picamera2()
                    cfg = self.picam2.create_preview_configuration(
                        main={"size": (args.width, args.height), "format": "RGB888"},
                        controls={"FrameRate": args.fps},
                    )
                    self.picam2.configure(cfg)
                    self.picam2.start()
                    time.sleep(0.5)
                    print(f"[INFO] Pi Camera {args.width}x{args.height}@{args.fps}")

                def isOpened(self):
                    return True

                def read(self):
                    rgb = self.picam2.capture_array()
                    if rgb is None:
                        return False, None
                    return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                def release(self):
                    self.picam2.stop()

            return PiCamWrap()
        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"[INFO] OpenCV camera id={args.camera}")
    return cap


_lock = threading.Lock()
_jpeg_frame = b""
_status = {
    "phase": "idle",
    "scene_idx": 0,
    "scene_counts": [0] * len(SCENARIOS),
    "total_collected": 0,
    "msg": "Idle",
    "instruction": "Nhap ten va bam bat dau dang ky.",
    "ratio": 0.0,
    "blur": 0.0,
    "fps": 0.0,
    "target": 12,
    "name": "",
    "last_log": None,
}
_cmd = {"action": None, "name": ""}
_detector = None
_embedder = None


def draw_register_ui(frame, msg, scenario_name, ratio, blur, scene_idx, total_scenes, bbox=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    color = (0, 220, 120) if msg == "OK" else (0, 160, 255)
    cv2.putText(frame, f"[{scene_idx + 1}/{total_scenes}] {scenario_name}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
    cv2.putText(frame, f"ratio={ratio:.2f} blur={blur:.0f} {msg}", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)


def worker_loop(args):
    global _jpeg_frame, _status, _cmd, _detector, _embedder

    cap = open_camera(args)
    _detector = YuNetDetector(score_threshold=args.score_threshold)
    _embedder = MobileFaceNet()

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    phase = "idle"
    enroll_name = ""
    all_embeddings = []
    scene_embeddings = []
    scene_idx = 0
    scene_counts = [0] * len(SCENARIOS)
    collecting = False
    collect_start = None
    start_good = None
    last_msg = "Idle"
    last_ratio = 0.0
    last_blur = 0.0
    last_bbox = None

    def reset_enroll():
        nonlocal all_embeddings, scene_embeddings, scene_idx, scene_counts, collecting, collect_start, start_good
        all_embeddings = []
        scene_embeddings = []
        scene_idx = 0
        scene_counts = [0] * len(SCENARIOS)
        collecting = False
        collect_start = None
        start_good = None

    print("[INFO] Worker loop start")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue
        frame = cv2.flip(frame, 1)

        fps_count += 1
        now = time.time()
        if fps_count >= 15:
            fps_display = fps_count / max(1e-6, (now - fps_t0))
            fps_t0 = now
            fps_count = 0

        with _lock:
            action = _cmd["action"]
            if action == "start":
                enroll_name = _cmd["name"]
                _cmd["action"] = None
                phase = "enrolling"
                reset_enroll()
                print(f"[ENROLL] Start: {enroll_name}")

        if phase in ("idle", "done"):
            faces = _detector.detect(frame)
            face = _detector.largest(faces)
            if face:
                x, y, bw, bh = face["box"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (100, 100, 100), 1)
            cv2.putText(frame, "Look at camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 170, 170), 1)
            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if ok:
                with _lock:
                    _jpeg_frame = buf.tobytes()
                    _status["fps"] = round(fps_display, 1)
            time.sleep(0.01)
            continue

        sc_name, instruction, expected_range, duration = SCENARIOS[scene_idx]
        faces = _detector.detect(frame)
        face = _detector.largest(faces)
        good = False
        crop = None

        if face:
            last_bbox = face["box"]
            last_ratio = _detector.face_ratio(face, frame)
            low, high = expected_range
            crop = _detector.crop(frame, face)
            last_blur = blur_score(crop)
            if last_ratio < low:
                last_msg = "Move closer"
            elif last_ratio > high:
                last_msg = "Move back"
            elif last_blur < args.min_blur:
                last_msg = "Blurred"
            else:
                last_msg = "OK"
                good = True
        else:
            last_bbox = None
            last_ratio = 0.0
            last_blur = 0.0
            last_msg = "No face"

        if good:
            if start_good is None:
                start_good = now
            if not collecting and now - start_good >= args.stable_time:
                collecting = True
                collect_start = now
                with _lock:
                    _status["last_log"] = f"[REC] {sc_name}"
        else:
            start_good = None
            if not collecting:
                collect_start = None

        if collecting and crop is not None and good:
            emb = _embedder.embed(crop)
            scene_embeddings.append(emb)

        sc_done = collecting and (
            len(scene_embeddings) >= args.target_frames
            or (collect_start is not None and now - collect_start >= duration)
        )

        if sc_done:
            arr = np.array(scene_embeddings, dtype=np.float32)
            centroid = arr.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            sims = arr @ centroid
            top_k = max(3, int(len(arr) * 0.6))
            top_idx = np.argsort(sims)[-top_k:]
            selected = [arr[i].tolist() for i in top_idx]
            all_embeddings.extend(selected)
            scene_counts[scene_idx] = len(selected)
            log_msg = f"[SC {scene_idx + 1}/{len(SCENARIOS)}] {sc_name}: {len(selected)} emb"
            print(log_msg)

            scene_idx += 1
            scene_embeddings = []
            collecting = False
            collect_start = None
            start_good = None

            if scene_idx >= len(SCENARIOS):
                db = load_db()
                db[enroll_name] = [e for e in all_embeddings]
                save_db(db)
                print(f"[DONE] {enroll_name}: {len(all_embeddings)} embeddings")
                with _lock:
                    _status.update(
                        {
                            "phase": "done",
                            "scene_idx": len(SCENARIOS) - 1,
                            "scene_counts": scene_counts[:],
                            "total_collected": len(all_embeddings),
                            "msg": "Done",
                            "instruction": f"Saved {len(all_embeddings)} embeddings.",
                            "last_log": f"[DONE] {enroll_name}: {len(all_embeddings)} emb",
                            "name": enroll_name,
                        }
                    )
                phase = "done"
                continue
            else:
                with _lock:
                    _status["last_log"] = log_msg

        draw_register_ui(frame, last_msg, sc_name, last_ratio, last_blur, scene_idx, len(SCENARIOS), last_bbox)
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if ok:
            with _lock:
                _jpeg_frame = buf.tobytes()
                _status.update(
                    {
                        "phase": "enrolling",
                        "scene_idx": scene_idx,
                        "scene_counts": scene_counts[:],
                        "total_collected": len(all_embeddings) + len(scene_embeddings),
                        "msg": last_msg,
                        "instruction": f"<b>{sc_name}</b> - {instruction}",
                        "ratio": round(last_ratio, 3),
                        "blur": round(last_blur, 1),
                        "fps": round(fps_display, 1),
                        "target": args.target_frames,
                        "name": enroll_name,
                        "last_log": None,
                    }
                )


app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with _lock:
                frame = _jpeg_frame
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with _lock:
        s = dict(_status)
    return jsonify(s)


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "msg": "Ten khong duoc rong"})
    with _lock:
        if _status["phase"] == "enrolling":
            return jsonify({"ok": False, "msg": "Dang dang ky, vui long doi"})
        _cmd["action"] = "start"
        _cmd["name"] = name
        _status["phase"] = "enrolling"
        _status["last_log"] = None
    return jsonify({"ok": True})


@app.route("/upload_register", methods=["POST"])
def upload_register():
    global _detector, _embedder
    name = (request.form.get("name") or "").strip()
    append = request.form.get("append", "0") == "1"
    file = request.files.get("image")
    if not name:
        return jsonify({"ok": False, "msg": "Thieu ten"})
    if file is None:
        return jsonify({"ok": False, "msg": "Thieu file anh"})
    # Use a separate YuNet instance with a lower threshold for still-image upload.
    # The global DNN lock in YuNetDetector.detect() keeps this safe with the camera thread.
    upload_detector = YuNetDetector(score_threshold=0.60)
    if _embedder is None:
        _embedder = MobileFaceNet()
    image_bytes = file.read()
    result = enroll_from_image_bytes(name, image_bytes, upload_detector, _embedder, append=append)
    print(f"[UPLOAD] {result}")
    return jsonify(result)


@app.route("/db")
def get_db():
    db = load_db()
    return jsonify({k: len(v) for k, v in db.items()})


@app.route("/delete", methods=["POST"])
def delete_person():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "msg": "Thieu ten"})
    db = load_db()
    if name not in db:
        return jsonify({"ok": False, "msg": f"Khong tim thay: {name}"})
    del db[name]
    save_db(db)
    print(f"[DEL] {name}")
    return jsonify({"ok": True})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=9)
    p.add_argument("--use-usb-camera", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--target-frames", type=int, default=12)
    p.add_argument("--min-blur", type=float, default=25.0)
    p.add_argument("--stable-time", type=float, default=0.6)
    p.add_argument("--score-threshold", type=float, default=0.82)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    t = threading.Thread(target=worker_loop, args=(args,), daemon=True)
    t.start()

    print(f"\n[WEB] Open browser: http://<PI_IP>:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
