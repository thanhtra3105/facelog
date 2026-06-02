#!/usr/bin/env python3
"""
register_upload_admin_new.py

Dang ky khuon mat bang anh upload, co dang nhap admin.
Ban nay dung model moi: output_model.tflite

Model moi:
    input  = [1, 112, 112, 3], float32
    output = [1, 128], float32

Chay:
    export ADMIN_USER=admin
    export ADMIN_PASSWORD=doi_mat_khau_manh
    python3 register_upload_admin_new.py

Mo browser:
    http://<PI_IP>:5001

Can co cung thu muc:
    output_model.tflite
    face_detection_yunet_2023mar.onnx  # neu chua co se tu tai
"""

import argparse
import io
import json
import os
import urllib.request
from pathlib import Path
from functools import wraps
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, session, redirect, url_for
from PIL import Image, ImageOps


# =========================
# TFLite Interpreter
# =========================
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "output_model.tflite"
DB_PATH = BASE_DIR / "face_database_new_model.json"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"

EXPECTED_EMB_DIM = 128

ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
APP_SECRET = os.environ.get("APP_SECRET", "change-this-secret-key-for-session")


# =========================
# HTML
# =========================
LOGIN_PAGE = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin Login</title>
<style>
*{box-sizing:border-box}
body{
    margin:0;
    background:#0f0f0f;
    color:#e8e8e8;
    font-family:Arial,sans-serif;
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:18px;
}
.card{
    width:100%;
    max-width:380px;
    background:#181818;
    border:1px solid #2a2a2a;
    border-radius:14px;
    padding:20px;
}
h1{font-size:1.2rem;margin:0 0 6px}
.muted{color:#999;font-size:.9rem;margin-bottom:14px}
label{display:block;margin:10px 0 6px;color:#ccc;font-size:.9rem}
input{
    width:100%;
    padding:11px;
    border-radius:8px;
    border:1px solid #333;
    background:#111;
    color:#eee;
}
button{
    width:100%;
    margin-top:14px;
    border:0;
    border-radius:8px;
    padding:11px 14px;
    font-weight:700;
    cursor:pointer;
    background:#2563eb;
    color:white;
}
.err{
    background:#3a1515;
    color:#fca5a5;
    border:1px solid #7f1d1d;
    border-radius:8px;
    padding:8px;
    margin:10px 0;
    font-size:.9rem;
}
.small{
    margin-top:12px;
    font-size:.8rem;
    color:#777;
    line-height:1.35;
}
</style>
</head>
<body>
  <form class="card" method="post" action="/login">
    <h1>Admin Login</h1>
    <div class="muted">Dang nhap de upload anh dang ky khuon mat.</div>
    {% if error %}<div class="err">{{ error }}</div>{% endif %}
    <label>Username</label>
    <input name="username" autocomplete="username" required autofocus>
    <label>Password</label>
    <input name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Dang nhap</button>
    <div class="small">Nen doi mat khau bang bien moi truong ADMIN_PASSWORD truoc khi chay.</div>
  </form>
</body>
</html>
"""


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dang ky khuon mat</title>
<style>
*{box-sizing:border-box}
body{
    margin:0;
    background:#0f0f0f;
    color:#e8e8e8;
    font-family:Arial,sans-serif;
    padding:18px;
}
.container{
    max-width:860px;
    margin:0 auto;
    display:grid;
    gap:14px;
}
.card{
    background:#181818;
    border:1px solid #2a2a2a;
    border-radius:12px;
    padding:16px;
}
h1{font-size:1.15rem;margin:0 0 4px;color:#fff}
.muted{color:#999;font-size:.9rem}
.row{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:14px;
}
@media(max-width:760px){
    .row{grid-template-columns:1fr}
}
label{
    display:block;
    margin:10px 0 6px;
    color:#ccc;
    font-size:.9rem;
}
input[type=text],input[type=number]{
    width:100%;
    padding:10px;
    border-radius:8px;
    border:1px solid #333;
    background:#111;
    color:#eee;
}
input[type=file]{
    width:100%;
    padding:10px;
    border-radius:8px;
    border:1px solid #333;
    background:#111;
    color:#eee;
}
button{
    border:0;
    border-radius:8px;
    padding:10px 14px;
    font-weight:700;
    cursor:pointer;
    background:#2563eb;
    color:white;
}
button:disabled{
    background:#333;
    color:#777;
    cursor:not-allowed;
}
.danger{background:#7f1d1d}
.ok{color:#22c55e}
.warn{color:#f59e0b}
.err{color:#f87171}
#preview{
    max-width:100%;
    border-radius:10px;
    border:1px solid #333;
    display:none;
}
.result{
    font-family:monospace;
    white-space:pre-wrap;
    background:#101010;
    border-radius:8px;
    padding:10px;
    min-height:52px;
}
table{
    width:100%;
    border-collapse:collapse;
    margin-top:8px;
}
td,th{
    border-bottom:1px solid #2a2a2a;
    padding:8px;
    text-align:left;
    font-size:.92rem;
}
th{color:#aaa;font-weight:500}
.right{text-align:right}
.small{font-size:.82rem;color:#999}
.pill{
    display:inline-block;
    background:#222;
    border:1px solid #333;
    padding:3px 7px;
    border-radius:999px;
    margin:2px;
    color:#bbb;
}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>Dang ky khuon mat bang anh upload</h1>
    <div class="muted">
      Ban nay dung model moi <b>output_model.tflite</b>, embedding 128 chieu.
      Anh upload se duoc detect bang YuNet, align 112x112, augment, roi trich embedding bang MobileFaceNet.
    </div>
    <div style="margin-top:10px">
      <a href="/logout" style="color:#93c5fd;text-decoration:none">Dang xuat admin</a>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h1>Upload anh</h1>
      <form id="form">
        <label>Ten nguoi dung</label>
        <input id="name" name="name" type="text" placeholder="VD: thanhtra" required maxlength="50">

        <label>Anh khuon mat</label>
        <input id="image" name="image" type="file" accept="image/*" required>

        <label>So bien the augmentation</label>
        <input id="num_aug" name="num_aug" type="number" min="1" max="30" value="16">

        <label>
          <input id="append" name="append" type="checkbox">
          Them vao nguoi cu neu da ton tai
        </label>

        <div style="margin-top:12px">
          <button id="btn" type="submit">Upload va dang ky</button>
        </div>
      </form>

      <div style="margin-top:12px" class="small">
        Nen dung anh ro mat, mat khong bi che, khong qua toi, chi co 1 mat chinh trong anh.
      </div>
    </div>

    <div class="card">
      <h1>Preview</h1>
      <img id="preview">
      <div id="result" class="result muted">Chua upload anh.</div>
      <div style="margin-top:8px">
        <span class="pill">YuNet detect</span>
        <span class="pill">Face align 112x112</span>
        <span class="pill">Augmentation</span>
        <span class="pill">Embedding 128D</span>
      </div>
    </div>
  </div>

  <div class="card">
    <h1>Database hien tai</h1>
    <table>
      <thead>
        <tr>
          <th>Ten</th>
          <th>So embedding</th>
          <th>Dim</th>
          <th class="right">Thao tac</th>
        </tr>
      </thead>
      <tbody id="dbbody"></tbody>
    </table>
  </div>
</div>

<script>
const form = document.getElementById('form');
const btn = document.getElementById('btn');
const result = document.getElementById('result');
const preview = document.getElementById('preview');
const imageInput = document.getElementById('image');

imageInput.addEventListener('change', () => {
  const f = imageInput.files[0];
  if (!f) return;
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  btn.disabled = true;
  result.className = 'result warn';
  result.textContent = 'Dang xu ly anh...';

  const fd = new FormData(form);

  try {
    const r = await fetch('/upload_register', {
        method: 'POST',
        body: fd
    });

    const d = await r.json();

    if (d.ok) {
      result.className = 'result ok';
      result.textContent =
        `OK\n` +
        `Ten: ${d.name}\n` +
        `Embeddings da luu: ${d.saved_embeddings}\n` +
        `Augment hop le: ${d.num_augmented}\n` +
        `Embedding dim: ${d.embedding_dim}\n` +
        `Face score: ${d.face_score.toFixed(3)}\n` +
        `Ratio: ${d.face_ratio.toFixed(3)}`;
      loadDB();
    } else {
      result.className = 'result err';
      result.textContent = 'LOI: ' + (d.msg || 'Khong ro loi');
    }
  } catch (err) {
    result.className = 'result err';
    result.textContent = 'LOI request: ' + err;
  }

  btn.disabled = false;
});

async function loadDB(){
  const r = await fetch('/db');
  const db = await r.json();

  const body = document.getElementById('dbbody');
  body.innerHTML = '';

  Object.entries(db).forEach(([name, info]) => {
    const safeName = name.replaceAll("'", "\\'");
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td>${name}</td>` +
      `<td>${info.count}</td>` +
      `<td>${info.dim}</td>` +
      `<td class="right"><button class="danger" onclick="delPerson('${safeName}')">Xoa</button></td>`;
    body.appendChild(tr);
  });

  if (Object.keys(db).length === 0) {
    body.innerHTML = '<tr><td colspan="4" class="muted">Database rong</td></tr>';
  }
}

async function delPerson(name){
  if(!confirm('Xoa ' + name + '?')) return;

  const r = await fetch('/delete', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name})
  });

  const d = await r.json();

  if(!d.ok) alert(d.msg || 'Loi');

  loadDB();
}

loadDB();
</script>
</body>
</html>
"""


# =========================
# Helpers
# =========================
def ensure_yunet(path: Path):
    if path.exists() and path.stat().st_size > 100_000:
        return

    print(f"[INFO] Download YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] YuNet downloaded: {path.stat().st_size // 1024} KB")


# =========================
# MobileFaceNet moi
# =========================
class MobileFaceNet:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

        self.interp = Interpreter(model_path=str(MODEL_PATH))
        self.interp.allocate_tensors()

        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]

        self.input_shape = self.inp["shape"]
        self.input_dtype = self.inp["dtype"]
        self.output_shape = self.outp["shape"]
        self.output_dtype = self.outp["dtype"]

        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])
        self.embedding_dim = int(self.output_shape[-1])

        print("[INFO] MobileFaceNet ready")
        print("[INFO] Model path:", MODEL_PATH)
        print("[INFO] Input :", self.input_shape, self.input_dtype)
        print("[INFO] Output:", self.output_shape, self.output_dtype)

        if self.embedding_dim != EXPECTED_EMB_DIM:
            print(f"[WARN] Model output dim = {self.embedding_dim}, expected {EXPECTED_EMB_DIM}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (self.input_w, self.input_h))

        # OpenCV BGR -> model RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.float32:
            face = face.astype(np.float32)
            face = (face - 127.5) / 128.0
        else:
            face = face.astype(self.input_dtype)

        face = np.expand_dims(face, axis=0)

        self.interp.set_tensor(self.inp["index"], face)
        self.interp.invoke()

        emb = self.interp.get_tensor(self.outp["index"])[0]
        emb = emb.astype(np.float32)

        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb


# =========================
# YuNet Detector
# =========================
class YuNetDetector:
    def __init__(self, score_threshold: float = 0.55):
        ensure_yunet(YUNET_PATH)

        self.det = cv2.FaceDetectorYN_create(
            str(YUNET_PATH),
            "",
            (320, 320),
            score_threshold,
            0.30,
            5000,
        )

        self.score_threshold = score_threshold

        print(f"[INFO] YuNet ready: {YUNET_PATH}, score_threshold={score_threshold}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]

        self.det.setInputSize((w, h))

        try:
            _, faces = self.det.detect(frame)
        except cv2.error as e:
            print(f"[WARN] YuNet detect error: {e}")
            return []

        if faces is None:
            return []

        results = []

        for f in faces:
            if not np.all(np.isfinite(f)):
                continue

            x, y, bw, bh = [float(v) for v in f[:4]]
            score = float(f[-1])

            if bw <= 5 or bh <= 5 or score < self.score_threshold:
                continue

            x = max(0, min(int(round(x)), w - 1))
            y = max(0, min(int(round(y)), h - 1))
            bw = max(1, min(int(round(bw)), w - x))
            bh = max(1, min(int(round(bh)), h - y))

            landmarks = f[4:14].reshape(5, 2).astype(np.float32)

            if not np.all(np.isfinite(landmarks)):
                landmarks = None

            results.append(
                {
                    "box": (x, y, bw, bh),
                    "landmarks": landmarks,
                    "score": score,
                }
            )

        return results

    @staticmethod
    def largest(faces: List[Dict]) -> Optional[Dict]:
        if not faces:
            return None

        return max(faces, key=lambda f: f["box"][2] * f["box"][3])

    @staticmethod
    def face_ratio(face: Dict, frame: np.ndarray) -> float:
        _, _, w, h = face["box"]
        return max(w, h) / max(1, min(frame.shape[:2]))

    @staticmethod
    def align_crop(frame: np.ndarray, landmarks: Optional[np.ndarray], size: int = 112) -> Optional[np.ndarray]:
        if landmarks is None:
            return None

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

        try:
            M, _ = cv2.estimateAffinePartial2D(
                landmarks.astype(np.float32),
                dst,
                method=cv2.LMEDS,
            )

            if M is None:
                return None

            return cv2.warpAffine(
                frame,
                M,
                (size, size),
                borderValue=0.0,
            )

        except cv2.error:
            return None

    @staticmethod
    def crop_fallback(frame: np.ndarray, face: Dict, pad_ratio: float = 0.20) -> Optional[np.ndarray]:
        x, y, w, h = face["box"]

        pad = int(pad_ratio * max(w, h))

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return cv2.resize(crop, (112, 112))

    def crop(self, frame: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        crop = self.align_crop(frame, face.get("landmarks"))

        if crop is not None:
            return crop

        return self.crop_fallback(frame, face)


# =========================
# Image read / augment
# =========================
def read_upload_image(file_storage) -> Optional[np.ndarray]:
    raw = file_storage.read()

    if not raw:
        return None

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        rgb = np.array(img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    except Exception:
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img.shape[:2]

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )


def adjust_bc(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_noise(img: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_face(face_bgr: np.ndarray, max_count: int = 16) -> List[np.ndarray]:
    face = cv2.resize(face_bgr, (112, 112))

    variants = [
        face,
        adjust_bc(face, 1.0, 18),
        adjust_bc(face, 1.0, -18),
        adjust_bc(face, 1.12, 0),
        adjust_bc(face, 0.88, 0),
        rotate_bound(face, -7),
        rotate_bound(face, 7),
        rotate_bound(face, -4),
        rotate_bound(face, 4),
        shift_image(face, -5, 0),
        shift_image(face, 5, 0),
        shift_image(face, 0, -5),
        shift_image(face, 0, 5),
        cv2.GaussianBlur(face, (3, 3), 0),
        add_noise(face, 3.0),
        cv2.flip(face, 1),
        adjust_bc(rotate_bound(face, -5), 1.08, 8),
        adjust_bc(rotate_bound(face, 5), 0.95, -6),
        add_noise(adjust_bc(face, 1.08, 8), 2.0),
        cv2.GaussianBlur(adjust_bc(face, 0.95, -5), (3, 3), 0),
    ]

    max_count = max(1, min(max_count, len(variants)))

    return variants[:max_count]


def try_find_face(detector: YuNetDetector, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict], np.ndarray]:
    candidates = [
        img_bgr,
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    for cand in candidates:
        faces = detector.detect(cand)
        face = detector.largest(faces)

        if face is None:
            continue

        crop = detector.crop(cand, face)

        if crop is not None:
            return crop, face, cand

    return None, None, img_bgr


# =========================
# Database
# =========================
def load_db() -> Dict[str, List]:
    if not DB_PATH.exists():
        return {}

    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    db = {}

    for name, value in raw.items():
        if isinstance(value, list):
            db[name] = value

        elif isinstance(value, dict) and "embeddings" in value:
            db[name] = value["embeddings"]

    return db


def save_db(db: Dict[str, List]):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def get_embedding_dim(embs: List) -> str:
    if not embs:
        return "-"

    try:
        return str(len(embs[0]))
    except Exception:
        return "?"


def filter_valid_db(db: Dict[str, List]) -> Dict[str, List]:
    """
    Bo qua embedding cu khac 128 chieu.
    Ham nay giup khong tron DB 192D cu voi DB 128D moi.
    """
    clean = {}

    for name, embs in db.items():
        valid = []

        for e in embs:
            try:
                arr = np.array(e, dtype=np.float32)

                if arr.shape[0] == EXPECTED_EMB_DIM:
                    arr = arr / (np.linalg.norm(arr) + 1e-8)
                    valid.append(arr.tolist())

                else:
                    print(f"[WARN] Bo qua embedding cu cua {name}, dim={arr.shape[0]}")

            except Exception as ex:
                print(f"[WARN] Bo qua embedding loi cua {name}: {ex}")

        if valid:
            clean[name] = valid

    return clean


# =========================
# Flask
# =========================
app = Flask(__name__)
app.secret_key = APP_SECRET

_detector: Optional[YuNetDetector] = None
_embedder: Optional[MobileFaceNet] = None


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            if request.path.startswith(("/upload_register", "/db", "/delete")):
                return jsonify({"ok": False, "msg": "Chua dang nhap admin"}), 401

            return redirect(url_for("login"))

        return fn(*args, **kwargs)

    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if username == ADMIN_USER and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            session["admin_user"] = username
            return redirect(url_for("index"))

        return render_template_string(LOGIN_PAGE, error="Sai username hoac password")

    return render_template_string(LOGIN_PAGE, error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template_string(HTML_PAGE)


@app.route("/upload_register", methods=["POST"])
@login_required
def upload_register():
    global _detector, _embedder

    name = (request.form.get("name") or "").strip()
    append = request.form.get("append") in ("on", "true", "1")

    try:
        num_aug = int(request.form.get("num_aug") or 16)
    except ValueError:
        num_aug = 16

    num_aug = max(1, min(num_aug, 30))

    if not name:
        return jsonify({"ok": False, "msg": "Thieu ten nguoi dung"})

    if "image" not in request.files:
        return jsonify({"ok": False, "msg": "Thieu file anh"})

    img_bgr = read_upload_image(request.files["image"])

    if img_bgr is None:
        return jsonify({"ok": False, "msg": "Khong doc duoc anh upload"})

    crop, face, used_img = try_find_face(_detector, img_bgr)

    if crop is None or face is None:
        return jsonify(
            {
                "ok": False,
                "msg": "Khong tim thay khuon mat trong anh. Hay dung anh ro hon, mat lon hon, it nghieng hon.",
            }
        )

    aug_faces = augment_face(crop, max_count=num_aug)

    embeddings = []

    for face_img in aug_faces:
        emb = _embedder.embed(face_img)

        if emb.shape[0] != EXPECTED_EMB_DIM:
            return jsonify(
                {
                    "ok": False,
                    "msg": f"Embedding dim sai: {emb.shape[0]}, can {EXPECTED_EMB_DIM}",
                }
            )

        embeddings.append(emb.tolist())

    db = load_db()

    # Loc bo embedding cu khac 128D truoc khi luu
    db = filter_valid_db(db)

    if append and name in db:
        db[name].extend(embeddings)
    else:
        db[name] = embeddings

    save_db(db)

    ratio = _detector.face_ratio(face, used_img)

    print(
        f"[UPLOAD] name={name}, aug={len(embeddings)}, append={append}, "
        f"score={face['score']:.3f}, ratio={ratio:.3f}, dim={EXPECTED_EMB_DIM}"
    )

    return jsonify(
        {
            "ok": True,
            "name": name,
            "saved_embeddings": len(db[name]),
            "num_augmented": len(embeddings),
            "embedding_dim": EXPECTED_EMB_DIM,
            "face_score": float(face["score"]),
            "face_ratio": float(ratio),
        }
    )


@app.route("/db")
@login_required
def get_db():
    db = load_db()

    result = {}

    for name, embs in db.items():
        result[name] = {
            "count": len(embs),
            "dim": get_embedding_dim(embs),
        }

    return jsonify(result)


@app.route("/delete", methods=["POST"])
@login_required
def delete_person():
    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()

    if not name:
        return jsonify({"ok": False, "msg": "Thieu ten"})

    db = load_db()

    if name not in db:
        return jsonify({"ok": False, "msg": "Khong tim thay nguoi nay"})

    del db[name]
    save_db(db)

    print(f"[DELETE] {name}")

    return jsonify({"ok": True})


# =========================
# Main
# =========================
def main():
    global _detector, _embedder

    p = argparse.ArgumentParser()

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5001)

    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.55,
        help="Nguong YuNet cho anh upload. Giam neu anh kho detect.",
    )

    args = p.parse_args()

    _detector = YuNetDetector(score_threshold=args.score_threshold)
    _embedder = MobileFaceNet()

    if ADMIN_PASSWORD == "admin123":
        print("[WARN] Dang dung mat khau mac dinh admin123.")
        print("[WARN] Nen doi bang lenh:")
        print("       export ADMIN_PASSWORD=mat_khau_manh")

    print(f"[INFO] Admin user: {ADMIN_USER}")

    print(f"\n[WEB] Open browser: http://<PI_IP>:{args.port}")
    print("[INFO] Upload-only mode: khong mo camera, khong co thread video.")
    print("[INFO] Database:", DB_PATH)
    print("[INFO] Model:", MODEL_PATH)
    print()

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
