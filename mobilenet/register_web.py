#!/usr/bin/env python3
"""
register_web.py
Dang ky khuon mat qua web browser — khong can man hinh, khong can VNC.

Chay:
    python register_web.py
    python register_web.py --use-usb-camera --camera 0

Roi mo browser: http://<PI_IP>:5001
"""

import argparse
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mobilefacenet.tflite"
DB_PATH    = BASE_DIR / "face_database.json"
YUNET_URL  = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = BASE_DIR / "face_detection_yunet_2023mar.onnx"
IMG_SIZE   = 112

SCENARIOS = [
    ("Thẳng vừa",    "Nhìn thẳng vào camera, khoảng cách vừa", (0.14, 0.35), 2.5),
    ("Nghiêng trái", "Quay mặt sang TRÁI nhẹ",                 (0.12, 0.35), 2.5),
    ("Nghiêng phải", "Quay mặt sang PHẢI nhẹ",                 (0.12, 0.35), 2.5),
    ("Ngửa lên",     "Ngửa đầu lên nhẹ",                       (0.12, 0.35), 2.5),
    ("Cúi xuống",    "Cúi đầu xuống nhẹ",                      (0.12, 0.35), 2.5),
]

# ──────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đăng ký khuôn mặt</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e0e0e0;font-family:monospace;
     display:flex;flex-direction:column;align-items:center;padding:16px;gap:12px}
h1{font-size:.95rem;color:#888;letter-spacing:2px;text-transform:uppercase}

/* stream */
#wrap{position:relative;width:100%;max-width:640px}
#stream{width:100%;border-radius:6px;display:block}
#overlay-badge{position:absolute;top:10px;left:10px;padding:4px 10px;
  border-radius:4px;font-size:.8rem;font-weight:bold;pointer-events:none;
  background:#000a;color:#fff;transition:background .3s}
#overlay-badge.ok    {background:#1a6b3acc}
#overlay-badge.warn  {background:#7a4e00cc}
#overlay-badge.error {background:#6b1a1acc}

/* input row */
#input-row{display:flex;gap:8px;width:100%;max-width:640px}
#name-input{flex:1;background:#1c1c1c;border:1px solid #333;border-radius:6px;
  padding:8px 12px;color:#eee;font-size:.9rem;outline:none}
#name-input:focus{border-color:#555}
#btn-start{padding:8px 20px;border:none;border-radius:6px;cursor:pointer;
  font-size:.9rem;font-weight:bold;background:#2563eb;color:#fff;transition:background .2s}
#btn-start:hover{background:#1d4ed8}
#btn-start:disabled{background:#333;color:#666;cursor:not-allowed}

/* scenario progress */
#scenarios{width:100%;max-width:640px;display:flex;flex-direction:column;gap:6px}
.sc-row{display:flex;align-items:center;gap:10px;padding:6px 10px;
  border-radius:6px;background:#1a1a1a;transition:background .3s}
.sc-row.active{background:#1a2a3a;border:1px solid #2563eb}
.sc-row.done  {background:#0d2218;border:1px solid #1a6b3a}
.sc-dot{width:10px;height:10px;border-radius:50%;background:#444;flex-shrink:0}
.sc-row.active .sc-dot{background:#2563eb;box-shadow:0 0 6px #2563eb}
.sc-row.done   .sc-dot{background:#22c55e}
.sc-label{font-size:.82rem;flex:1}
.sc-count{font-size:.75rem;color:#888}

/* instruction box */
#instruction{width:100%;max-width:640px;background:#1a1a1a;border-radius:6px;
  padding:10px 14px;font-size:.85rem;min-height:42px;color:#ccc;
  border-left:3px solid #2563eb}

/* stats row */
#stats{display:flex;gap:16px;width:100%;max-width:640px;flex-wrap:wrap}
.stat{background:#1a1a1a;border-radius:6px;padding:6px 12px;font-size:.78rem}
.stat span{color:#888}

/* log */
#log{width:100%;max-width:640px;background:#141414;border:1px solid #222;
  border-radius:6px;padding:8px 10px;font-size:.75rem;height:100px;
  overflow-y:auto;color:#666}

/* db table */
#db-wrap{width:100%;max-width:640px}
#db-wrap h2{font-size:.8rem;color:#666;margin-bottom:6px;letter-spacing:1px}
#db-table{width:100%;border-collapse:collapse;font-size:.8rem}
#db-table th,#db-table td{padding:5px 10px;text-align:left;border-bottom:1px solid #222}
#db-table th{color:#666;font-weight:normal}
#db-table td button{background:#6b1a1a;border:none;color:#f87171;
  padding:2px 8px;border-radius:4px;cursor:pointer;font-size:.75rem}
#db-table td button:hover{background:#7f1d1d}
</style>
</head>
<body>
<h1>&#x1F464; Đăng ký khuôn mặt — Raspberry Pi</h1>

<div id="wrap">
  <img id="stream" src="/video_feed" alt="stream">
  <div id="overlay-badge" class="warn">Chờ bắt đầu</div>
</div>

<div id="input-row">
  <input id="name-input" placeholder="Nhập tên (VD: tra, nam, ...)" maxlength="40">
  <button id="btn-start" onclick="startRegister()">Bắt đầu đăng ký</button>
</div>

<div id="scenarios"></div>

<div id="instruction">Nhập tên và bấm <b>Bắt đầu đăng ký</b>.</div>

<div id="stats">
  <div class="stat"><span>FPS: </span><span id="s-fps">—</span></div>
  <div class="stat"><span>Tỉ lệ mặt: </span><span id="s-ratio">—</span></div>
  <div class="stat"><span>Blur: </span><span id="s-blur">—</span></div>
  <div class="stat"><span>Đã thu: </span><span id="s-collected">0</span></div>
</div>

<div id="log"></div>

<div id="db-wrap">
  <h2>DATABASE HIỆN TẠI</h2>
  <table id="db-table">
    <thead><tr><th>Tên</th><th>Embeddings</th><th></th></tr></thead>
    <tbody id="db-body"></tbody>
  </table>
</div>

<script>
const SCENARIOS = [
  "Thẳng vừa","Nghiêng trái","Nghiêng phải","Ngửa lên","Cúi xuống"
];

function buildScenarios() {
  const wrap = document.getElementById('scenarios');
  wrap.innerHTML = '';
  SCENARIOS.forEach((s,i) => {
    const d = document.createElement('div');
    d.className = 'sc-row'; d.id = 'sc-'+i;
    d.innerHTML = `<div class="sc-dot"></div>
      <div class="sc-label">${i+1}. ${s}</div>
      <div class="sc-count" id="sc-cnt-${i}">0 emb</div>`;
    wrap.appendChild(d);
  });
}
buildScenarios();

let polling = false;

function startRegister() {
  const name = document.getElementById('name-input').value.trim();
  if (!name) { alert('Nhập tên trước!'); return; }
  fetch('/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})
  }).then(r=>r.json()).then(d=>{
    if (d.ok) {
      document.getElementById('btn-start').disabled = true;
      addLog(`[START] Bắt đầu đăng ký: ${name}`);
      if (!polling) startPoll();
    } else {
      alert(d.msg || 'Lỗi');
    }
  });
}

function startPoll() {
  polling = true;
  poll();
}

function poll() {
  fetch('/status').then(r=>r.json()).then(d=>{
    // stats
    document.getElementById('s-fps').textContent    = d.fps.toFixed(1);
    document.getElementById('s-ratio').textContent  = d.ratio.toFixed(2);
    document.getElementById('s-blur').textContent   = d.blur.toFixed(0);
    document.getElementById('s-collected').textContent = d.total_collected;

    // badge
    const badge = document.getElementById('overlay-badge');
    if (d.phase === 'idle') {
      badge.textContent = 'Chờ bắt đầu'; badge.className='overlay-badge warn';
    } else if (d.phase === 'done') {
      badge.textContent = '✓ Hoàn thành!'; badge.className='overlay-badge ok';
    } else {
      badge.textContent = d.msg; 
      badge.className = 'overlay-badge ' + (d.msg==='OK' ? 'ok' : 'warn');
    }

    // instruction
    document.getElementById('instruction').innerHTML = d.instruction || '';

    // scenarios
    SCENARIOS.forEach((_,i) => {
      const row = document.getElementById('sc-'+i);
      const cnt = document.getElementById('sc-cnt-'+i);
      cnt.textContent = (d.scene_counts[i]||0) + ' emb';
      row.className = 'sc-row';
      if (i === d.scene_idx && d.phase === 'enrolling') row.className += ' active';
      if ((d.scene_counts[i]||0) >= d.target) row.className += ' done';
    });

    // log
    if (d.last_log) addLog(d.last_log);

    // done
    if (d.phase === 'done') {
      document.getElementById('btn-start').disabled = false;
      document.getElementById('instruction').innerHTML =
        `<b style="color:#22c55e">✓ Đã lưu ${d.total_collected} embeddings cho "${d.name}"</b>`;
      loadDB();
      polling = false;
      return;
    }
    if (d.phase !== 'idle') setTimeout(poll, 350);
    else { polling = false; document.getElementById('btn-start').disabled = false; }
  }).catch(()=>{ if(polling) setTimeout(poll, 600); });
}

function addLog(msg) {
  const log = document.getElementById('log');
  const d = document.createElement('div');
  d.textContent = new Date().toLocaleTimeString('vi') + '  ' + msg;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function loadDB() {
  fetch('/db').then(r=>r.json()).then(db=>{
    const tbody = document.getElementById('db-body');
    tbody.innerHTML = '';
    Object.entries(db).forEach(([name, count])=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${name}</td><td>${count}</td>
        <td><button onclick="deletePerson('${name}')">Xóa</button></td>`;
      tbody.appendChild(tr);
    });
  });
}

function deletePerson(name) {
  if (!confirm(`Xóa "${name}" khỏi database?`)) return;
  fetch('/delete', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})
  }).then(r=>r.json()).then(d=>{
    if (d.ok) { addLog(`[DEL] Đã xóa: ${name}`); loadDB(); }
    else alert(d.msg);
  });
}

loadDB();
// auto reload db mỗi 5s khi không đang enroll
setInterval(()=>{ if(!polling) loadDB(); }, 5000);
</script>
</body>
</html>"""
# ──────────────────────────────────────────────


def ensure_yunet(path):
    if path.exists() and path.stat().st_size > 100_000: return
    print(f"[INFO] Tai YuNet -> {path}")
    urllib.request.urlretrieve(YUNET_URL, str(path))
    print(f"[INFO] Xong: {path.stat().st_size//1024} KB")


class MobileFaceNet:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Khong tim thay {MODEL_PATH}")
        self.interp = Interpreter(model_path=str(MODEL_PATH))
        self.interp.allocate_tensors()
        self.inp  = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]
        print("[INFO] MobileFaceNet ready")

    def embed(self, face_bgr):
        face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = (face.astype(np.float32) - 127.5) / 128.0
        self.interp.set_tensor(self.inp["index"], face[np.newaxis])
        self.interp.invoke()
        emb = self.interp.get_tensor(self.outp["index"])[0]
        return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)


class YuNetDetector:
    def __init__(self, score_threshold=0.82):
        ensure_yunet(YUNET_PATH)
        self.det = cv2.FaceDetectorYN_create(
            str(YUNET_PATH), "", (320,320), score_threshold, 0.3, 5000)
        print("[INFO] YuNet ready")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        if faces is None: return []
        return [{"box": tuple(int(x) for x in f[:4]),
                 "landmarks": f[4:14].reshape(5,2).astype(np.float32),
                 "score": float(f[-1])} for f in faces]

    @staticmethod
    def largest(faces):
        return max(faces, key=lambda f: f["box"][2]*f["box"][3]) if faces else None

    @staticmethod
    def face_ratio(face, frame):
        _,_,w,h = face["box"]
        return max(w,h)/max(1, min(frame.shape[:2]))

    @staticmethod
    def align_crop(frame, landmarks, size=112):
        dst = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                        [41.5493,92.3655],[70.7299,92.2041]], dtype=np.float32)
        try:
            M,_ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)
            if M is None: return None
            return cv2.warpAffine(frame, M, (size,size), borderValue=0.0)
        except cv2.error: return None

    @staticmethod
    def crop_fallback(frame, face, pad_ratio=0.18):
        x,y,w,h = face["box"]
        pad = int(pad_ratio*max(w,h))
        x1=max(0,x-pad); y1=max(0,y-pad)
        x2=min(frame.shape[1],x+w+pad); y2=min(frame.shape[0],y+h+pad)
        crop = frame[y1:y2,x1:x2]
        return cv2.resize(crop,(112,112)) if crop.size else None

    def crop(self, frame, face):
        c = self.align_crop(frame, face["landmarks"])
        return c if c is not None else self.crop_fallback(frame, face)


def blur_score(img):
    if img is None or img.size == 0: return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def load_db():
    if not DB_PATH.exists(): return {}
    with open(DB_PATH,"r",encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH,"w",encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def open_camera(args):
    if not args.use_usb_camera:
        try:
            from picamera2 import Picamera2
            class PiCamWrap:
                def __init__(self):
                    self.picam2 = Picamera2()
                    cfg = self.picam2.create_preview_configuration(
                        main={"size":(args.width,args.height),"format":"RGB888"},
                        controls={"FrameRate":args.fps})
                    self.picam2.configure(cfg); self.picam2.start(); time.sleep(0.5)
                    print(f"[INFO] Pi Camera {args.width}x{args.height}")
                def isOpened(self): return True
                def read(self):
                    rgb = self.picam2.capture_array()
                    return (True, cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)) if rgb is not None else (False,None)
                def release(self): self.picam2.stop()
            return PiCamWrap()
        except Exception as e:
            print(f"[WARN] Pi Camera loi: {e}")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    print(f"[INFO] OpenCV camera id={args.camera}")
    return cap


# ──────────────────────────────────────────────
# Shared state
# ──────────────────────────────────────────────
_lock = threading.Lock()
_jpeg_frame = b""
_status = {
    "phase": "idle",          # idle | enrolling | done
    "scene_idx": 0,
    "scene_counts": [0]*5,    # embeddings thu được mỗi scenario
    "total_collected": 0,
    "msg": "Chờ bắt đầu",
    "instruction": "Nhập tên và bấm <b>Bắt đầu đăng ký</b>.",
    "ratio": 0.0,
    "blur": 0.0,
    "fps": 0.0,
    "target": 12,
    "name": "",
    "last_log": None,
}
_cmd = {"action": None, "name": ""}   # Flask -> worker


def draw_register_ui(frame, msg, scenario_name, ratio, blur,
                     collected, total_scenarios, bbox=None):
    h, w = frame.shape[:2]
    # top bar
    cv2.rectangle(frame,(0,0),(w,52),(0,0,0),-1)
    color = (0,220,120) if msg=="OK" else (0,160,255)
    cv2.putText(frame, f"[{collected}/{total_scenarios}] {scenario_name}",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,255), 1)
    cv2.putText(frame, f"ratio={ratio:.2f}  blur={blur:.0f}  {msg}",
                (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    if bbox:
        x,y,bw,bh = bbox
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),color,2)


def worker_loop(args):
    """Chạy trong background thread: đọc camera, encode JPEG, và khi được lệnh thì enroll."""
    global _jpeg_frame, _status, _cmd

    cap      = open_camera(args)
    detector = YuNetDetector(score_threshold=args.score_threshold)
    embedder = MobileFaceNet()

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    fps_t0 = time.time(); fps_count = 0; fps_display = 0.0
    enroll_name      = ""
    all_embeddings   = []
    scene_embeddings = []    # embeddings kịch bản hiện tại
    scene_idx        = 0
    collecting       = False
    collect_start    = None
    start_good       = None
    phase            = "idle"
    last_msg         = "Chờ bắt đầu"
    last_ratio       = 0.0
    last_blur        = 0.0
    last_bbox        = None
    scene_counts     = [0]*len(SCENARIOS)

    def reset_enroll():
        nonlocal all_embeddings,scene_embeddings,scene_idx,collecting
        nonlocal collect_start,start_good,scene_counts
        all_embeddings=[]; scene_embeddings=[]; scene_idx=0
        collecting=False; collect_start=None; start_good=None
        scene_counts=[0]*len(SCENARIOS)

    print("[INFO] Worker loop start")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02); continue
        frame = cv2.flip(frame, 1)

        # FPS
        fps_count += 1
        now = time.time()
        if fps_count >= 15:
            fps_display = fps_count/(now - fps_t0)
            fps_t0 = now; fps_count = 0

        # Nhận lệnh từ Flask
        with _lock:
            action = _cmd["action"]
            if action == "start":
                enroll_name = _cmd["name"]
                _cmd["action"] = None
                phase = "enrolling"
                reset_enroll()
                print(f"[ENROLL] Bat dau: {enroll_name}")

        # ── Chế độ preview đơn giản ──
        if phase == "idle" or phase == "done":
            faces = detector.detect(frame)
            face  = detector.largest(faces)
            if face:
                x,y,bw,bh = face["box"]
                cv2.rectangle(frame,(x,y),(x+bw,y+bh),(100,100,100),1)
            cv2.putText(frame,"Nhin vao camera",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(150,150,150),1)
            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if ok:
                with _lock:
                    _jpeg_frame = buf.tobytes()
                    _status["fps"] = round(fps_display,1)
            time.sleep(0.01)
            continue

        # ── Chế độ enroll ──
        sc_name, instruction, expected_range, duration = SCENARIOS[scene_idx]
        faces = detector.detect(frame)
        face  = detector.largest(faces)
        good  = False
        crop  = None

        if face:
            last_bbox  = face["box"]
            last_ratio = detector.face_ratio(face, frame)
            low, high  = expected_range
            crop       = detector.crop(frame, face)
            last_blur  = blur_score(crop)
            if last_ratio < low:        last_msg = "Tiến lại gần"
            elif last_ratio > high:     last_msg = "Lùi ra xa"
            elif last_blur < args.min_blur: last_msg = "Ảnh mờ"
            else:                       last_msg = "OK"; good = True
        else:
            last_bbox=None; last_ratio=0.0; last_blur=0.0; last_msg="Không thấy mặt"

        if good:
            if start_good is None: start_good = now
            if not collecting and now - start_good >= args.stable_time:
                collecting = True; collect_start = now
                with _lock: _status["last_log"] = f"[REC] Bắt đầu thu: {sc_name}"
        else:
            start_good = None
            if not collecting: collect_start = None

        if collecting and crop is not None and good:
            emb = embedder.embed(crop)
            scene_embeddings.append(emb)

        # Hoàn thành kịch bản
        sc_done = (len(scene_embeddings) >= args.target_frames or
                   (collect_start and now - collect_start >= duration))
        if collecting and sc_done:
            # lọc top-K quanh centroid
            arr      = np.array(scene_embeddings, dtype=np.float32)
            centroid = arr.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid)+1e-8)
            sims     = arr @ centroid
            top_k    = max(3, int(len(arr)*0.6))
            top_idx  = np.argsort(sims)[-top_k:]
            selected = [arr[i].tolist() for i in top_idx]
            all_embeddings.extend(selected)
            scene_counts[scene_idx] = len(selected)
            log_msg = f"[SC {scene_idx+1}/{len(SCENARIOS)}] {sc_name}: {len(selected)} emb"
            print(log_msg)

            scene_idx += 1
            scene_embeddings = []
            collecting = False; collect_start = None; start_good = None

            if scene_idx >= len(SCENARIOS):
                # LƯU DB
                db = load_db()
                db[enroll_name] = [e for e in all_embeddings]
                save_db(db)
                print(f"[DONE] {enroll_name}: {len(all_embeddings)} embeddings")
                with _lock:
                    _status.update({
                        "phase":"done","scene_idx":len(SCENARIOS)-1,
                        "scene_counts": scene_counts[:],
                        "total_collected": len(all_embeddings),
                        "msg":"Hoàn thành!",
                        "instruction": f"Đã lưu <b>{len(all_embeddings)}</b> embeddings.",
                        "last_log": f"[DONE] {enroll_name}: {len(all_embeddings)} emb",
                        "name": enroll_name,
                    })
                phase = "done"
                time.sleep(0.01)
                continue
            else:
                with _lock:
                    _status["last_log"] = log_msg

        # Vẽ overlay
        draw_register_ui(frame, last_msg, sc_name,
                         last_ratio, last_blur,
                         scene_idx, len(SCENARIOS), last_bbox)

        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if ok:
            with _lock:
                _jpeg_frame = buf.tobytes()
                _status.update({
                    "phase": "enrolling",
                    "scene_idx": scene_idx,
                    "scene_counts": scene_counts[:],
                    "total_collected": len(all_embeddings) + len(scene_embeddings),
                    "msg": last_msg,
                    "instruction": f"<b>{sc_name}</b> — {instruction}",
                    "ratio": round(last_ratio,3),
                    "blur":  round(last_blur,1),
                    "fps":   round(fps_display,1),
                    "target": args.target_frames,
                    "name": enroll_name,
                    "last_log": None,
                })


# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
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
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with _lock:
        s = dict(_status)
    return jsonify(s)

@app.route("/start", methods=["POST"])
def start():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "msg": "Tên không được rỗng"})
    with _lock:
        if _status["phase"] == "enrolling":
            return jsonify({"ok": False, "msg": "Đang đăng ký, vui lòng đợi"})
        _cmd["action"] = "start"
        _cmd["name"]   = name
        _status["phase"] = "enrolling"
        _status["last_log"] = None
    return jsonify({"ok": True})

@app.route("/db")
def get_db():
    db = load_db()
    return jsonify({k: len(v) for k, v in db.items()})

@app.route("/delete", methods=["POST"])
def delete_person():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "msg": "Thiếu tên"})
    db = load_db()
    if name not in db:
        return jsonify({"ok": False, "msg": f"Không tìm thấy: {name}"})
    del db[name]
    save_db(db)
    print(f"[DEL] Da xoa: {name}")
    return jsonify({"ok": True})


# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera",          type=int,   default=9)
    p.add_argument("--use-usb-camera",  action="store_true")
    p.add_argument("--width",           type=int,   default=640)
    p.add_argument("--height",          type=int,   default=480)
    p.add_argument("--fps",             type=int,   default=15)
    p.add_argument("--target-frames",   type=int,   default=12)
    p.add_argument("--min-blur",        type=float, default=25.0)
    p.add_argument("--stable-time",     type=float, default=0.6)
    p.add_argument("--score-threshold", type=float, default=0.82)
    p.add_argument("--jpeg-quality",    type=int,   default=70)
    p.add_argument("--port",            type=int,   default=5001)
    p.add_argument("--host",            default="0.0.0.0")
    args = p.parse_args()

    t = threading.Thread(target=worker_loop, args=(args,), daemon=True)
    t.start()

    print(f"\n[WEB] Mo browser: http://<PI_IP>:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
