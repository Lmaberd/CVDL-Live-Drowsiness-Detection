#!/usr/bin/env python3
# Panels aligned with boxes (same Y-center). Horizontal leader lines only.
# Panels sit beside boxes with gap; no overlap. Bottom-left status display.
# Drowsy timer (raw prob >= 0.60). Continuous beep after 3s.

import sys, time, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ---------- Optional audio (continuous beeping while drowsy > 3s) ----------
USE_SIMPLEAUDIO = False
try:
    import simpleaudio as sa
    USE_SIMPLEAUDIO = True
except Exception:
    USE_SIMPLEAUDIO = False

def _build_beep_wave(sr=22050, freq=1000, dur=0.15, vol=0.35):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    wave = (vol*np.sin(2*math.pi*freq*t)).astype(np.float32)
    audio = np.int16(np.clip(wave, -1, 1) * 32767)
    return audio.tobytes(), sr

_BEEP_WAV = None
if USE_SIMPLEAUDIO:
    _BEEP_WAV, _BEEP_SR = _build_beep_wave()

def play_beep_nonblocking():
    if USE_SIMPLEAUDIO and _BEEP_WAV is not None:
        try:
            sa.play_buffer(_BEEP_WAV, 1, 2, _BEEP_SR)
        except Exception:
            sys.stdout.write('\a'); sys.stdout.flush()
    else:
        sys.stdout.write('\a'); sys.stdout.flush()

# -------------------- Device --------------------
def get_best_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
device = get_best_device()

# -------------------- Model (matches your notebook’s Multitask model) --------------------
class MultitaskDrowsinessModel(nn.Module):
    def __init__(self, landmark_dim=18):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B,512,1,1]
        self.landmark_fc = nn.Sequential(
            nn.Linear(landmark_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )
        comb = 512 + 32
        self.drowsy_head = nn.Sequential(
            nn.Linear(comb, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2)
        )
        self.eye_head = nn.Sequential(
            nn.Linear(comb, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3)
        )
        self.mouth_head = nn.Sequential(
            nn.Linear(comb, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3)
        )
    def forward(self, x, landmarks):
        f_img = self.backbone(x).flatten(1)
        f_lm  = self.landmark_fc(landmarks)
        z = torch.cat([f_img, f_lm], 1)
        return self.drowsy_head(z), self.eye_head(z), self.mouth_head(z)

# -------------------- Checkpoint --------------------
def find_checkpoint():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "model2_best.pth",
        script_dir / "model2_multitask" / "model2_best.pth",
        script_dir.parent / "model2_multitask" / "model2_best.pth",
        script_dir.parent.parent / "model2_multitask" / "model2_best.pth",
        Path.cwd() / "model2_multitask" / "model2_best.pth",
        Path.cwd() / "model" / "model2_multitask" / "model2_best.pth",
    ]
    for p in candidates:
        if p.exists(): return p
    sys.exit("Checkpoint not found at any expected path.")
def load_model():
    model = MultitaskDrowsinessModel().to(device)
    ckpt_path = find_checkpoint()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✅ Loaded checkpoint: {ckpt_path} on {device}")
    return model

# -------------------- MediaPipe (for landmarks / geometry) --------------------
try:
    import mediapipe as mp
except Exception:
    sys.exit("Please install mediapipe: python3 -m pip install mediapipe")

mp_face_mesh = mp.solutions.face_mesh
face_mesh_full   = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
face_mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True,  max_num_faces=1, min_detection_confidence=0.5)

LM_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
LM_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LM_MOUTH     = [61, 291, 0, 17, 39, 269]
LM_18 = LM_LEFT_EYE + LM_RIGHT_EYE + LM_MOUTH

# -------------------- Transforms --------------------
IMG_SIZE = 224
eval_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# -------------------- Geometry helpers --------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))
def bbox_from_points(pts_px: np.ndarray, pad_scale: float, w: int, h: int):
    x_min, y_min = float(pts_px[:,0].min()), float(pts_px[:,1].min())
    x_max, y_max = float(pts_px[:,0].max()), float(pts_px[:,1].max())
    bw, bh = x_max - x_min, y_max - y_min
    px = (pad_scale - 1.0) * bw / 2.0
    py = (pad_scale - 1.0) * bh / 2.0
    x1, y1 = int(clamp(x_min - px, 0, w-1)), int(clamp(y_min - py, 0, h-1))
    x2, y2 = int(clamp(x_max + px, 1, w)),   int(clamp(y_max + py, 1, h))
    return x1, y1, x2, y2
def union_boxes(b1, b2):
    x1 = min(b1[0], b2[0]); y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2]); y2 = max(b1[3], b2[3])
    return (x1, y1, x2, y2)
def face_bbox_from_all_points(pts_px: np.ndarray, w: int, h: int, scale: float=1.20):
    x_min, y_min = float(pts_px[:,0].min()), float(pts_px[:,1].min())
    x_max, y_max = float(pts_px[:,0].max()), float(pts_px[:,1].max())
    bw, bh = x_max - x_min, y_max - y_min
    cx, cy = x_min + bw/2.0, y_min + bh/2.0
    bw2, bh2 = bw*scale, bh*scale
    x1, y1 = int(clamp(cx - bw2/2, 0, w-1)), int(clamp(cy - bh2/2, 0, h-1))
    x2, y2 = int(clamp(cx + bw2/2, 1, w)),   int(clamp(cy + bh2/2, 1, h))
    return x1, y1, x2, y2

# -------------------- Pre/post --------------------
def extract_18y_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh_static.process(img_rgb)
    if not res.multi_face_landmarks:
        return np.zeros(18, dtype=np.float32)
    lms = res.multi_face_landmarks[0].landmark
    ys = [lms[idx].y for idx in LM_18]
    return np.array(ys, dtype=np.float32)
def prepare_img_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    return eval_tfm(pil).unsqueeze(0).to(device)

# -------------------- Prediction on a frame --------------------
def predict_on_frame(model: nn.Module, frame_bgr: np.ndarray):
    """
    Returns:
      label, pred_id, face_box, confs, region_boxes, region_confs
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh_full.process(rgb)
    if not res.multi_face_landmarks:
        return "No face", None, None, None, None, None

    mesh = res.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * w, lm.y * h] for lm in mesh], dtype=np.float32)

    fx1, fy1, fx2, fy2 = face_bbox_from_all_points(pts, w, h, scale=1.20)
    crop = frame_bgr[fy1:fy2, fx1:fx2]
    if crop.size == 0:
        return "No face", None, None, None, None, None

    left_pts  = pts[LM_LEFT_EYE]
    right_pts = pts[LM_RIGHT_EYE]
    mouth_pts = pts[LM_MOUTH]

    lx1, ly1, lx2, ly2 = bbox_from_points(left_pts,  1.40, w, h)
    rx1, ry1, rx2, ry2 = bbox_from_points(right_pts, 1.40, w, h)
    eyes_box = union_boxes((lx1,ly1,lx2,ly2), (rx1,ry1,rx2,ry2))
    mx1, my1, mx2, my2 = bbox_from_points(mouth_pts,  1.55, w, h)

    region_boxes = {"eyes": eyes_box, "mouth": (mx1, my1, mx2, my2)}

    lm18 = extract_18y_from_crop(crop)
    img_t = prepare_img_tensor(crop)
    lm_t  = torch.from_numpy(lm18).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_d, logits_e, logits_m = model(img_t, lm_t)
        probs_d = F.softmax(logits_d, dim=1)[0].cpu().numpy()  # [drowsy, alert]
        probs_e = F.softmax(logits_e, dim=1)[0].cpu().numpy()  # [closed, open, uncertain]
        probs_m = F.softmax(logits_m, dim=1)[0].cpu().numpy()  # [closed, yawn, uncertain]

    pred_id = int(np.argmax(probs_d))
    label = "Alert" if pred_id == 1 else "Drowsy"

    confs = {"Drowsy": float(probs_d[0]), "Alert": float(probs_d[1])}
    region_confs = {
        "eyes":  {"Eyes Open": float(probs_e[1]), "Eyes Closed": float(probs_e[0])},
        "mouth": {"Yawn": float(probs_m[1]), "Mouth Closed": float(probs_m[0])},
    }
    return label, pred_id, (fx1, fy1, fx2, fy2), confs, region_boxes, region_confs

# -------------------- Visual helpers --------------------
WHITE=(255,255,255); GREEN=(60,200,60); RED=(0,0,230)
CYAN=(255,255,0); MAG=(255,0,180); AMBER=(0,200,255); GREY=(90,90,90); BLACK=(0,0,0)

def draw_box(frame, box, color, th=2):
    x1,y1,x2,y2 = box
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, th)

def draw_label(frame, text, org, color=WHITE, scale=1.0, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def _panel_dims(lines, bar_w=260, bar_h=16, pad=10, text_h=26, gap=10):
    total_h = pad*2 + len(lines)*(text_h + bar_h + gap) - gap
    total_w = bar_w + 2*pad
    return total_w, total_h, pad, text_h, gap, bar_h

def _place_panel_beside_box(frame_w, frame_h, box, prefer_side="right", offset=14, panel_w=260, panel_h=120, margin=8):
    """Place the panel beside the box on the preferred side; flip if no space.
       Keeps same vertical center as the box; clamps to frame; ensures no overlap."""
    x1,y1,x2,y2 = box
    mid_y = (y1 + y2) // 2
    # Try preferred side first
    if prefer_side == "right":
        px1 = x2 + offset
        if px1 + panel_w + margin > frame_w:
            px1 = max(x1 - offset - panel_w, margin)  # flip to left
    else:
        px1 = max(x1 - offset - panel_w, margin)
        if px1 < margin:
            px1 = min(x2 + offset, frame_w - panel_w - margin)  # flip to right

    py1 = int(mid_y - panel_h/2)
    # Clamp vertically
    py1 = max(margin, min(py1, frame_h - panel_h - margin))
    return int(px1), int(py1), int(px1 + panel_w), int(py1 + panel_h), mid_y

def draw_panel_aligned(frame, box, lines, prefer_side="right", offset=14, bar_w=260):
    """
    Draws a compact side panel horizontally aligned with the box's vertical center,
    placed just beside the box (no overlap). Returns the panel rect and the
    horizontal leader line endpoints (for a single straight line).
    """
    H, W = frame.shape[:2]
    panel_w, panel_h, pad, text_h, gap, bar_h = _panel_dims(lines, bar_w=bar_w)
    px1, py1, px2, py2, mid_y = _place_panel_beside_box(W, H, box, prefer_side, offset, panel_w, panel_h)

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), WHITE, 1)

    # Text + bars
    x_text = px1 + pad
    y = py1 + pad + 20
    for label, value, color in lines:
        cv2.putText(frame, f"{label}: {value*100:.1f}%", (x_text, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2)
        yb = y + 8
        # bar background
        cv2.rectangle(frame, (x_text, yb), (px2 - pad, yb + bar_h), (80,80,80), -1)
        # bar fill
        bw = int((px2 - pad - x_text) * float(np.clip(value, 0, 1)))
        cv2.rectangle(frame, (x_text, yb), (x_text + bw, yb + bar_h), color, -1)
        # border
        cv2.rectangle(frame, (x_text, yb), (px2 - pad, yb + bar_h), WHITE, 1)
        y += text_h + bar_h + gap

    # Leader line (horizontal only)
    bx1, by1, bx2, by2 = box
    if px1 >= bx2:  # panel is to the RIGHT of box
        p_src = (bx2, mid_y)
        p_dst = (px1, mid_y)
    else:          # panel is to the LEFT of box
        p_src = (bx1, mid_y)
        p_dst = (px2, mid_y)
    cv2.line(frame, p_src, p_dst, WHITE, 2)

    return (px1, py1, px2, py2)

def draw_status_bottom_left(frame, status, drowsy_time):
    H, _ = frame.shape[:2]
    base_y = H - 40
    color = RED if status == "Drowsy" else GREEN
    cv2.putText(frame, f"State: {status}", (20, base_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"Drowsy Time: {drowsy_time:.1f}s", (20, base_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

def draw_fps(frame, fps):
    draw_label(frame, f"FPS {fps:.1f}", (20, 40), WHITE, 0.9, 2)

# -------------------- Main --------------------
def main():
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Cannot open webcam (index 0).")
    print("🎥 Webcam running. Press 'Q' to quit.")

    drowsy_time = 0.0
    last_beep = 0.0
    BEEP_INTERVAL = 0.25
    BEEP_START_SEC = 3.0
    DROWSY_THRESH = 0.60

    t_last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break

        t_now = time.time()
        dt = t_now - t_last
        t_last = t_now
        if dt > 0:
            fps = 0.9*fps + 0.1*(1.0/dt) if fps > 0 else (1.0/dt)

        label, pred, face_box, confs, regions, rconfs = predict_on_frame(model, frame)

        if pred is None:
            draw_label(frame, "No face", (30, 80), AMBER, 1.1, 2)
            draw_fps(frame, fps)
            draw_status_bottom_left(frame, "Alert", 0.0)
            cv2.imshow("Drowsiness Detection (q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            drowsy_time = 0.0
            continue

        # Raw drowsy prob + simple threshold (no EMA/hysteresis)
        drowsy_prob = confs["Drowsy"]
        active = drowsy_prob >= DROWSY_THRESH
        if active:
            drowsy_time += dt
            if drowsy_time >= BEEP_START_SEC and (t_now - last_beep) >= BEEP_INTERVAL:
                play_beep_nonblocking()
                last_beep = t_now
        else:
            drowsy_time = 0.0

        # Colors and boxes
        color_face = GREEN if pred == 1 else RED
        draw_box(frame, face_box, color_face, 2)
        draw_box(frame, regions["eyes"],  CYAN, 2)
        draw_box(frame, regions["mouth"], MAG,  2)

        # ------- Side panels aligned with boxes (same Y center). Horizontal leaders only. -------
        # Face panel (prefer left), Eyes panel (prefer right), Mouth panel (prefer left)
        draw_panel_aligned(
            frame, face_box,
            [("Alert", 1.0 - drowsy_prob, GREEN), ("Drowsy", drowsy_prob, RED)],
            prefer_side="left", offset=14, bar_w=300
        )
        draw_panel_aligned(
            frame, regions["eyes"],
            [("Eyes Open",  rconfs["eyes"]["Eyes Open"],  CYAN),
             ("Eyes Closed", rconfs["eyes"]["Eyes Closed"], AMBER)],
            prefer_side="right", offset=14, bar_w=300
        )
        draw_panel_aligned(
            frame, regions["mouth"],
            [("Mouth Closed", rconfs["mouth"]["Mouth Closed"], GREEN),
             ("Yawn",          rconfs["mouth"]["Yawn"],         MAG)],
            prefer_side="left", offset=14, bar_w=300
        )

        # Bottom-left status + timer + fps
        draw_status_bottom_left(frame, "Drowsy" if active else "Alert", drowsy_time)
        draw_fps(frame, fps)

        cv2.imshow("Drowsiness Detection (q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Stopped.")

if __name__ == "__main__":
    main()
