#!/usr/bin/env python3
# Pygame FULLSCREEN renderer (no OpenCV window).
# Keeps: aligned side panels + horizontal leader lines + bottom-left state + beep/timer.

import sys, time, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pygame

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

# -------------------- Model --------------------
class MultitaskDrowsinessModel(nn.Module):
    def __init__(self, landmark_dim=18):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
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

def find_checkpoint():
    script_dir = Path(__file__).resolve().parent
    for p in [
        script_dir / "model2_best.pth",
        script_dir / "model2_multitask" / "model2_best.pth",
        script_dir.parent / "model2_multitask" / "model2_best.pth",
        Path.cwd() / "model2_multitask" / "model2_best.pth",
        Path.cwd() / "model" / "model2_multitask" / "model2_best.pth",
    ]:
        if p.exists(): return p
    sys.exit("Checkpoint not found.")
def load_model():
    model = MultitaskDrowsinessModel().to(device)
    ckpt_path = find_checkpoint()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✅ Loaded checkpoint: {ckpt_path} on {device}")
    return model

# -------------------- MediaPipe --------------------
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
def bbox_from_points(pts_px, pad_scale, w, h):
    x_min, y_min = float(pts_px[:,0].min()), float(pts_px[:,1].min())
    x_max, y_max = float(pts_px[:,0].max()), float(pts_px[:,1].max())
    bw, bh = x_max - x_min, y_max - y_min
    px = (pad_scale - 1.0) * bw / 2.0
    py = (pad_scale - 1.0) * bh / 2.0
    x1, y1 = int(clamp(x_min - px, 0, w-1)), int(clamp(y_min - py, 0, h-1))
    x2, y2 = int(clamp(x_max + px, 1, w)),   int(clamp(y_max + py, 1, h))
    return x1, y1, x2, y2
def union_boxes(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
def face_bbox_from_all_points(pts_px, w, h, scale=1.20):
    x_min, y_min = float(pts_px[:,0].min()), float(pts_px[:,1].min())
    x_max, y_max = float(pts_px[:,0].max()), float(pts_px[:,1].max())
    bw, bh = x_max - x_min, y_max - y_min
    cx, cy = x_min + bw/2.0, y_min + bh/2.0
    bw2, bh2 = bw*scale, bh*scale
    x1, y1 = int(clamp(cx - bw2/2, 0, w-1)), int(clamp(cy - bh2/2, 0, h-1))
    x2, y2 = int(clamp(cx + bw2/2, 1, w)),   int(clamp(cy + bh2/2, 1, h))
    return x1, y1, x2, y2

# -------------------- Prediction --------------------
def extract_18y_from_crop(crop_bgr):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh_static.process(img_rgb)
    if not res.multi_face_landmarks:
        return np.zeros(18, dtype=np.float32)
    lms = res.multi_face_landmarks[0].landmark
    ys = [lms[idx].y for idx in LM_18]
    return np.array(ys, dtype=np.float32)
def prepare_img_tensor(crop_bgr):
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    return eval_tfm(pil).unsqueeze(0).to(device)

def predict_on_frame(model, frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh_full.process(rgb)
    if not res.multi_face_landmarks:
        return "No face", None, None, None, None, None
    mesh = res.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * w, lm.y * h] for lm in mesh], dtype=np.float32)
    fx1, fy1, fx2, fy2 = face_bbox_from_all_points(pts, w, h, 1.20)
    crop = frame_bgr[fy1:fy2, fx1:fx2]
    if crop.size == 0: return "No face", None, None, None, None, None
    left_pts, right_pts, mouth_pts = pts[LM_LEFT_EYE], pts[LM_RIGHT_EYE], pts[LM_MOUTH]
    lx1, ly1, lx2, ly2 = bbox_from_points(left_pts, 1.4, w, h)
    rx1, ry1, rx2, ry2 = bbox_from_points(right_pts, 1.4, w, h)
    eyes_box = union_boxes((lx1,ly1,lx2,ly2), (rx1,ry1,rx2,ry2))
    mx1, my1, mx2, my2 = bbox_from_points(mouth_pts, 1.55, w, h)
    region_boxes = {"eyes": eyes_box, "mouth": (mx1,my1,mx2,my2)}
    lm18 = extract_18y_from_crop(crop)
    img_t, lm_t = prepare_img_tensor(crop), torch.from_numpy(lm18).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_d, logits_e, logits_m = model(img_t, lm_t)
        probs_d = F.softmax(logits_d, dim=1)[0].cpu().numpy()
        probs_e = F.softmax(logits_e, dim=1)[0].cpu().numpy()
        probs_m = F.softmax(logits_m, dim=1)[0].cpu().numpy()
    pred_id = int(np.argmax(probs_d))
    label = "Alert" if pred_id == 1 else "Drowsy"
    confs = {"Drowsy": float(probs_d[0]), "Alert": float(probs_d[1])}
    region_confs = {
        "eyes": {"Eyes Open": float(probs_e[1]), "Eyes Closed": float(probs_e[0])},
        "mouth": {"Yawn": float(probs_m[1]), "Mouth Closed": float(probs_m[0])},
    }
    return label, pred_id, (fx1, fy1, fx2, fy2), confs, region_boxes, region_confs

# -------------------- Visual helpers --------------------
WHITE=(255,255,255); GREEN=(60,200,60); RED=(0,0,230)
CYAN=(255,255,0); MAG=(255,0,180); AMBER=(0,200,255)
def draw_box(f, b, c, t=2): cv2.rectangle(f, (b[0],b[1]), (b[2],b[3]), c, t)
def draw_label(f, text, org, color=WHITE, scale=1.0, thick=2):
    cv2.putText(f, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
def draw_status_bottom_left(f, s, d):
    H,_ = f.shape[:2]; base=H-40
    c=RED if s=="Drowsy" else GREEN
    cv2.putText(f,f"State: {s}",(20,base),cv2.FONT_HERSHEY_SIMPLEX,1.2,c,3)
    cv2.putText(f,f"Drowsy Time: {d:.1f}s",(20,base+35),cv2.FONT_HERSHEY_SIMPLEX,1.0,WHITE,2)
def draw_fps(f,fps): draw_label(f,f"FPS {fps:.1f}",(20,40),WHITE,0.9,2)

# -------------------- Panel drawing --------------------
def draw_panel_aligned(frame, box, lines, side="right", offset=14, bar_w=300):
    H,W=frame.shape[:2]; pad=12; text_h=26; gap=10; bar_h=16
    panel_h=pad*2+len(lines)*(text_h+bar_h+gap)-gap; panel_w=bar_w+2*pad
    x1,y1,x2,y2=box; mid_y=(y1+y2)//2
    if side=="right":
        px1=x2+offset
        if px1+panel_w>W: px1=max(x1-offset-panel_w,8)
    else:
        px1=max(x1-offset-panel_w,8)
        if px1<8: px1=min(x2+offset,W-panel_w-8)
    py1=max(8,min(int(mid_y-panel_h/2),H-panel_h-8))
    px2,py2=px1+panel_w,py1+panel_h
    overlay=frame.copy(); cv2.rectangle(overlay,(px1,py1),(px2,py2),(20,20,20),-1)
    cv2.addWeighted(overlay,0.6,frame,0.4,0,frame); cv2.rectangle(frame,(px1,py1),(px2,py2),WHITE,1)
    x_text,y=px1+pad,py1+pad+20
    for label,value,color in lines:
        cv2.putText(frame,f"{label}: {value*100:.1f}%",(x_text,y),cv2.FONT_HERSHEY_SIMPLEX,0.95,WHITE,2)
        yb=y+8
        cv2.rectangle(frame,(x_text,yb),(px2-pad,yb+bar_h),(80,80,80),-1)
        bw=int((px2-pad-x_text)*float(np.clip(value,0,1)))
        cv2.rectangle(frame,(x_text,yb),(x_text+bw,yb+bar_h),color,-1)
        cv2.rectangle(frame,(x_text,yb),(px2-pad,yb+bar_h),WHITE,1)
        y+=text_h+bar_h+gap
    bx1,by1,bx2,by2=box
    p_src=(bx2,mid_y) if px1>=bx2 else (bx1,mid_y)
    p_dst=(px1,mid_y) if px1>=bx2 else (px2,mid_y)
    cv2.line(frame,p_src,p_dst,WHITE,2)

# -------------------- Pygame display helpers --------------------
def scale_fit_surface(surface, target_w, target_h):
    """Scale the camera frame to fit screen exactly (no clipping or grey bars)."""
    sw, sh = surface.get_size()
    s = min(target_w / sw, target_h / sh)
    nw, nh = int(sw * s), int(sh * s)
    scaled = pygame.transform.smoothscale(surface, (nw, nh))
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    final = pygame.Surface((target_w, target_h))
    final.fill((0, 0, 0))
    final.blit(scaled, (x, y))
    return final

# -------------------- Main --------------------
def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit("Cannot open webcam.")
    pygame.init()
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    screen_w, screen_h = screen.get_size()
    clock = pygame.time.Clock()
    drowsy_time=0.0; last_beep=0.0
    BEEP_INTERVAL=0.25; BEEP_START_SEC=3.0; DROWSY_THRESH=0.60
    fps=0.0; t_last=time.time()

    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return
            if e.type==pygame.KEYDOWN and e.key in (pygame.K_q,pygame.K_ESCAPE): return
        ok,frame=cap.read()
        if not ok: break
        work=frame.copy()
        t_now=time.time(); dt=t_now-t_last; t_last=t_now
        if dt>0: fps=0.9*fps+0.1*(1.0/dt) if fps>0 else (1.0/dt)
        label,pred,face_box,confs,regions,rconfs=predict_on_frame(model,work)
        if pred is None:
            draw_label(work,"No face",(30,80),AMBER,1.1,2)
            draw_fps(work,fps); draw_status_bottom_left(work,"Alert",0.0)
        else:
            dprob=confs["Drowsy"]; active=dprob>=DROWSY_THRESH
            if active:
                drowsy_time+=dt
                if drowsy_time>=BEEP_START_SEC and (t_now-last_beep)>=BEEP_INTERVAL:
                    play_beep_nonblocking(); last_beep=t_now
            else: drowsy_time=0.0
            color_face=GREEN if pred==1 else RED
            draw_box(work,face_box,color_face,2)
            draw_box(work,regions["eyes"],CYAN,2)
            draw_box(work,regions["mouth"],MAG,2)
            draw_panel_aligned(work,face_box,[("Alert",1.0-dprob,GREEN),("Drowsy",dprob,RED)],"left",14,300)
            draw_panel_aligned(work,regions["eyes"],[("Eyes Open",rconfs["eyes"]["Eyes Open"],CYAN),("Eyes Closed",rconfs["eyes"]["Eyes Closed"],AMBER)],"right",14,300)
            draw_panel_aligned(work,regions["mouth"],[("Mouth Closed",rconfs["mouth"]["Mouth Closed"],GREEN),("Yawn",rconfs["mouth"]["Yawn"],MAG)],"left",14,300)
            draw_status_bottom_left(work,"Drowsy" if active else "Alert",drowsy_time)
            draw_fps(work,fps)
        rgb=cv2.cvtColor(work,cv2.COLOR_BGR2RGB)
        frame_surf=pygame.image.frombuffer(rgb.tobytes(),(rgb.shape[1],rgb.shape[0]),'RGB')
        fitted=scale_fit_surface(frame_surf,screen_w,screen_h)
        screen.blit(fitted,(0,0)); pygame.display.flip(); clock.tick(60)

    cap.release(); pygame.quit(); print("🛑 Stopped (pygame).")

if __name__ == "__main__":
    main()