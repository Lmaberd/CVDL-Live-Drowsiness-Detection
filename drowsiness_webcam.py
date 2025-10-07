#!/usr/bin/env python3
# Pygame FULLSCREEN renderer (no OpenCV window).
# Compact glass UI + face box (green=Alert, red=Drowsy).

import sys, time, math
from pathlib import Path
import cv2, numpy as np, pygame
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms

# ---------- Optional beep ----------
USE_SIMPLEAUDIO=False
try:
    import simpleaudio as sa  # pip install simpleaudio
    USE_SIMPLEAUDIO=True
except Exception: pass

def _build_beep_wave(sr=22050,freq=1100,dur=0.10,vol=0.35):
    t=np.linspace(0,dur,int(sr*dur),endpoint=False)
    wave=(vol*np.sin(2*math.pi*freq*t)).astype(np.float32)
    audio=np.int16(np.clip(wave,-1,1)*32767)
    return audio.tobytes(),sr
if USE_SIMPLEAUDIO:_BEEP_WAV,_BEEP_SR=_build_beep_wave()
def play_beep_nonblocking():
    if USE_SIMPLEAUDIO:
        try: sa.play_buffer(_BEEP_WAV,1,2,_BEEP_SR)
        except: sys.stdout.write('\a');sys.stdout.flush()
    else: sys.stdout.write('\a');sys.stdout.flush()

# ---------- Device ----------
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model ----------
class MultitaskDrowsinessModel(nn.Module):
    def __init__(self, landmark_dim=18):
        super().__init__()
        resnet=models.resnet18(weights=None)
        self.backbone=nn.Sequential(*list(resnet.children())[:-1])
        self.landmark_fc=nn.Sequential(
            nn.Linear(landmark_dim,64),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(64,32),nn.ReLU())
        comb=512+32
        self.drowsy_head=nn.Sequential(
            nn.Linear(comb,256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,2))
        self.eye_head=nn.Sequential(
            nn.Linear(comb,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,3))
        self.mouth_head=nn.Sequential(
            nn.Linear(comb,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,3))
    def forward(self,x,lm):
        f_img=self.backbone(x).flatten(1)
        f_lm=self.landmark_fc(lm)
        z=torch.cat([f_img,f_lm],1)
        return self.drowsy_head(z),self.eye_head(z),self.mouth_head(z)

def find_checkpoint():
    script=Path(__file__).resolve().parent
    for p in [
        script/"model2_best.pth",
        script/"model2_multitask"/"model2_best.pth",
        script.parent/"model2_multitask"/"model2_best.pth",
        Path.cwd()/"model2_multitask"/"model2_best.pth",
        Path.cwd()/"model"/"model2_multitask"/"model2_best.pth",
    ]:
        if p.exists(): return p
    sys.exit("Checkpoint not found.")

def load_model():
    m=MultitaskDrowsinessModel().to(device)
    ck=find_checkpoint()
    state=torch.load(ck,map_location=device)
    m.load_state_dict(state,strict=True)
    m.eval()
    print(f"✅ Loaded checkpoint: {ck}")
    return m

# ---------- MediaPipe ----------
try:
    import mediapipe as mp
except Exception:
    sys.exit("Please install mediapipe: pip install mediapipe")
mp_face_mesh=mp.solutions.face_mesh
face_mesh_full=mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=0.5)
face_mesh_static=mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5)
LM_LEFT_EYE=[33,160,158,133,153,144]
LM_RIGHT_EYE=[362,385,387,263,373,380]
LM_MOUTH=[61,291,0,17,39,269]
LM_18=LM_LEFT_EYE+LM_RIGHT_EYE+LM_MOUTH

# ---------- Transforms ----------
IMG_SIZE=224
eval_tfm=transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------- Colors / UI scales ----------
WHITE=(245,245,245);GREEN=(60,200,60);RED=(235,80,80)
CYAN=(255,255,80);MAG=(255,100,200);AMBER=(60,205,255)
GLASS=(24,24,28);BORDER=(85,85,90);SHADOW=(0,0,0)
TITLE_SCALE=0.46;LABEL_SCALE=0.40;VALUE_SCALE=0.38
SIDEBAR_W=180;CARD_GAP=6;CARD_PAD=8;BAR_H=7

# ---------- Helpers ----------
def extract_18y_from_crop(crop_bgr):
    img_rgb=cv2.cvtColor(crop_bgr,cv2.COLOR_BGR2RGB)
    res=face_mesh_static.process(img_rgb)
    if not res.multi_face_landmarks: return np.zeros(18,dtype=np.float32)
    lms=res.multi_face_landmarks[0].landmark
    ys=[lms[idx].y for idx in LM_18];return np.array(ys,dtype=np.float32)

def prepare_img_tensor(crop):
    pil=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
    return eval_tfm(pil).unsqueeze(0).to(device)

def face_bbox_from_all_points(pts,w,h,scale=1.2):
    x_min,y_min=float(pts[:,0].min()),float(pts[:,1].min())
    x_max,y_max=float(pts[:,0].max()),float(pts[:,1].max())
    bw,bh=x_max-x_min,y_max-y_min;cx,cy=x_min+bw/2,y_min+bh/2
    bw2,bh2=bw*scale,bh*scale
    x1,y1=int(max(cx-bw2/2,0)),int(max(cy-bh2/2,0))
    x2,y2=int(min(cx+bw2/2,w)),int(min(cy+bh2/2,h))
    return x1,y1,x2,y2

def rounded_rect(img,tl,br,color,radius=7):
    x1,y1=tl;x2,y2=br;r=max(2,min(radius,min(x2-x1,y2-y1)//4))
    overlay=img.copy()
    cv2.rectangle(overlay,(x1+r,y1),(x2-r,y2),color,-1)
    cv2.rectangle(overlay,(x1,y1+r),(x2,y2-r),color,-1)
    for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(overlay,(cx,cy),r,color,-1)
    return overlay

def draw_card(frame,x,y,w,h,title):
    shadow=frame.copy()
    cv2.rectangle(shadow,(x+2,y+2),(x+w+2,y+h+2),SHADOW,-1)
    cv2.addWeighted(shadow,0.22,frame,0.78,0,frame)
    overlay=rounded_rect(frame,(x,y),(x+w,y+h),GLASS,8)
    cv2.addWeighted(overlay,0.84,frame,0.16,0,frame)
    cv2.rectangle(frame,(x,y),(x+w,y+h),BORDER,1)
    cv2.putText(frame,title,(x+CARD_PAD,y+CARD_PAD+14),
                cv2.FONT_HERSHEY_SIMPLEX,TITLE_SCALE,WHITE,1,cv2.LINE_AA)

def bar_line(frame,x,y,w,label,val,color):
    cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,LABEL_SCALE,WHITE,1,cv2.LINE_AA)
    yb=y+3
    cv2.rectangle(frame,(x,yb),(x+w,yb+BAR_H),(70,70,70),-1)
    bw=int(w*np.clip(val,0,1))
    cv2.rectangle(frame,(x,yb),(x+bw,yb+BAR_H),color,-1)
    cv2.rectangle(frame,(x,yb),(x+w,yb+BAR_H),(105,105,110),1)
    cv2.putText(frame,f"{val*100:.1f}%",(x+w-48,y+BAR_H+10),
                cv2.FONT_HERSHEY_SIMPLEX,VALUE_SCALE,WHITE,1,cv2.LINE_AA)

# ---------- Prediction ----------
def predict_on_frame(model,frame):
    h,w=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=face_mesh_full.process(rgb)
    if not res.multi_face_landmarks:
        return "No face",None,None,{"Drowsy":0.0,"Alert":1.0},{
            "eyes":{"Eyes Open":0.0,"Eyes Closed":0.0,"Uncertain":0.0},
            "mouth":{"Mouth Closed":0.0,"Yawn":0.0}}
    mesh=res.multi_face_landmarks[0].landmark
    pts=np.array([[lm.x*w,lm.y*h] for lm in mesh],dtype=np.float32)
    x1,y1,x2,y2=face_bbox_from_all_points(pts,w,h)
    crop=frame[y1:y2,x1:x2]
    if crop.size==0: return "No face",None,(x1,y1,x2,y2),{"Drowsy":0.0,"Alert":1.0},{
        "eyes":{"Eyes Open":0.0,"Eyes Closed":0.0,"Uncertain":0.0},
        "mouth":{"Mouth Closed":0.0,"Yawn":0.0}}
    lm18=extract_18y_from_crop(crop)
    img_t,lm_t=prepare_img_tensor(crop),torch.from_numpy(lm18).unsqueeze(0).to(device)
    with torch.no_grad():
        d,e,m=model(img_t,lm_t)
        pd,pe,pm=F.softmax(d,1)[0].cpu().numpy(),F.softmax(e,1)[0].cpu().numpy(),F.softmax(m,1)[0].cpu().numpy()
    confs={"Drowsy":float(pd[0]),"Alert":float(pd[1])}
    rconfs={"eyes":{"Eyes Open":float(pe[1]),"Eyes Closed":float(pe[0]),"Uncertain":float(pe[2])},
            "mouth":{"Mouth Closed":float(pm[0]),"Yawn":float(pm[1])}}
    pred="Alert" if np.argmax(pd)==1 else "Drowsy"
    return pred,(x1,y1,x2,y2),confs,rconfs

# ---------- Sidebar ----------
def draw_sidebar(frame,state,dt,fps,confs,rconfs):
    x0,y=8,8;w=SIDEBAR_W
    h=56;draw_card(frame,x0,y,w,h,"Status")
    c=RED if state=="Drowsy" else (AMBER if state=="No face" else GREEN)
    cv2.putText(frame,state,(x0+CARD_PAD,y+CARD_PAD+30),cv2.FONT_HERSHEY_SIMPLEX,0.52,c,2)
    cv2.putText(frame,f"Time {dt:.1f}s",(x0+CARD_PAD,y+CARD_PAD+46),cv2.FONT_HERSHEY_SIMPLEX,0.40,WHITE,1)
    y+=h+CARD_GAP
    h=82;draw_card(frame,x0,y,w,h,"Face")
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+22,w-2*CARD_PAD,"Alert",confs["Alert"],GREEN)
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+44,w-2*CARD_PAD,"Drowsy",confs["Drowsy"],RED)
    y+=h+CARD_GAP
    h=104;draw_card(frame,x0,y,w,h,"Eyes")
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+22,w-2*CARD_PAD,"Open",rconfs["eyes"]["Eyes Open"],CYAN)
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+44,w-2*CARD_PAD,"Closed",rconfs["eyes"]["Eyes Closed"],AMBER)
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+66,w-2*CARD_PAD,"Uncertain",rconfs["eyes"]["Uncertain"],WHITE)
    y+=h+CARD_GAP
    h=82;draw_card(frame,x0,y,w,h,"Mouth")
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+22,w-2*CARD_PAD,"Closed",rconfs["mouth"]["Mouth Closed"],GREEN)
    bar_line(frame,x0+CARD_PAD,y+CARD_PAD+44,w-2*CARD_PAD,"Yawn",rconfs["mouth"]["Yawn"],MAG)
    y+=h+CARD_GAP
    h=46;draw_card(frame,x0,y,w,h,"Performance")
    cv2.putText(frame,f"FPS {fps:.1f}",(x0+CARD_PAD,y+CARD_PAD+26),
                cv2.FONT_HERSHEY_SIMPLEX,0.48,WHITE,2)

# ---------- Pygame ----------
def scale_fit_surface(surface,W,H):
    sw,sh=surface.get_size();s=min(W/sw,H/sh)
    nw,nh=int(sw*s),int(sh*s)
    scaled=pygame.transform.smoothscale(surface,(nw,nh))
    x=(W-nw)//2;y=(H-nh)//2
    final=pygame.Surface((W,H))
    final.fill((0,0,0))
    final.blit(scaled,(x,y))
    return final

# ---------- Main ----------
def main():
    model=load_model()
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit("No webcam.")
    pygame.init();screen=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    W,H=screen.get_size();clock=pygame.time.Clock()
    dt,last_beep=0.0,0.0;BEEP_INTERVAL=0.25;BEEP_START_SEC=3.0
    DROWSY_THRESH=0.60;CONF_MARGIN=0.15;fps=0.0;t_last=time.time()

    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: return
            if e.type==pygame.KEYDOWN and e.key in (pygame.K_q,pygame.K_ESCAPE): return
        ok,frame=cap.read()
        if not ok: break
        work=frame.copy()
        t_now=time.time();dt_frame=t_now-t_last;t_last=t_now
        if dt_frame>0: fps=0.9*fps+0.1*(1.0/dt_frame) if fps>0 else (1.0/dt_frame)
        label,face_box,confs,rconfs=predict_on_frame(model,work)

        if face_box is not None:
            x1,y1,x2,y2=face_box
            color=GREEN if label=="Alert" else RED
            cv2.rectangle(work,(x1,y1),(x2,y2),color,2)

        if label=="No face":
            state="No face";dt=0.0
        else:
            dprob=confs["Drowsy"];margin=abs(confs["Drowsy"]-confs["Alert"])
            confident=(margin>=CONF_MARGIN);active=(dprob>=DROWSY_THRESH and confident)
            if active:
                dt+=dt_frame
                if dt>=BEEP_START_SEC and (t_now-last_beep)>=BEEP_INTERVAL:
                    play_beep_nonblocking();last_beep=t_now
            else: dt=0.0
            state="Drowsy" if active else "Alert"

        draw_sidebar(work,state,dt,fps,confs,rconfs)
        rgb=cv2.cvtColor(work,cv2.COLOR_BGR2RGB)
        surf=pygame.image.frombuffer(rgb.tobytes(),(rgb.shape[1],rgb.shape[0]),'RGB')
        fitted=scale_fit_surface(surf,W,H)
        screen.blit(fitted,(0,0));pygame.display.flip();clock.tick(60)

    cap.release();pygame.quit();print("🛑 Stopped.")

if __name__=="__main__": main()
