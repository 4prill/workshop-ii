"""
evaluate_web.py - Web version for Flask Video Streaming
Adapted from evaluate.py for browser-based assessment.
"""

import cv2
import numpy as np
import math
import time
import pandas as pd
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# ── YOLOv8 Landmark indices ─────────────────────────────
LM = {
    "NOSE": 0,
    "LEFT_SHOULDER": 5,   "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW":    7,   "RIGHT_ELBOW":    8,
    "LEFT_WRIST":    9,   "RIGHT_WRIST":    10,
    "LEFT_HIP":      11,  "RIGHT_HIP":      12,
    "LEFT_KNEE":     13,  "RIGHT_KNEE":     14,
    "LEFT_ANKLE":    15,  "RIGHT_ANKLE":    16,
}

class DummyLM:
    def __init__(self, x, y, conf):
        self.x = x; self.y = y; self.z = 0.0; self.visibility = conf

C = {
    "green":  (60, 200, 80),  "yellow": (0,  210, 230), "orange": (0,  165, 255),
    "red":    (50,  60, 220), "white":  (240, 240, 240), "black":  (20,  20,  20),
    "gray":   (130, 130, 130),"blue":   (210, 140,  40), "panel":  (18,  18,  18),
    "cyan":   (255, 255, 0),  "magenta": (255, 50, 255) 
}

SKELETON = [
    (LM["LEFT_SHOULDER"],  LM["RIGHT_SHOULDER"]), (LM["LEFT_SHOULDER"],  LM["LEFT_ELBOW"]),
    (LM["LEFT_ELBOW"],     LM["LEFT_WRIST"]),     (LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"]),
    (LM["RIGHT_ELBOW"],    LM["RIGHT_WRIST"]),    (LM["LEFT_SHOULDER"],  LM["LEFT_HIP"]),
    (LM["RIGHT_SHOULDER"], LM["RIGHT_HIP"]),      (LM["LEFT_HIP"],       LM["RIGHT_HIP"]),
    (LM["LEFT_HIP"],       LM["LEFT_KNEE"]),      (LM["LEFT_KNEE"],      LM["LEFT_ANKLE"]),
    (LM["RIGHT_HIP"],      LM["RIGHT_KNEE"]),     (LM["RIGHT_KNEE"],     LM["RIGHT_ANKLE"]),
]

# 🚀 ค่าคงที่สำหรับนับรอบและประเมินความเสี่ยง
BEND_START  = 18.0  
STAND_BACK  = 16.0  
BAD_POSTURE = 45.0  
MIN_FRAMES  = 3     

class Smoother:
    def __init__(self, n=20, ema_alpha=0.15, deadband=0.0):
        self.buf = deque(maxlen=n); self.alpha = ema_alpha; self.deadband = deadband; self._ema = None
    def update(self, v):
        if v is None: return self._ema
        if self._ema is not None and abs(v - self._ema) < self.deadband: return self._ema
        self.buf.append(v); median = float(np.median(self.buf))
        if self._ema is None: self._ema = median
        else: self._ema = self.alpha * median + (1.0 - self.alpha) * self._ema
        return self._ema

def get_spine_angle(lms, frame_w, frame_h):
    ls, rs = lms[LM["LEFT_SHOULDER"]], lms[LM["RIGHT_SHOULDER"]]
    lh, rh = lms[LM["LEFT_HIP"]], lms[LM["RIGHT_HIP"]]
    
    if (ls.visibility + lh.visibility) > (rs.visibility + rh.visibility):
        sx, sy = ls.x * frame_w, ls.y * frame_h
        hx, hy = lh.x * frame_w, lh.y * frame_h
    else:
        sx, sy = rs.x * frame_w, rs.y * frame_h
        hx, hy = rh.x * frame_w, rh.y * frame_h
        
    dx = sx - hx
    dy = hy - sy
    
    if dx == 0 and dy == 0: return 0.0
    return abs(math.degrees(math.atan2(dx, dy)))

def get_step2_score(wrist_y, shoulder_y, hip_y, knee_y, H_cm):
    if H_cm is None: return 0, "Unknown Zone"
    H_in = H_cm / 2.54
    if H_in <= 7:   zx, zx_lbl = 0, "Close (<7in)"
    elif H_in <= 12: zx, zx_lbl = 1, "Moderate (7-12in)"
    else:            zx, zx_lbl = 2, "Reach (>12in)"

    if shoulder_y is None: shoulder_y = 0.2
    if hip_y is None:      hip_y = shoulder_y + 0.3
    if knee_y is None:     knee_y = hip_y + 0.25
    if wrist_y is None:    return 0, "Unknown Zone"

    if wrist_y < shoulder_y: zy_lbl, scores = "Above Shoulder", [29, 18, 14]
    elif wrist_y < hip_y:    zy_lbl, scores = "Waist to Shoulder", [32, 23, 18]
    elif wrist_y < knee_y:   zy_lbl, scores = "Knee to Waist", [41, 25, 18]
    else:                    zy_lbl, scores = "Below Knee", [31, 23, 16]
    return scores[zx], f"{zy_lbl} / {zx_lbl}"

def get_step3_multiplier(lpm, duration_mode):
    if lpm < 0.2:    return 1.0  
    elif lpm <= 0.6: return [1.0, 0.95, 0.85][duration_mode] 
    elif lpm <= 1.5: return [0.95, 0.9, 0.75][duration_mode] 
    elif lpm <= 3.5: return [0.9, 0.85, 0.65][duration_mode] 
    elif lpm <= 5.5: return [0.85, 0.7, 0.45][duration_mode] 
    elif lpm <= 7.5: return [0.75, 0.5, 0.25][duration_mode] 
    elif lpm <= 9.5: return [0.6, 0.35, 0.15][duration_mode] 
    else:            return [0.3, 0.2, 0.0][duration_mode]   

def get_step4_multiplier(rot_deg):
    if rot_deg is None: return 1.0
    return 0.85 if rot_deg >= 45.0 else 1.0

def get_posture_multiplier(spine_deg):
    if spine_deg is None: return 1.0
    if spine_deg >= BAD_POSTURE:
        return 0.85
    else:
        return 1.0

def assess_risk(base_score, freq_mult, rot_mult, posture_mult, weight_kg):
    if base_score == 0: return 0.0, 0.0, 1.0, 1, "WAITING", C["gray"]
    
    # แบบฟอร์มขั้นตอนที่ 5 ของแบบประเมิน
    rwl = base_score * freq_mult * rot_mult
    
    if weight_kg is None or weight_kg <= 0: return rwl, 0.0, rot_mult, 1, "WAITING WGT", C["gray"]
    
    lhi = (weight_kg / max(rwl, 0.1)) * 100
    
    if lhi < 50: 
        return rwl, lhi, rot_mult, 1, "Level 1: Acceptable", C["green"]
    elif lhi <= 75: 
        return rwl, lhi, rot_mult, 2, "Level 2: Monitor", C["yellow"]
    elif lhi <= 100: 
        return rwl, lhi, rot_mult, 3, "Level 3: Investigate", C["orange"]
    else: 
        return rwl, lhi, rot_mult, 4, "Level 4: Fix Now!", C["red"]

def extract_features(landmarks, frame_w, frame_h, user_h_cm, current_ppc=None):
    lms = landmarks
    def get(name):
        idx = LM[name]; lm = lms[idx]
        if getattr(lm, "visibility", 1.0) < 0.3: return None
        return int(lm.x * frame_w), int(lm.y * frame_h), lm.x, lm.y

    ls, rs = get("LEFT_SHOULDER"), get("RIGHT_SHOULDER")
    lh, rh = get("LEFT_HIP"), get("RIGHT_HIP")
    la, ra = get("LEFT_ANKLE"), get("RIGHT_ANKLE")
    lk, rk = get("LEFT_KNEE"), get("RIGHT_KNEE")
    nose = get("NOSE")
    lw, rw = get("LEFT_WRIST"), get("RIGHT_WRIST")

    feats = {}
    def avg_y(l1, l2):
        if l1 and l2: return (l1[3] + l2[3]) / 2.0
        return l1[3] if l1 else (l2[3] if l2 else None)
        
    feats["shoulder_y"] = avg_y(ls, rs)
    feats["hip_y"]      = avg_y(lh, rh)
    feats["knee_y"]     = avg_y(lk, rk)
    feats["wrist_y"]    = avg_y(lw, rw)

    raw_ppc = None
    if nose and (la or ra):
        ankle = la if (la and (ra is None or la[1] > ra[1])) else ra
        if ankle:
            px_h = math.hypot(ankle[0]-nose[0], ankle[1]-nose[1])
            if px_h > 20: raw_ppc = px_h / user_h_cm
    feats["raw_ppc"] = raw_ppc
    active_ppc = current_ppc if current_ppc else raw_ppc

    rot_2d = 0.0
    if ls and rs and lh and rh:
        shoulder_angle = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
        hip_angle = math.degrees(math.atan2(rh[1] - lh[1], rh[0] - lh[0]))
        twist_diff = abs(shoulder_angle - hip_angle)
        if twist_diff > 180: twist_diff = 360 - twist_diff
            
        reach_twist = 0.0
        mid_hip_tmp = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
        mid_wrist_tmp = ((lw[0]+rw[0])//2, (lw[1]+rw[1])//2) if lw and rw else None
        
        if mid_wrist_tmp and nose:
            facing_right = nose[0] > mid_hip_tmp[0]
            dx_hand = mid_wrist_tmp[0] - mid_hip_tmp[0]
            if (facing_right and dx_hand < -10) or (not facing_right and dx_hand > 10):
                sh_width = max(abs(ls[0] - rs[0]), 20.0)
                reach_twist = min((abs(dx_hand) / sh_width) * 45.0, 90.0)
        rot_2d = max(twist_diff, reach_twist)

    feats["rotation_deg"] = rot_2d

    mid_hip = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2) if lh and rh else (lh[:2] if lh else (rh[:2] if rh else None))
    mid_wrist = ((lw[0]+rw[0])//2, (lw[1]+rw[1])//2) if lw and rw else (lw[:2] if lw else (rw[:2] if rw else None))
    mid_shoulder = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2) if ls and rs else (ls[:2] if ls else (rs[:2] if rs else None))
    
    feats["mid_hip"] = mid_hip
    feats["mid_wrist"] = mid_wrist
    feats["mid_shoulder"] = mid_shoulder 

    if mid_hip and mid_shoulder and active_ppc:
        direction = 1
        if nose: direction = 1 if nose[0] >= mid_hip[0] else -1
        elif mid_wrist: direction = 1 if mid_wrist[0] >= mid_hip[0] else -1
            
        belly_offset_px = 12.0 * active_ppc 
        belly_x = int(((mid_shoulder[0] + mid_hip[0]) / 2) + (direction * belly_offset_px))
        belly_y = int(mid_shoulder[1] * 0.4 + mid_hip[1] * 0.6)
        belly = (belly_x, belly_y)
    else:
        belly = mid_hip 
        
    feats["belly"] = belly

    raw_h = None
    if belly and mid_wrist and active_ppc:
        raw_h = abs(mid_wrist[0] - belly[0]) / active_ppc
    feats["raw_h"] = raw_h

    return feats

def blend_rect(frame, x1, y1, x2, y2, color, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def put_text(frame, text, pos, scale=0.55, color=C["white"], thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, C["black"], thickness+2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_body_axes(frame, feats, H_cm, posture_mult):
    mid_shoulder = feats.get("mid_shoulder")
    mid_hip = feats.get("mid_hip")
    mid_wrist = feats.get("mid_wrist")
    belly = feats.get("belly")

    if mid_shoulder and mid_hip:
        sx, sy = int(mid_shoulder[0]), int(mid_shoulder[1])
        hx, hy = int(mid_hip[0]), int(mid_hip[1])
        spine_color = C["red"] if posture_mult < 1.0 else C["magenta"]
        cv2.line(frame, (sx, sy), (hx, hy), spine_color, 4, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), 6, C["yellow"], -1) 
        cv2.circle(frame, (hx, hy), 6, C["yellow"], -1) 

    if belly and mid_wrist:
        body_x, body_y = int(belly[0]), int(belly[1])
        wrist_x, wrist_y = int(mid_wrist[0]), int(mid_wrist[1])
        
        cv2.line(frame, (body_x, body_y - 40), (body_x, wrist_y + 40), C["orange"], 1, cv2.LINE_AA)
        cv2.line(frame, (body_x, wrist_y), (wrist_x, wrist_y), C["cyan"], 3, cv2.LINE_AA)
        cv2.circle(frame, (body_x, wrist_y), 5, C["cyan"], -1)
        cv2.circle(frame, (wrist_x, wrist_y), 8, C["green"], -1)

        if H_cm is not None:
            mid_line_x = (body_x + wrist_x) // 2
            put_text(frame, f"H: {H_cm/2.54:.1f} inch", (mid_line_x - 30, wrist_y - 10), 0.55, C["cyan"], 2)

def draw_skeleton(frame, landmarks, frame_w, frame_h, trunk_deg):
    pts = {}
    for name, idx in LM.items():
        lm = landmarks[idx]
        if getattr(lm, "visibility", 1.0) > 0.3:
            pts[idx] = (int(lm.x * frame_w), int(lm.y * frame_h))
            
    col = C["red"] if trunk_deg >= BAD_POSTURE else C["orange"] if trunk_deg > BEND_START else C["green"]

    for i, j in SKELETON:
        if i in pts and j in pts: 
            cv2.line(frame, pts[i], pts[j], col, 2, cv2.LINE_AA)
    for px, py in pts.values():
        cv2.circle(frame, (px, py), 4, C["white"], -1)

def draw_dashboard(frame, feats, risk_label, risk_col,
                   rwl_kg, lhi_val, weight_kg, fps, frame_h, frame_w, base_score, zone_label,
                   rep_count, lpm, dyn_freq_mult, rot_mult, posture_mult, duration_str, spine_deg):
    PW = 310   
    px = frame_w - PW
    blend_rect(frame, px, 0, frame_w, frame_h, C["panel"], 0.9)
    cv2.line(frame, (px, 0), (px, frame_h), (60, 60, 60), 1)

    y = 20
    put_text(frame, "ERGONOMIC RISK ASSESSMENT", (px+10, y), 0.55, C["white"], 1); y += 22
    cv2.line(frame, (px+10, y), (frame_w-10, y), (70, 70, 70), 1); y += 12

    put_text(frame, f"[STEP 2] BASE SCORE", (px+10, y), 0.45, C["cyan"]); y += 18
    put_text(frame, f"Zone: {zone_label}", (px+20, y), 0.4, C["white"]); y += 16
    put_text(frame, f"-> Base Value = {base_score} kg", (px+20, y), 0.4, C["green"]); y += 16

    put_text(frame, f"[STEP 3] FREQUENCY ({duration_str})", (px+10, y), 0.45, C["cyan"]); y += 18
    put_text(frame, f"Reps: {rep_count} | Speed: {lpm:.1f} /min", (px+20, y), 0.4, C["white"]); y += 16
    put_text(frame, f"-> Freq Mult = {dyn_freq_mult:.2f}", (px+20, y), 0.4, C["green"]); y += 16

    put_text(frame, f"[STEP 4] ROTATION", (px+10, y), 0.45, C["cyan"]); y += 18
    rot_val = feats.get("rotation_deg") or 0.0
    put_text(frame, f"Twist Angle: {rot_val:.1f} deg", (px+20, y), 0.4, C["white"]); y += 16
    put_text(frame, f"-> Rot Mult = {rot_mult:.2f}", (px+20, y), 0.4, C["green"]); y += 16

    put_text(frame, f"[NEW] POSTURE / BENDING", (px+10, y), 0.45, C["cyan"]); y += 18
    posture_color = C["red"] if posture_mult < 1.0 else C["white"]
    put_text(frame, f"Spine Angle: {spine_deg:.1f} deg", (px+20, y), 0.4, posture_color); y += 16
    put_text(frame, f"-> Posture Mult = {posture_mult:.2f}", (px+20, y), 0.4, C["green"] if posture_mult==1.0 else C["red"]); y += 20

    cv2.line(frame, (px+10, y), (frame_w-10, y), (70, 70, 70), 1); y += 18

    put_text(frame, "FINAL SCORE (RWL)", (px+10, y), 0.55, C["yellow"], 1); y += 22
    put_text(frame, f"RWL = Step2 x Step3 x Step4", (px+10, y), 0.35, C["gray"]); y += 16
    put_text(frame, f"RWL = {base_score} x {dyn_freq_mult:.2f} x {rot_mult:.2f} = {base_score * dyn_freq_mult * rot_mult:.1f}", (px+10, y), 0.40, C["white"]); y += 20
    
    if rwl_kg is not None and rwl_kg > 0:
        put_text(frame, f"RWL Limit : {rwl_kg:.1f} kg", (px+10, y), 0.55, C["yellow"], 1); y += 20
        put_text(frame, f"Real Load : {weight_kg} kg", (px+10, y), 0.50, C["white"]); y += 25
        
        badge_h = 40
        blend_rect(frame, px+10, y, frame_w-10, y+badge_h, risk_col, 0.4)
        cv2.rectangle(frame, (px+10, y), (frame_w-10, y+badge_h), risk_col, 2)
        put_text(frame, risk_label, (px+15, y+25), 0.50, C["white"], 1); y += badge_h + 15
        
        put_text(frame, f"Risk Index (LHI) = {lhi_val:.1f}%", (px+10, y), 0.5, risk_col, 1); y += 20
    else:
        put_text(frame, "Waiting for evaluation...", (px+10, y), 0.45, C["gray"]); y += 40

    put_text(frame, f"FPS: {fps:.1f}", (px+10, frame_h-12), 0.38, C["gray"])

# ── Generator Function สำหรับส่งภาพให้หน้าเว็บ ───────────────────────────
def generate_frames(source=0, user_h_cm=170.0, weight_kg=15.0, duration_mode=0, duration_str="< 1 hr", yolo_model=None):
    if yolo_model is None:
        yolo_model = YOLO("yolov8n-pose.pt")
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Camera not accessible.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    PANEL_W = 310
    out_w = frame_w + PANEL_W

    sm_ppc = Smoother(n=60, ema_alpha=0.05, deadband=0.0)
    sm_H = Smoother(n=15, ema_alpha=0.15, deadband=1.0)
    
    sm_wy = Smoother(n=5, ema_alpha=0.4); sm_sy = Smoother(n=5, ema_alpha=0.4)
    sm_hy = Smoother(n=5, ema_alpha=0.4); sm_ky = Smoother(n=5, ema_alpha=0.4)
    sm_spine = Smoother(n=3, ema_alpha=0.6) 

    t_prev = time.time()
    fps_disp = 0.0
    script_start_time = time.time()
    
    body_state = "STAND"
    sequences = []
    rep_count = 0
    bend_count = 0
    last_rep_time = 0.0 
    
    lift_times = deque() 
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        t_now = time.time()
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        results = yolo_model(frame, verbose=False, device='cpu')
        
        canvas = np.zeros((frame_h, out_w, 3), dtype=np.uint8)
        canvas[:, :frame_w] = frame

        if results and len(results[0].keypoints) > 0:
            kpts = results[0].keypoints
            if kpts.xyn.shape[0] > 0 and kpts.xyn.shape[1] >= 17:
                kpts_norm, confs = kpts.xyn[0].cpu().numpy(), kpts.conf[0].cpu().numpy()
                lms = [DummyLM(kpts_norm[i][0], kpts_norm[i][1], confs[i]) for i in range(17)]
                
                feats = extract_features(lms, frame_w, frame_h, user_h_cm, current_ppc=None)
                stable_ppc = sm_ppc.update(feats.get("raw_ppc"))
                feats = extract_features(lms, frame_w, frame_h, user_h_cm, current_ppc=stable_ppc)

                if lms[LM["NOSE"]].visibility > 0.2:
                    H_cm = sm_H.update(feats.get("raw_h"))
                    feats["H_cm"] = H_cm 
                    
                    wy = sm_wy.update(feats.get("wrist_y"))
                    sy = sm_sy.update(feats.get("shoulder_y"))
                    hy = sm_hy.update(feats.get("hip_y"))
                    ky = sm_ky.update(feats.get("knee_y"))

                    raw_spine = get_spine_angle(lms, frame_w, frame_h)
                    spine_deg = sm_spine.update(raw_spine)

                    if body_state == "STAND":
                        if spine_deg > BEND_START:
                            body_state = "BENDING"
                            sequences = []
                    elif body_state == "BENDING":
                        sequences.append(spine_deg)
                        if spine_deg < STAND_BACK:
                            if len(sequences) >= MIN_FRAMES:
                                if time.time() - last_rep_time > 1.5:
                                    bend_count += 1
                                    last_rep_time = time.time()
                                    
                                    if bend_count % 2 != 0:
                                        rep_count += 1
                                        lift_times.append(time.time())
                                        
                            body_state = "STAND"

                    curr_time = time.time()
                    while len(lift_times) > 0 and curr_time - lift_times[0] > 300:
                        lift_times.popleft()
                    
                    elapsed_minutes = min(max(curr_time - script_start_time, 60.0), 300.0) / 60.0
                    lpm = rep_count / elapsed_minutes

                    dyn_freq_mult = get_step3_multiplier(lpm, duration_mode)
                    rot_mult = get_step4_multiplier(feats.get("rotation_deg"))
                    posture_mult_live = get_posture_multiplier(spine_deg)

                    base_score, zone_label = get_step2_score(wy, sy, hy, ky, H_cm)
                    rwl, lhi, _, _, risk_label, risk_col = assess_risk(base_score, dyn_freq_mult, rot_mult, posture_mult_live, weight_kg)

                    draw_skeleton(canvas, lms, frame_w, frame_h, spine_deg)
                    draw_body_axes(canvas, feats, H_cm, posture_mult_live)
                    
                    draw_dashboard(canvas, feats, risk_label, risk_col, 
                                   rwl, lhi, weight_kg, fps_disp, frame_h, out_w, base_score, zone_label,
                                   rep_count, lpm, dyn_freq_mult, rot_mult, posture_mult_live, duration_str, spine_deg)

        # Encode image to jpeg for web streaming
        ret, buffer = cv2.imencode('.jpg', canvas)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
