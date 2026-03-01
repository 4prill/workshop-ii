import cv2
import numpy as np
import math
from ultralytics import YOLO

# โหลดโมเดล
pose_model = YOLO("yolov8n-pose.pt")
detect_model = YOLO("yolov8n.pt")   # detect object

# -----------------------------
# Config
# -----------------------------
BEND_START = 25      # เริ่มย่อ (หลังเอียงเกินนี้)
STAND_BACK = 15      # กลับมาตรง
BAD_POSTURE = 45     # ค่อมอันตราย
MIN_FRAMES = 6

# -----------------------------
# Angle
# -----------------------------
def trunk_angle(person):
    shoulder = person[5]
    hip = person[11]

    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    angle_rad = math.atan2(dx, -dy)
    return abs(math.degrees(angle_rad))

# -----------------------------
# ตรวจเส้นตัดกล่อง
# -----------------------------
def line_intersects_box(p1, p2, box):
    x1,y1 = p1
    x2,y2 = p2
    bx1,by1,bx2,by2 = box

    # เช็ค midpoint แบบเร็วพอสำหรับงานนี้
    mx = (x1+x2)/2
    my = (y1+y2)/2

    if bx1 <= mx <= bx2 and by1 <= my <= by2:
        return True
    return False

# -----------------------------
# Baseline Saver
# -----------------------------
def save_baseline(video_path):

    cap = cv2.VideoCapture(video_path)

    state = "STAND"
    collecting = False

    spine_seq = []
    rep_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_res = pose_model(frame, verbose=False)[0]
        obj_res = detect_model(frame, verbose=False)[0]

        if pose_res.keypoints is None:
            continue

        persons = pose_res.keypoints.xy.cpu().numpy()
        if len(persons) == 0:
            continue

        p = persons[0]   # ใช้คนแรก

        spine_angle = trunk_angle(p)

        shoulder = p[5]
        hip = p[11]

        # ------------------ detect object ------------------
        object_boxes = []
        for box in obj_res.boxes:
            cls = int(box.cls[0])
            label = detect_model.names[cls]

            # จะนับเฉพาะกล่อง / crate / object
            if label in ["box", "suitcase", "backpack", "handbag", "book"]:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                object_boxes.append((x1,y1,x2,y2))

        # เช็คค่อมทับวัตถุ
        bad_overlap = False
        for box in object_boxes:
            if line_intersects_box(shoulder, hip, box):
                bad_overlap = True
                break

        # ------------------ State Machine ------------------

        if state == "STAND":
            if spine_angle > BEND_START:
                state = "BENDING"
                spine_seq = []
                collecting = True

        elif state == "BENDING":

            if not bad_overlap:   # เก็บเฉพาะท่าที่ไม่ค่อมทับของ
                spine_seq.append(spine_angle)

            if spine_angle < STAND_BACK:
                if len(spine_seq) > MIN_FRAMES:
                    rep_count += 1
                    print(f"Rep {rep_count} saved")

                state = "STAND"
                collecting = False

        # ------------------ Visual ------------------
        cv2.putText(frame, f"Spine: {int(spine_angle)}",
                    (30,40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        if bad_overlap:
            cv2.putText(frame, "BAD: Bending on object!",
                        (30,80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)

        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(spine_seq) == 0:
        print("No valid reps")
        return

    np.savez("baseline_spine_only.npz",
             spine=np.array(spine_seq))

    print("======================")
    print("Baseline Saved")
    print("Total reps:", rep_count)


if __name__ == "__main__":
    save_baseline("assets/baselinetrain.mp4")