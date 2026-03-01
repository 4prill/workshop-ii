import cv2
import numpy as np
import math
from ultralytics import YOLO

# -----------------------------
# LOAD MODEL
# -----------------------------
pose_model = YOLO("yolov8n-pose.pt")

# -----------------------------
# CONFIG
# -----------------------------
BEND_START = 25      # เริ่มย่อ
STAND_BACK = 15      # กลับมาตรง
BAD_POSTURE = 45     # ค่อมหนัก
MIN_FRAMES = 8       # จำนวนเฟรมขั้นต่ำต่อ 1 rep

# -----------------------------
# CALCULATE SPINE ANGLE
# -----------------------------
def trunk_angle(person):
    shoulder = person[5]
    hip = person[11]

    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    angle_rad = math.atan2(dx, -dy)
    return abs(math.degrees(angle_rad))

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run(video_path):

    cap = cv2.VideoCapture(video_path)

    states = {}
    sequences = {}
    scores = {}
    reps = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = pose_model(frame, verbose=False)[0]

        if result.keypoints is None:
            continue

        persons = result.keypoints.xy.cpu().numpy()
        if len(persons) == 0:
            continue

        # เรียงซ้าย → ขวา
        persons = sorted(persons, key=lambda p: np.mean(p[:,0]))

        # ---------------- LOOP EACH PERSON ----------------
        for idx, person in enumerate(persons):

            pid = idx + 1

            if pid not in states:
                states[pid] = "STAND"
                sequences[pid] = []
                scores[pid] = []
                reps[pid] = 0

            spine = trunk_angle(person)

            # ---------- STATE MACHINE ----------
            if states[pid] == "STAND":
                if spine > BEND_START:
                    states[pid] = "BENDING"
                    sequences[pid] = []

            elif states[pid] == "BENDING":

                sequences[pid].append(spine)

                if spine < STAND_BACK:

                    if len(sequences[pid]) >= MIN_FRAMES:

                        avg_spine = np.mean(sequences[pid])

                        # ---------- SCORING ----------
                        score = 100

                        if avg_spine > BAD_POSTURE:
                            score -= 40
                        elif avg_spine > 35:
                            score -= 20
                        elif avg_spine > 25:
                            score -= 10

                        score = max(score, 0)

                        scores[pid].append(score)
                        reps[pid] += 1

                        print(f"Person {pid} Rep {reps[pid]} Score: {score}")

                    states[pid] = "STAND"

            # ---------- DISPLAY ----------
            color = (0,255,0)
            if spine > BAD_POSTURE:
                color = (0,0,255)

            cx = int(np.mean(person[:,0]))
            cy = int(np.mean(person[:,1]))

            cv2.putText(frame,
                        f"P{pid} {int(spine)}deg",
                        (cx-40, cy-40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        cv2.imshow("Lifting Posture Score", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---------------- FINAL RESULT ----------------
    print("\n===== FINAL RESULT =====")
    for pid in scores:
        if len(scores[pid]) == 0:
            print(f"Person {pid} Score: 0.00% (no valid reps)")
        else:
            final_score = np.mean(scores[pid])
            print(f"Person {pid} Score: {final_score:.2f}%")

# -----------------------------
if __name__ == "__main__":
    run("assets/test3.mp4")