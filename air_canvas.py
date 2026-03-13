# ============================================================
#  ✋ AI Air Canvas — Draw in the air with your finger!
#  Compatible with mediapipe 0.10.30+ and Python 3.14
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import collections
import urllib.request
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── DOWNLOAD MODEL IF NEEDED ────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading hand tracking model (~10MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ Model downloaded!")

# ── SETTINGS ────────────────────────────────────────────────
COLORS = {
    "Blue"  : (255, 100,   0),
    "Green" : (  0, 220,  80),
    "Red"   : (  0,  60, 255),
    "Yellow": (  0, 220, 220),
    "White" : (255, 255, 255),
}
COLOR_NAMES   = list(COLORS.keys())
BRUSH_SIZE    = 8
TIP_IDS       = [4, 8, 12, 16, 20]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── MEDIAPIPE SETUP ─────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# ── STATE ────────────────────────────────────────────────────
canvas         = None
prev_x, prev_y = 0, 0
color_idx      = 0
current_color  = COLORS[COLOR_NAMES[color_idx]]
gesture_label  = ""
smooth_x       = collections.deque(maxlen=5)
smooth_y       = collections.deque(maxlen=5)


def count_fingers(lm, is_right):
    fingers = []
    fingers.append(1 if (is_right and lm[4].x < lm[3].x) or
                       (not is_right and lm[4].x > lm[3].x) else 0)
    for tip in TIP_IDS[1:]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return fingers


def draw_ui(frame, color_name, gesture):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (name, bgr) in enumerate(COLORS.items()):
        x = 20 + i * 110
        cv2.rectangle(frame, (x, 10), (x + 90, 60), bgr, -1)
        if name == color_name:
            cv2.rectangle(frame, (x, 10), (x + 90, 60), (255, 255, 255), 3)
        cv2.putText(frame, name, (x + 5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(frame, f"Gesture: {gesture}", (w - 280, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, "1 finger=Draw | 2 fingers=Move | Palm=Clear | Pinch=Color | Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


# ── MAIN LOOP ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("🎥 Camera started! Show your hand to the camera.")
print("   Press Q to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if result.hand_landmarks:
        lm       = result.hand_landmarks[0]
        is_right = result.handedness[0][0].display_name == "Right"

        for c in HAND_CONNECTIONS:
            x1, y1 = int(lm[c[0]].x * w), int(lm[c[0]].y * h)
            x2, y2 = int(lm[c[1]].x * w), int(lm[c[1]].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for point in lm:
            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (255, 255, 255), -1)

        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)

        smooth_x.append(ix); smooth_y.append(iy)
        sx = int(sum(smooth_x) / len(smooth_x))
        sy = int(sum(smooth_y) / len(smooth_y))

        fingers = count_fingers(lm, is_right)
        total   = sum(fingers)

        pinch_dist = np.hypot(ix - tx, iy - ty)
        if pinch_dist < 35:
            color_idx     = (color_idx + 1) % len(COLOR_NAMES)
            current_color = COLORS[COLOR_NAMES[color_idx]]
            gesture_label = "Color Change!"
            prev_x, prev_y = 0, 0
            cv2.waitKey(400)
        elif total == 5:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            gesture_label = "Clear ✓"
            prev_x, prev_y = 0, 0
        elif fingers[1] == 1 and fingers[2] == 1 and total == 2:
            gesture_label = "Moving ✌"
            prev_x, prev_y = sx, sy
        elif fingers[1] == 1 and total == 1:
            gesture_label = "Drawing ☝"
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = sx, sy
            cv2.line(canvas, (prev_x, prev_y), (sx, sy), current_color, BRUSH_SIZE)
            prev_x, prev_y = sx, sy
        else:
            gesture_label = "..."
            prev_x, prev_y = 0, 0

        cv2.circle(frame, (sx, sy), BRUSH_SIZE // 2 + 4, current_color, -1)

    else:
        gesture_label = "No hand detected"
        prev_x, prev_y = 0, 0

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    frame[mask > 0] = canvas[mask > 0]

    draw_ui(frame, COLOR_NAMES[color_idx], gesture_label)
    cv2.imshow("AI Air Canvas  |  Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Done!")
