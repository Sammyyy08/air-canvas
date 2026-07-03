# ============================================================
#  🤖 AI Air Canvas ML — KNN-powered gesture detection
#  Fork of air_canvas.py with ML-based gesture recognition
# ============================================================
#
#  WHAT THIS IS:
#  Same Air Canvas app, but with a toggleable ML gesture engine.
#  Press M to switch between RULE-based and ML-based detection.
#
#  The ML mode uses the KNN classifier trained by
#  gesture_trainer.py — the same algorithm from the Iris
#  Classifier, applied to real-time hand landmark data.
#
#  WHY THIS MATTERS:
#  Rule-based gesture detection uses hardcoded thresholds that
#  break with different hand sizes, angles, and lighting.
#  ML-based detection learns from real examples — it adapts.
#  Having BOTH in one app lets you demo the difference.
#
#  USAGE:
#    python3 air_canvas_ml.py             — ML mode (if model exists)
#    python3 air_canvas_ml.py --rule       — start in rule mode
#    python3 air_canvas_ml.py --record     — record samples, then launch
#    [M] in-app       — toggle between rule and ML modes
#    [R] in ML mode   — re-record gestures (opens recorder)
#    [Q]              — quit
#
#  DEPENDENCIES:
#    opencv-python, mediapipe, numpy, scikit-learn
#    (Same stack, no new deps)
#
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import collections
import urllib.request
import os
import sys
import pickle

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── CONFIG ──────────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
MODEL_FILE = "gesture_model.pkl"

COLORS = {
    "Blue"  : (255, 100,   0),
    "Green" : (  0, 220,  80),
    "Red"   : (  0,  60, 255),
    "Yellow": (  0, 220, 220),
    "White" : (255, 255, 255),
}
COLOR_NAMES = list(COLORS.keys())
BRUSH_SIZE  = 8
TIP_IDS     = [4, 8, 12, 16, 20]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Gesture-to-action mapping (same meaning as gesture_trainer.py)
GESTURE_ACTIONS = {
    1: "draw",      # one finger
    2: "move",      # two fingers
    3: "clear",     # palm
    4: "color",     # pinch
    5: "idle",      # fist
}
GESTURE_NAMES = {
    1: "☝ Draw",
    2: "✌ Move",
    3: "✋ Clear",
    4: "🤏 Color",
    5: "✊ Idle",
}


# ── MEDIAPIPE SETUP ─────────────────────────────────────────

def create_detector():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading hand tracking model (~10MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            MODEL_PATH
        )
        print("✅ Model downloaded!")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def load_ml_model():
    """Load the trained KNN model and scaler. Returns None if unavailable."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)
        return saved  # {model, scaler, classes}
    except Exception:
        return None


def extract_features(landmarks, width, height):
    """Normalize landmarks relative to wrist (landmark 0) — 63 features."""
    wrist = landmarks[0]
    feats = []
    for lm in landmarks:
        feats.extend([
            (lm.x - wrist.x) * width,
            (lm.y - wrist.y) * height,
            lm.z - wrist.z,
        ])
    return np.array(feats, dtype=np.float32)


# ── RULE-BASED GESTURE DETECTION (original) ─────────────────

def count_fingers(lm, is_right):
    fingers = []
    fingers.append(1 if (is_right and lm[4].x < lm[3].x) or
                       (not is_right and lm[4].x > lm[3].x) else 0)
    for tip in TIP_IDS[1:]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return fingers


def rule_based_gesture(lm, is_right, ix, iy, tx, ty):
    """
    Returns (action_string, color_index_change).
    Same logic as original air_canvas.py but returns structured output.
    """
    fingers = count_fingers(lm, is_right)
    total = sum(fingers)
    pinch_dist = np.hypot(ix - tx, iy - ty)

    if pinch_dist < 35:
        return "color", 1, "..."
    elif total == 5:
        return "clear", 0, "Palm"
    elif fingers[1] == 1 and fingers[2] == 1 and total == 2:
        return "move", 0, "✌ Move"
    elif fingers[1] == 1 and total == 1:
        return "draw", 0, "☝ Draw"
    else:
        return "idle", 0, "..."


# ── ML-BASED GESTURE DETECTION ──────────────────────────────

def ml_based_gesture(lm, ml_model, scaler, history, w, h):
    """
    Returns (action_string, prediction_label_name, confidence).
    Uses smoothed majority-vote over last 5 frames.
    """
    feats = extract_features(lm, w, h).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    pred = ml_model.predict(feats_scaled)[0]
    proba = ml_model.predict_proba(feats_scaled).max()

    history.append(pred)
    smoothed = max(set(history), key=list(history).count)
    final_pred = smoothed if list(history).count(smoothed) >= 3 else pred

    action = GESTURE_ACTIONS.get(final_pred, "idle")
    name = GESTURE_NAMES.get(final_pred, "?")
    return action, f"{name} ({proba*100:.0f}%)", final_pred


# ── RECORDER (inline, simplified) ──────────────────────────

def record_gestures(detector, samples_per=50):
    """
    In-app recorder — press 1-5 to record samples for each gesture.
    Runs until Q is pressed. Returns (X, y) or None.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    GESTURES = {1: "one_finger", 2: "two_fingers", 3: "palm", 4: "pinch", 5: "fist"}
    X, y = [], []
    current_label = None
    collected = 0
    recording = False

    cap = cv2.VideoCapture(0)
    print("\n  🎥 RECORDING MODE — Press 1-5 to record each gesture")
    print("  Hold the gesture steady. Press Q when done.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        display = frame.copy()
        status_color = (0, 255, 0) if recording else (200, 200, 200)

        if recording:
            label_name = GESTURES.get(current_label, "?")
            text = f"RECORDING {label_name} ({collected}/{samples_per})"
        else:
            text = "Press 1-5 to record a gesture | Q = done"

        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            for c in HAND_CONNECTIONS:
                x1, y1 = int(lm[c[0]].x * w), int(lm[c[0]].y * h)
                x2, y2 = int(lm[c[1]].x * w), int(lm[c[1]].y * h)
                cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(display, (cx, cy), 4, (255, 255, 255), -1)

            if recording:
                feats = extract_features(lm, w, h)
                X.append(feats)
                y.append(current_label)
                collected += 1
                if collected >= samples_per:
                    print(f"  ✅ {GESTURES[current_label]} done ({len(X)} total)")
                    recording = False
                    collected = 0
                    current_label = None
        else:
            if not recording:
                cv2.putText(display, "No hand detected", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Air Canvas ML — Recording Gestures", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif not recording and 49 <= key <= 53:  # digits 1-5
            label = key - 48
            current_label = label
            collected = 0
            recording = True
            print(f"  ▶ Recording {GESTURES[label]}...")

    cap.release()
    cv2.destroyAllWindows()

    if len(X) == 0:
        print("  ⚠ No samples recorded.")
        return None

    X_arr = np.array(X)
    y_arr = np.array(y)

    # Train on the spot
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y_arr)

    # Save
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "classes": np.unique(y_arr)}, f)

    unique, counts = np.unique(y_arr, return_counts=True)
    print(f"\n  ✅ Trained on {len(X_arr)} samples, {len(unique)} classes")
    for u, c in zip(unique, counts):
        print(f"     {GESTURES.get(u, u)}: {c} samples")
    print(f"     Saved to {MODEL_FILE}")
    print(f"  Toggle back to the main window with the camera.")
    print(f"  ML mode is now active with your custom model!\n")

    return {"model": model, "scaler": scaler, "classes": np.unique(y_arr)}


# ── MAIN LOOP ───────────────────────────────────────────────

def run_air_canvas(start_with_ml=True):
    detector = create_detector()
    ml_model_data = load_ml_model()
    use_ml = start_with_ml and (ml_model_data is not None)
    ml_history = collections.deque(maxlen=5)

    canvas = None
    prev_x, prev_y = 0, 0
    color_idx = 0
    current_color = COLORS[COLOR_NAMES[color_idx]]
    gesture_label = ""
    mode_label = "ML" if use_ml else "RULE"
    smooth_x = collections.deque(maxlen=5)
    smooth_y = collections.deque(maxlen=5)

    if use_ml:
        print("  🤖 Mode: ML (KNN gesture classification)")
        print("     Press [M] to toggle Rule mode")
        print("     Press [R] to re-record gestures")
    else:
        print("  📐 Mode: RULE (geometric gesture detection)")
        print("     Press [M] to toggle ML mode")
        if ml_model_data is None:
            print("     (No ML model found — record gestures with gesture_trainer.py)")

    print("  [Q] to quit\n")

    cap = cv2.VideoCapture(0)
    print("🎥 Camera started! Show your hand.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            is_right = result.handedness[0][0].display_name == "Right"

            # Draw skeleton
            for c in HAND_CONNECTIONS:
                x1, y1 = int(lm[c[0]].x * w), int(lm[c[0]].y * h)
                x2, y2 = int(lm[c[1]].x * w), int(lm[c[1]].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for point in lm:
                cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (255, 255, 255), -1)

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            smooth_x.append(ix)
            smooth_y.append(iy)
            sx = int(sum(smooth_x) / len(smooth_x))
            sy = int(sum(smooth_y) / len(smooth_y))

            if use_ml and ml_model_data is not None:
                # ── ML-BASED GESTURE ──
                model = ml_model_data["model"]
                scaler = ml_model_data["scaler"]
                action, label, _ = ml_based_gesture(lm, model, scaler, ml_history, w, h)
                gesture_label = label

                if action == "draw":
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = sx, sy
                    cv2.line(canvas, (prev_x, prev_y), (sx, sy), current_color, BRUSH_SIZE)
                    prev_x, prev_y = sx, sy
                    gesture_label = f"ML: {label}"
                elif action == "move":
                    prev_x, prev_y = sx, sy
                    gesture_label = f"ML: {label}"
                elif action == "clear":
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    prev_x, prev_y = 0, 0
                    gesture_label = f"ML: Clear ✓"
                elif action == "color":
                    color_idx = (color_idx + 1) % len(COLOR_NAMES)
                    current_color = COLORS[COLOR_NAMES[color_idx]]
                    prev_x, prev_y = 0, 0
                    gesture_label = f"ML: {label}"
                    cv2.waitKey(400)
                else:  # idle
                    prev_x, prev_y = 0, 0
                    gesture_label = f"ML: {label}"

            else:
                # ── RULE-BASED GESTURE (original logic) ──
                total_fingers = sum(count_fingers(lm, is_right))
                pinch_dist = np.hypot(ix - tx, iy - ty)

                if pinch_dist < 35:
                    color_idx = (color_idx + 1) % len(COLOR_NAMES)
                    current_color = COLORS[COLOR_NAMES[color_idx]]
                    gesture_label = "RULE: Color Change!"
                    prev_x, prev_y = 0, 0
                    cv2.waitKey(400)
                elif total_fingers == 5:
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    gesture_label = "RULE: Clear ✓"
                    prev_x, prev_y = 0, 0
                elif count_fingers(lm, is_right)[1] == 1 and count_fingers(lm, is_right)[2] == 1 and total_fingers == 2:
                    gesture_label = "RULE: Moving ✌"
                    prev_x, prev_y = sx, sy
                elif count_fingers(lm, is_right)[1] == 1 and total_fingers == 1:
                    gesture_label = "RULE: Drawing ☝"
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = sx, sy
                    cv2.line(canvas, (prev_x, prev_y), (sx, sy), current_color, BRUSH_SIZE)
                    prev_x, prev_y = sx, sy
                else:
                    gesture_label = "RULE: ..."
                    prev_x, prev_y = 0, 0

            cv2.circle(frame, (sx, sy), BRUSH_SIZE // 2 + 4, current_color, -1)

        else:
            gesture_label = "No hand detected"
            prev_x, prev_y = 0, 0

        # Composite canvas onto frame
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        frame[mask > 0] = canvas[mask > 0]

        # ── UI ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Color swatches
        for i, (name, bgr) in enumerate(COLORS.items()):
            x = 20 + i * 110
            cv2.rectangle(frame, (x, 10), (x + 90, 60), bgr, -1)
            if name == COLOR_NAMES[color_idx]:
                cv2.rectangle(frame, (x, 10), (x + 90, 60), (255, 255, 255), 3)
            cv2.putText(frame, name, (x + 5, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Mode indicator
        mode_color = (0, 255, 255) if use_ml else (180, 180, 255)
        cv2.putText(frame, f"MODE: {mode_label}", (w - 220, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

        # Gesture label
        cv2.putText(frame, gesture_label, (w - 280, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Instructions
        cv2.putText(frame, "M=Mode | R=Record | Q=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        cv2.imshow("AI Air Canvas ML  |  M=toggle mode  |  Q=quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Toggle mode
            if use_ml:
                use_ml = False
                mode_label = "RULE"
                print("  Switched to RULE mode")
            else:
                if ml_model_data is not None:
                    use_ml = True
                    mode_label = "ML"
                    ml_history.clear()
                    print("  Switched to ML mode")
                else:
                    print("  ⚠ No ML model available. Record gestures first (R)")
        elif key == ord('r'):
            print("\n  Starting gesture recorder...")
            result_data = record_gestures(detector)
            if result_data is not None:
                ml_model_data = result_data
                use_ml = True
                mode_label = "ML"
                ml_history.clear()
            else:
                print("  Recording cancelled or failed.")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Done!")


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    start_ml = True
    record_first = False

    for arg in sys.argv[1:]:
        if arg == "--rule":
            start_ml = False
        elif arg == "--record":
            record_first = True

    if record_first:
        print("=" * 55)
        print("  AI Air Canvas ML — Setup Mode")
        print("=" * 55)
        print("  Let's train your custom gesture model first.")
        print("  Then the canvas will launch in ML mode.\n")
        detector = create_detector()
        result = record_gestures(detector)
        if result is not None:
            start_ml = True
        else:
            print("\n  Starting without ML model. Press R in-app to record later.\n")

    run_air_canvas(start_with_ml=start_ml)
