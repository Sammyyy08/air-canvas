# ============================================================
#  Gesture Trainer — KNN-based Hand Gesture Classifier
#  Companion to AI Air Canvas
# ============================================================
#
#  WHAT THIS IS:
#  A proof of concept that replaces the rule-based gesture
#  detection in air_canvas.py with a KNN classifier trained
#  on real MediaPipe hand-landmark data.
#
#  HOW IT WORKS:
#  1. RECORD MODE — Shows your webcam. Hold a gesture steady
#     and press a number key (1-5) to record a sample. Each
#     gesture gets ~50 frames of 21 landmarks = 63 features.
#  2. TRAIN MODE — Trains a KNeighborsClassifier on your
#     recorded samples, just like the Iris Classifier does.
#  3. TEST MODE — Real-time gesture recognition using the
#     trained model. Press Q to quit at any time.
#
#  GESTURE LEGEND (in the Air Canvas context):
#    1 = ☝ ONE FINGER  → Draw
#    2 = ✌ TWO FINGERS → Move (no drawing)
#    3 = ✋ PALM       → Clear canvas
#    4 = 🤏 PINCH      → Cycle color
#    5 = ✊ FIST       → Stop / idle
#
#  USAGE:
#    python3 gesture_trainer.py record   — collect samples
#    python3 gesture_trainer.py train    — train the model
#    python3 gesture_trainer.py test     — live recognition
#    python3 gesture_trainer.py all      — record → train → test
#
#  DEPENDENCIES:
#    opencv-python, mediapipe, numpy, scikit-learn
#    (Same stack as Air Canvas + one extra)
#
# ============================================================

import cv2
import numpy as np
import os
import sys
import json

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── CONFIG ──────────────────────────────────────────────────

GESTURES = {
    1: "one_finger",
    2: "two_fingers",
    3: "palm",
    4: "pinch",
    5: "fist",
}

SAMPLES_PER_GESTURE = 50  # frames to record per keypress
DATA_FILE = "gesture_data.npz"   # saved numpy arrays
MODEL_FILE = "gesture_model.pkl"  # trained model + scaler
MODEL_PATH = "hand_landmarker.task"

# ── MEDIAPIPE SETUP ─────────────────────────────────────────

def create_detector():
    """Initialize the MediaPipe Hand Landmarker."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading hand tracking model (~10MB)...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
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


def extract_features(landmarks, width, height):
    """
    Convert 21 MediaPipe landmarks into a feature vector.
    
    Each landmark has (x, y, z) → 63 features total.
    Features are normalized relative to the wrist (landmark 0)
    to be scale/position invariant — the wrist becomes (0,0,0)
    and all other landmarks are relative to it.
    """
    wrist = landmarks[0]
    features = []
    for lm in landmarks:
        features.extend([
            (lm.x - wrist.x) * width,   # relative x, in pixels
            (lm.y - wrist.y) * height,  # relative y, in pixels
            lm.z - wrist.z,             # relative depth
        ])
    return np.array(features, dtype=np.float32)


# ── RECORD MODE ─────────────────────────────────────────────

def record_samples():
    """Record hand-landmark samples for each gesture class."""
    print("=" * 55)
    print("  RECORD MODE")
    print("=" * 55)
    print("  Hold a gesture steady and press a NUMBER KEY (1-5):")
    for k, v in GESTURES.items():
        print(f"    {k} = {v}")
    print("  Q = quit")
    print("-" * 55)

    detector = create_detector()
    cap = cv2.VideoCapture(0)

    X, y = [], []
    current_label = None
    samples_collected = 0
    recording = False

    print("\n  Ready! Show your hand and press 1-5 to start recording.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # --- MediaPipe inference ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        # --- Status overlay ---
        status_color = (0, 255, 0) if recording else (200, 200, 200)
        mode_text = f"RECORDING class {GESTURES.get(current_label, '?')} ({samples_collected}/{SAMPLES_PER_GESTURE})" if recording else "Press 1-5 to record"
        cv2.putText(display, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display, "Q=Quit", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            # Draw skeleton
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
                samples_collected += 1

                # Show a progress bar
                bar_len = 30
                filled = int(bar_len * samples_collected / SAMPLES_PER_GESTURE)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r  [{bar}] {samples_collected}/{SAMPLES_PER_GESTURE}")
                sys.stdout.flush()

                if samples_collected >= SAMPLES_PER_GESTURE:
                    print(f"\n  ✅ {GESTURES[current_label]} done! ({len(X)} samples total)\n")
                    recording = False
                    samples_collected = 0
                    current_label = None

        else:
            if not recording:
                cv2.putText(display, "No hand detected", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Gesture Trainer — Record Mode", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif not recording and chr(key).isdigit():
            label = int(chr(key))
            if label in GESTURES:
                current_label = label
                samples_collected = 0
                recording = True
                print(f"\n  ▶ Recording {GESTURES[label]}...")

    cap.release()
    cv2.destroyAllWindows()

    if len(X) == 0:
        print("\n⚠  No samples recorded. Nothing saved.")
        return False

    X = np.array(X)
    y = np.array(y)
    np.savez(DATA_FILE, X=X, y=y)
    
    # Print class breakdown
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{'='*55}")
    print(f"  SUMMARY: {len(X)} samples, {len(unique)} classes")
    for u, c in zip(unique, counts):
        print(f"    {GESTURES.get(u, u)}: {c} samples")
    print(f"{'='*55}")
    print(f"  Saved to {DATA_FILE}")
    return True


# ── TRAIN MODE ──────────────────────────────────────────────

def train_model():
    """Train a KNN classifier on recorded landmark data."""
    print("=" * 55)
    print("  TRAIN MODE")
    print("=" * 55)

    if not os.path.exists(DATA_FILE):
        print(f"✗ No data file found ({DATA_FILE}). Run 'record' first.")
        return False

    data = np.load(DATA_FILE)
    X, y = data["X"], data["y"]
    print(f"  Loaded {len(X)} samples, {len(np.unique(y))} classes")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features (matching the Iris Classifier pattern)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN
    # n_neighbors=5 matches the Iris Classifier; sqrt(n_samples) is another common choice
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    print("  ✅ Model trained!")

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*55}")
    print(f"  ACCURACY: {acc*100:.1f}%")
    print(f"{'='*55}")
    print("\n  Per-class breakdown:")
    target_names = [GESTURES.get(i, f"class_{i}") for i in sorted(np.unique(y))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save model and scaler
    import pickle
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "classes": np.unique(y)}, f)
    print(f"  ✅ Model saved to {MODEL_FILE}")
    return True


# ── TEST MODE ───────────────────────────────────────────────

def test_model():
    """Real-time gesture recognition using the trained model."""
    print("=" * 55)
    print("  TEST MODE — Live Recognition")
    print("=" * 55)

    if not os.path.exists(MODEL_FILE):
        print(f"✗ No model file found ({MODEL_FILE}). Run 'train' first.")
        return

    import pickle
    with open(MODEL_FILE, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    scaler = saved["scaler"]
    classes = saved["classes"]

    # Build reverse mapping: class_id → gesture name
    label_names = {c: GESTURES.get(c, f"class_{c}") for c in classes}
    print(f"  Loaded model with {len(classes)} classes: {list(label_names.values())}")
    print("  Show your hand. Press Q to quit.\n")

    # Color coding per gesture (BGR)
    gesture_colors = {
        1: (255, 0, 0),    # one_finger — blue
        2: (0, 255, 0),    # two_fingers — green
        3: (0, 255, 255),  # palm — yellow
        4: (255, 0, 255),  # pinch — magenta
        5: (0, 165, 255),  # fist — orange
    }

    detector = create_detector()
    cap = cv2.VideoCapture(0)

    # Simple smoothing: keep last N predictions
    from collections import deque
    history = deque(maxlen=5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            # Draw skeleton
            for c in HAND_CONNECTIONS:
                x1, y1 = int(lm[c[0]].x * w), int(lm[c[0]].y * h)
                x2, y2 = int(lm[c[1]].x * w), int(lm[c[1]].y * h)
                cv2.line(display, (x1, y1), (x2, y2), (100, 200, 200), 2)
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)

            # Predict
            feats = extract_features(lm, w, h).reshape(1, -1)
            feats_scaled = scaler.transform(feats)
            pred = model.predict(feats_scaled)[0]
            proba = model.predict_proba(feats_scaled).max()

            # Smooth prediction
            history.append(pred)
            smoothed = max(set(history), key=list(history).count)
            final_label = smoothed if list(history).count(smoothed) >= 3 else pred

            gesture_name = label_names.get(final_label, "unknown")
            color = gesture_colors.get(final_label, (255, 255, 255))

            # Display
            label_text = f"{gesture_name} ({proba*100:.0f}%)"
            cv2.putText(display, label_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Map to Air Canvas action
            action_map = {
                1: "✏ DRAW",
                2: "✌ MOVE",
                3: "🗑 CLEAR",
                4: "🎨 COLOR",
                5: "⏸ IDLE",
            }
            action = action_map.get(final_label, "...")
            cv2.putText(display, action, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Confidence bar
            bar_w = int(200 * proba)
            cv2.rectangle(display, (10, h - 40), (10 + bar_w, h - 20), color, -1)
            cv2.rectangle(display, (10, h - 40), (210, h - 20), (255, 255, 255), 1)

        else:
            cv2.putText(display, "No hand", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, "Q=Quit", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("Gesture Trainer — Live Recognition", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Done!")


# ── HAND CONNECTIONS (for skeleton drawing) ─────────────────

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["record", "train", "test", "all"]:
        print("  Usage:")
        print("    python3 gesture_trainer.py record   — collect gesture samples")
        print("    python3 gesture_trainer.py train    — train KNN classifier")
        print("    python3 gesture_trainer.py test     — live recognition")
        print("    python3 gesture_trainer.py all      — record → train → test")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "record":
        record_samples()
    elif mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    elif mode == "all":
        if record_samples():
            train_model()
            test_model()
