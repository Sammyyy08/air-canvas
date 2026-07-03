# ✋ AI Air Canvas

Draw in the air with just your finger — no touch needed! Uses your **laptop camera + AI hand tracking** to turn your hand into a paintbrush.

---

## 🎮 Controls

| Gesture | Action |
|---|---|
| ☝️ One finger up | **Draw** on canvas |
| ✌️ Two fingers up | **Move** without drawing |
| 🖐️ Open palm (all 5 fingers) | **Clear** the canvas |
| 🤌 Pinch (index + thumb close) | **Change color** |
| M | Toggle ML / Rule mode |
| R | Record custom gestures (in ML mode) |
| Q | Quit |

---

## 🎨 Available Colors
Blue · Green · Red · Yellow · White

---

## 🛠️ Tech Stack

- **Python 3.x**
- `OpenCV` — camera feed and drawing
- `MediaPipe` — Google's real-time hand tracking AI (detects 21 hand landmarks)
- `NumPy` — image array manipulation
- `scikit-learn` — KNN classifier for ML gesture mode

---

## 🚀 How to Run

### Rule-based mode (original)

```bash
pip install -r requirements.txt
python air_canvas.py
```

### ML-powered mode (recommended)

```bash
pip install -r requirements.txt
python air_canvas_ml.py            # starts in ML mode if model exists
python air_canvas_ml.py --rule     # starts in rule-based mode
python air_canvas_ml.py --record   # record gestures, then launch in ML mode
```

---

## 🤖 ML Gesture Recognition

This repo includes two gesture engines:

### 📐 Rule-based (`air_canvas.py`)
The original approach. Uses geometric rules — finger angles, distance thresholds, and relative landmark positions — to determine the gesture. Fast and dependency-light, but thresholds are resolution-dependent and can break with different hand sizes or angles.

### 🧠 ML-based (`air_canvas_ml.py`)
Uses a **K-Nearest Neighbors classifier** (same algorithm from my [Iris Classifier](https://github.com/Sammyyy08/iris-flower-classifier)) trained on real hand-landmark data. Features are normalized relative to the wrist (landmark 0) so the model is scale/position invariant — the same gesture works anywhere on screen.

**Press M in-app** to switch between modes and compare them live.

To train your own model:
```bash
python air_canvas_ml.py --record
```
Press **1** (one finger), **2** (two fingers), **3** (palm), **4** (pinch), **5** (fist) while holding each gesture steady. The app trains a KNN on the spot and launches in ML mode.

Or use the standalone trainer for more control:
```bash
python gesture_trainer.py all    # record → train → test pipeline
```

### Why both?
Rule-based is the baseline — it works without data. ML-based learns from your actual hand, adapting to your hand size and style. Having both in one app is a clean demo of the progression from hardcoded heuristics to learned models. Same pipeline concept, applied to a real-time CV task.

---

## 🧠 How It Works

1. **OpenCV** captures your webcam feed frame by frame
2. **MediaPipe** detects your hand and returns 21 landmark points (fingertips, knuckles, wrist)
3. We extract normalized features (rule-based: finger state check; ML-based: 63 wrist-relative landmark coordinates)
4. The KNN classifier (or rule engine) identifies the gesture in real time
5. Based on the gesture, we draw lines on a transparent canvas layer
6. The canvas is blended onto the camera feed in real time

---

## 📖 What I Learned

- Real-time video processing with **OpenCV**
- AI hand landmark detection with **MediaPipe**
- Working with image arrays using **NumPy**
- Building gesture-based human-computer interaction
- Blending layers to create augmented reality effects
- **K-Nearest Neighbors** for real-time classification
- Feature engineering: wrist-relative normalization for scale/position invariance
- Training a model from webcam data and deploying it in the same loop

---

## 📬 Connect
[LinkedIn](https://linkedin.com/in/swayam-chaudhari-714a892a2) · [GitHub](https://github.com/Sammyyy08)
