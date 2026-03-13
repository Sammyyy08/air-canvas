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
| Q key | Quit |

---

## 🎨 Available Colors
Blue · Green · Red · Yellow · White

---

## 🛠️ Tech Stack

- **Python 3.x**
- `OpenCV` — camera feed and drawing
- `MediaPipe` — Google's real-time hand tracking AI (detects 21 hand landmarks)
- `NumPy` — image array manipulation

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/air-canvas.git
   cd air-canvas
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run it**
   ```bash
   python air_canvas.py
   ```

4. **Show your hand** to the camera and start drawing! ✋

---

## 🧠 How It Works

1. **OpenCV** captures your webcam feed frame by frame
2. **MediaPipe** detects your hand and returns 21 landmark points (fingertips, knuckles, wrist)
3. We check which fingers are raised to identify the gesture
4. Based on the gesture, we draw lines on a transparent canvas layer
5. The canvas is blended onto the camera feed in real time

---

## 📖 What I Learned

- Real-time video processing with **OpenCV**
- AI hand landmark detection with **MediaPipe**
- Working with image arrays using **NumPy**
- Building gesture-based human-computer interaction
- Blending layers to create augmented reality effects

---

## 📬 Connect
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/Sammyyy08)
