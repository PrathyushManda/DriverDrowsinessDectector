# Driver Drowsiness Detection

A modern, real-time platform to enhance road safety by detecting driver fatigue and distraction using computer vision and machine learning. This end-to-end solution captures webcam video, analyzes eye closure and head pose, and delivers timely alerts, robust logging, and live analytics through a sleek web dashboard.

---

## 🔎 Features

* **Real-Time Video Capture**: Leveraging OpenCV for reliable webcam input at 10 FPS.
* **Advanced Landmark Detection**: MediaPipe Face Mesh for 468-point facial landmarks.
* **Eye Aspect Ratio (EAR)**: Calculates eye openness to differentiate blinks from prolonged closure.
* **Head-Pose Estimation**: Computes pitch and yaw angles to detect head nods and distractions.
* **Customizable Thresholds**: Interactive Streamlit sliders for EAR, blink duration, and drowsiness duration.
* **Blink & Drowsiness Classification**: Distinguishes short blinks from sustained eye closure events.
* **Audio & Visual Alerts**: Red border flash on the video and plays an alarm sound when fatigue is detected.
* **SMS Notifications**: Integrates with Twilio to send real-time SMS alerts on drowsiness.
* **Persistent Storage**: SQLite database logs all events with timestamps, EAR, pitch, and yaw for analysis.
* **REST API**: FastAPI endpoint (`/events`) to retrieve logged events programmatically.
* **Live Dashboard**: Streamlit UI displays webcam feed, EAR trend chart, and real-time statistics.
* **Containerized Deployment**: Dockerfile and `requirements.txt` for one-step deployment anywhere.

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   export TWILIO_ACCOUNT_SID=your_sid
   export TWILIO_AUTH_TOKEN=your_token
   export TWILIO_FROM=+1234567890
   export TWILIO_TO=+0987654321
   ```

4. **Run the application**

   ```bash
   python -m streamlit run app.py
   ```

   Your browser will open at `http://localhost:8501`.

5. **(Optional) Run with Docker**

   ```bash
   docker build -t drowsy-app .
   docker run -p 8501:8501 drowsy-app
   ```

---

## ⚙️ Configuration

* **EAR Threshold**: Set on the sidebar to adjust sensitivity for eye closure detection.
* **Drowsy Duration**: Duration (in seconds) of sustained eye closure to trigger a drowsiness event.
* **Blink Max Duration**: Maximum duration (in seconds) for a blink (short eye closure).

---

## 📂 Project Structure

```
├── app.py             # Main Streamlit application
├── Dockerfile         # Containerization instructions
├── requirements.txt   # Python dependencies
├── drowsy_events.db   # SQLite database (auto-created)
├── drowsy_log.csv     # CSV log (auto-created)
├── alarm.mp3          # Alarm sound file (user-provided)
└── README.md          # This documentation
```
