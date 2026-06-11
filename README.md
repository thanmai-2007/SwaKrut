# SwaKrut — AI Confidence Model Extension
## Complete Setup Guide

---

## What Was Built

This is a **hybrid confidence prediction system** added as an extension to the existing SwaKrut interview platform. It combines:

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML + JS (existing) | Interview UI, speech recognition, webcam |
| Scoring | Local rule-based (fallback) | Always-on instant scoring |
| AI Model | Python Flask + scikit-learn RandomForest | Upgraded hybrid scoring |
| Database | MongoDB (optional) / JSON fallback | Session persistence |
| Dashboard | model_dashboard.html | Live prediction testing + analytics |

---

## File Structure

```
Swa-Krut-2/
├── index.html               ← Landing page (existing)
├── interview.html           ← ✅ UPGRADED — now calls AI backend
├── feedback.html            ← Feedback report (existing)
├── model_dashboard.html     ← 🆕 AI Model dashboard + live tester
│
└── backend/
    ├── app.py               ← 🆕 Flask API with RandomForest model
    ├── requirements.txt     ← Python dependencies
    ├── confidence_model.pkl ← Auto-generated on first run
    └── sessions.json        ← Session fallback storage
```

---

## Step 1 — Install Python & Flask

### Option A — pip (recommended)
```bash
pip install flask flask-cors scikit-learn numpy pymongo
```

### Option B — virtual environment (cleaner)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install flask flask-cors scikit-learn numpy pymongo
```

---

## Step 2 — Start the Backend

```bash
cd backend
python app.py
```

You should see:
```
[ML] Training hybrid confidence model...
[ML] Model trained ✓
🚀 SwaKrut Confidence API starting...
   POST /predict        — full hybrid prediction
   POST /predict/text   — text-only prediction
   POST /session/save   — save session
   GET  /model/info     — model metadata
* Running on http://0.0.0.0:5050
```

The model trains itself **automatically** on first run using 2,000 synthetic training samples based on interview confidence research. This takes about 2–3 seconds. After that, it caches to `confidence_model.pkl`.

---

## Step 3 — Open the Frontend

Open `interview.html` in your browser (or via Node.js server as before).

The interview page **auto-detects** whether the Flask backend is running:
- ✅ **Backend online** → Uses RandomForest AI scoring (shown in nav as `🤖 AI: loaded`)
- 🔄 **Backend offline** → Falls back to local rule-based scoring seamlessly
- After each answer, an **AI Confidence Analysis panel** appears with specific feedback

---

## Step 4 — Open the Model Dashboard

Open `model_dashboard.html` to:
- **Test live predictions** — paste any text, adjust visual feature sliders, get instant scores
- **See feature importances** — which signals matter most for confidence
- **View session history** — all your interview sessions
- **Radar/bar charts** — visual breakdown of your performance

---

## How the Hybrid Model Works

### Pipeline
```
Webcam (JS) ─────→ Visual Features    ─┐
                   eye_contact          │
                   smile_score          │
                   posture_score        ├──→ RandomForest → Confidence (0-100)
                   pause_frequency      │       200 trees
Speech API ──────→ Text Features      ─┘       depth 8
                   filler_ratio                 StandardScaler
                   wpm
                   answer_length
                   sentence_structure
                   tech_keyword_count
                   speech_duration
```

### Feature Importances (from trained model)
| Feature | Weight | Description |
|---|---|---|
| eye_contact | ~21% | Strongest predictor of confidence |
| filler_ratio | ~17% | um/uh/like frequency |
| posture_score | ~14% | Upright vs slouched |
| wpm | ~12% | Speaking pace (ideal: 120-160) |
| sentence_structure | ~10% | STAR method, transitional words |
| answer_length | ~9% | Word count |
| smile_score | ~7% | Facial expression |
| tech_keywords | ~5% | Domain knowledge signals |
| pause_frequency | ~3% | Pauses per word |
| speech_duration | ~2% | Total time on question |

### Scoring Flow
1. User answers question → transcript captured
2. **Local score returned instantly** (rule-based, no latency)
3. **Async AI call** to `/predict` with text + simulated visual features
4. If AI responds within 4 seconds → local score upgraded to AI score
5. AI Feedback panel appears with specific improvement tips
6. Session saved to MongoDB (or sessions.json fallback)

---

## API Endpoints

### POST /predict
Full hybrid prediction with visual + text features.

**Request:**
```json
{
  "text": "I implemented it using a hash map for O(1) lookup...",
  "duration": 45.2,
  "eye_contact": 0.78,
  "smile_score": 0.62,
  "posture_score": 0.71,
  "pause_frequency": 0.12
}
```

**Response:**
```json
{
  "success": true,
  "scores": {
    "confidence": 76.3,
    "technical": 82,
    "communication": 71,
    "overall": 76.4
  },
  "text_analysis": {
    "word_count": 45,
    "filler_count": 1,
    "filler_ratio": 0.022,
    "has_structure": 1,
    "tech_keyword_count": 4,
    "wpm": 59.9
  },
  "feedback": {
    "confidence_score": 76.3,
    "overall_tip": "👍 Good confidence. Focus on reducing filler words.",
    "weaknesses": [...],
    "strengths": [...]
  }
}
```

### POST /predict/text
Text-only (no visual features needed).

### POST /session/save
Save complete session to database.

### GET /model/info
Returns model metadata and feature importances.

### GET /health
Returns backend status, model type, MongoDB status.

---

## Optional: MongoDB Setup

If you want full session persistence with MongoDB:

1. Download MongoDB Community: https://www.mongodb.com/try/download/community
2. Start MongoDB: `mongod` (runs on port 27017 by default)
3. The backend detects MongoDB automatically and uses it

Without MongoDB, sessions are saved to `sessions.json` (last 100 sessions).

---

## Optional: Real MediaPipe Vision (Advanced)

Currently, visual features (eye contact, posture, smile) are **simulated** by the JavaScript frontend since running MediaPipe in the browser requires WASM + model files.

To add real computer vision:

```bash
pip install mediapipe opencv-python deepface
```

Then add a webcam frame analysis endpoint to `app.py`:

```python
import mediapipe as mp
import base64, cv2, numpy as np

@app.route("/analyze/frame", methods=["POST"])
def analyze_frame():
    # Accept base64 frame from JS
    data = request.get_json()
    img_data = base64.b64decode(data["frame"].split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # MediaPipe face mesh for eye contact
    # MediaPipe pose for posture
    # DeepFace for emotion
    ...
```

And in JS, send frames every 2 seconds:
```javascript
canvas.toBlob(function(blob){ /* send to /analyze/frame */ });
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CORS error` in browser | Make sure `flask-cors` is installed and backend is on port 5050 |
| Model not loading | Delete `confidence_model.pkl` and restart — will retrain |
| MongoDB not connecting | Backend auto-falls back to JSON storage, no action needed |
| `scoreLocal is not defined` | Clear browser cache and hard-reload interview.html |
| Backend shows 404 | Make sure you're running `python app.py` from the `backend/` folder |

---

## Architecture Summary

```
Browser
  ├── interview.html
  │     ├── Web Speech API (microphone → transcript)
  │     ├── getUserMedia (webcam display)
  │     ├── scoreLocal() ← instant rule-based scoring
  │     └── scoreAnsAI() ── HTTP POST ──→ Flask API (port 5050)
  │                                            ├── Feature extraction
  │                                            ├── RandomForest.predict()
  │                                            └── Feedback generation
  │
  └── model_dashboard.html
        ├── Live prediction tester (sliders + text)
        ├── Feature importance bars
        ├── Session history table
        └── Radar + bar charts (Chart.js)
```

Built with: HTML5, CSS3, JavaScript, Python 3, Flask, scikit-learn, NumPy, MongoDB
