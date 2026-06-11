"""
app.py — SwaKrut Confidence Flask API
======================================
Exposes the trained RandomForest model (confidence_model.pkl) via HTTP.

Endpoints
---------
  POST /predict        — full prediction, visual + text features
  POST /predict/text   — text-only (derives visual defaults)
  GET  /health         — liveness check

Run
---
  pip install flask flask-cors scikit-learn pandas numpy joblib textblob
  python app.py
"""

import os, re, math, time
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Optional TextBlob sentiment (graceful fallback) ─────────────────────────
try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "confidence_model.pkl")
PORT       = int(os.environ.get("RF_PORT", 5050))
HOST       = os.environ.get("RF_HOST", "0.0.0.0")

FEATURES = [
    "total_words", "total_fillers", "face_detected",
    "avg_sentiment", "avg_words_per_q", "filler_ratio",
    "eye_contact", "posture_score", "speech_rate_wpm",
]

# Label → numeric confidence score mapping (used for a 0-100 "conf_score")
LABEL_SCORE = {"Low": 28, "Medium": 62, "High": 88}

# ── Load model once at startup ───────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[RF] {MODEL_PATH} not found — training now...")
        _auto_train()
    model = joblib.load(MODEL_PATH)
    print(f"[RF] Model loaded  classes={model.classes_.tolist()}")
    return model

def _auto_train():
    """Train a quick synthetic model if the pkl is missing."""
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(42)
    n = 200

    def cls(label, wc, fc, ec, wpm, post, sent):
        tw  = rng.integers(*wc,  size=n).astype(float)
        tf  = rng.integers(*fc,  size=n).astype(float)
        eye = rng.uniform(*ec,   size=n).round(3)
        spd = rng.uniform(*wpm,  size=n).round(1)
        pst = rng.uniform(*post, size=n).round(3)
        snt = rng.uniform(*sent, size=n).round(3)
        return pd.DataFrame({
            "total_words":     tw,
            "total_fillers":   tf,
            "face_detected":   (eye > 0.3).astype(int),
            "avg_sentiment":   snt,
            "avg_words_per_q": (tw / 8).round(1),
            "filler_ratio":    (tf / tw.clip(1)).round(4),
            "eye_contact":     eye,
            "posture_score":   pst,
            "speech_rate_wpm": spd,
            "confidence":      label,
        })

    df = pd.concat([
        cls("Low",    (20,110),(8,28),(0.05,0.38),(60,105),(0.0,0.35),(-0.35,0.10)),
        cls("Medium", (90,260),(3,10),(0.35,0.72),(105,148),(0.30,0.72),(0.05,0.40)),
        cls("High",  (220,520),(0,5),(0.65,1.00),(130,175),(0.65,1.00),(0.25,0.75)),
    ], ignore_index=True).sample(frac=1, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(df[FEATURES], df["confidence"])
    joblib.dump(clf, MODEL_PATH)
    print(f"[RF] Model trained & saved → {MODEL_PATH}")


# ── Text analysis helpers ─────────────────────────────────────────────────────
FILLER_RE = re.compile(
    r"\b(um+|uh+|like|you know|basically|literally|right|so|kind of|sort of|i mean)\b",
    re.IGNORECASE,
)

def analyse_text(text: str, num_questions: int = 1) -> dict:
    """Extract all text-based features from a transcript."""
    words = text.strip().split()
    total_words   = len(words)
    total_fillers = len(FILLER_RE.findall(text))
    filler_ratio  = total_fillers / max(total_words, 1)
    avg_words_per_q = total_words / max(num_questions, 1)

    # Sentiment
    if _HAS_TEXTBLOB and total_words > 0:
        avg_sentiment = round(TextBlob(text).sentiment.polarity, 4)
    else:
        # Simple fallback: count positive vs negative indicator words
        pos = len(re.findall(r"\b(good|great|excellent|strong|succeed|achieve|built|led|solved|improved)\b", text, re.I))
        neg = len(re.findall(r"\b(bad|fail|wrong|never|problem|difficult|struggle|couldn.t)\b", text, re.I))
        avg_sentiment = round(min(max((pos - neg) / max(total_words / 20, 1), -1.0), 1.0), 4)

    # Estimate speech rate (assume ~130 wpm default; scale by filler density)
    # A rough proxy: confident answers tend toward 130-160 wpm
    speech_rate_wpm = max(60.0, min(180.0, 130.0 - filler_ratio * 200 + (1 if total_words > 150 else 0) * 15))

    return {
        "total_words":     total_words,
        "total_fillers":   total_fillers,
        "filler_ratio":    round(filler_ratio, 4),
        "avg_words_per_q": round(avg_words_per_q, 1),
        "avg_sentiment":   avg_sentiment,
        "speech_rate_wpm": round(speech_rate_wpm, 1),
    }


def build_feature_row(text_feats: dict, visual: dict, num_questions: int = 1) -> pd.DataFrame:
    """Combine text + visual features into the exact feature row the model expects."""
    row = {
        "total_words":     text_feats["total_words"],
        "total_fillers":   text_feats["total_fillers"],
        "face_detected":   int(visual.get("face_detected", 1)),
        "avg_sentiment":   text_feats["avg_sentiment"],
        "avg_words_per_q": text_feats["avg_words_per_q"],
        "filler_ratio":    text_feats["filler_ratio"],
        "eye_contact":     float(visual.get("eye_contact", 0.65)),
        "posture_score":   float(visual.get("posture_score", 0.65)),
        "speech_rate_wpm": text_feats["speech_rate_wpm"],
    }
    return pd.DataFrame([row])[FEATURES]


def label_to_score(label: str, proba: np.ndarray, classes: np.ndarray) -> int:
    """Convert RF label + class probabilities into a 0-100 confidence score."""
    base = LABEL_SCORE.get(label, 55)
    # Weighted centroid: smooth the discrete label into a continuous score
    weights = {"Low": 20, "Medium": 55, "High": 90}
    score = sum(weights.get(c, 55) * p for c, p in zip(classes, proba))
    # Blend base label and weighted probability score
    blended = 0.35 * base + 0.65 * score
    return max(10, min(100, round(blended)))


def generate_feedback(label: str, text_feats: dict) -> dict:
    """Produce actionable strengths / improvements based on features."""
    strengths, improvements = [], []
    wc    = text_feats["total_words"]
    fr    = text_feats["filler_ratio"]
    sent  = text_feats["avg_sentiment"]
    wpm   = text_feats["speech_rate_wpm"]

    # Strengths
    if wc >= 150:   strengths.append("Detailed, well-developed answer — good length.")
    if fr < 0.02:   strengths.append("Minimal filler words — fluent and composed delivery.")
    if sent > 0.2:  strengths.append("Positive framing and confident language tone.")
    if wpm >= 120 and wpm <= 165: strengths.append("Natural speaking pace in the ideal range.")

    # Improvements
    if wc < 50:     improvements.append("Expand your answer — aim for at least 60–80 words per response.")
    if fr > 0.06:   improvements.append(f"Reduce filler words (detected {text_feats['total_fillers']}) — pause instead of saying 'um' or 'like'.")
    if sent < 0.0:  improvements.append("Use more positive, assertive language to project confidence.")
    if wpm < 100:   improvements.append("Speak at a slightly faster pace — aim for 120–160 words per minute.")

    # Always-on tip
    if label == "Low":
        improvements.append("Structure answers using STAR: Situation, Task, Action, Result.")
    elif label == "Medium":
        improvements.append("Back up claims with concrete, specific examples from your experience.")

    tip_map = {
        "Low":    "Focus on clear structure and reducing filler words to project more confidence.",
        "Medium": "Good foundation — add specific examples and assertive language to reach High.",
        "High":   "Excellent confidence level — maintain this consistency across all questions.",
    }
    return {
        "label":        label,
        "tip":          tip_map.get(label, ""),
        "strengths":    strengths[:3],
        "improvements": improvements[:3],
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
app   = Flask(__name__)
CORS(app)
model = load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   "RandomForestClassifier",
        "classes": model.classes_.tolist(),
        "features": FEATURES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Full prediction — accepts text + optional visual features.

    Request JSON:
      {
        "text":          "My answer transcript...",
        "num_questions": 8,          // optional, default 1
        "face_detected": 1,          // optional (0/1)
        "eye_contact":   0.72,       // optional (0.0–1.0)
        "posture_score": 0.68,       // optional (0.0–1.0)
      }

    Response JSON:
      {
        "success":        true,
        "label":          "High",          // Low / Medium / High
        "conf_score":     84,              // 0-100
        "probabilities":  {"Low":0.05,"Medium":0.12,"High":0.83},
        "text_analysis":  { ... },
        "feedback":       { ... }
      }
    """
    data = request.get_json(silent=True) or {}
    text          = str(data.get("text", "")).strip()
    num_questions = int(data.get("num_questions", 1))

    if not text:
        return jsonify({"success": False, "error": "text is required"}), 400

    try:
        t0         = time.perf_counter()
        text_feats = analyse_text(text, num_questions)
        visual     = {
            "face_detected": data.get("face_detected", 1),
            "eye_contact":   data.get("eye_contact",   0.65),
            "posture_score": data.get("posture_score",  0.65),
        }
        row        = build_feature_row(text_feats, visual, num_questions)
        label      = model.predict(row)[0]
        proba      = model.predict_proba(row)[0]
        conf_score = label_to_score(label, proba, model.classes_)
        feedback   = generate_feedback(label, text_feats)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            "success":       True,
            "label":         label,
            "conf_score":    conf_score,
            "probabilities": {c: round(float(p), 3) for c, p in zip(model.classes_, proba)},
            "text_analysis": text_feats,
            "feedback":      feedback,
            "elapsed_ms":    elapsed_ms,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict/text", methods=["POST"])
def predict_text_only():
    """Text-only shortcut — identical to /predict with default visual features."""
    return predict()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀 SwaKrut Confidence API")
    print(f"   POST /predict        — full prediction")
    print(f"   POST /predict/text   — text-only alias")
    print(f"   GET  /health         — liveness check")
    print(f"   Running on http://{HOST}:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=False)