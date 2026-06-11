"""
train_model.py — SwaKrut Confidence Model Trainer
===================================================
PRIMARY:  Downloads Khubaib01/ConfiDetect-Confidence-Posture-Dataset
          from HuggingFace and trains the model on real data.

FALLBACK: If internet / huggingface_hub is unavailable, trains on
          a 600-sample synthetic dataset that mirrors ConfiDetect's
          exact column structure so you can swap it out seamlessly.

Features:
  total_words      <- word_count
  total_fillers    <- filler_word_count
  face_detected    <- derived (eye_contact_score > 0.3)
  avg_sentiment    <- sentiment_score
  avg_words_per_q  <- word_count / 8
  filler_ratio     <- filler_word_count / word_count
  eye_contact      <- eye_contact_score
  posture_score    <- posture_label encoded
  speech_rate_wpm  <- speech_rate_wpm

Target: confidence_label -> Low / Medium / High

Run:
  pip install scikit-learn pandas numpy joblib huggingface_hub
  python train_model.py
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATASET_REPO   = "Khubaib01/ConfiDetect-Confidence-Posture-Dataset"
MODEL_PATH     = "confidence_model.pkl"
REPORT_PATH    = "training_report.txt"
NUM_QUESTIONS  = 8
RANDOM_STATE   = 42

FEATURES = [
    "total_words", "total_fillers", "face_detected",
    "avg_sentiment", "avg_words_per_q", "filler_ratio",
    "eye_contact", "posture_score", "speech_rate_wpm",
]


# ─────────────────────────────────────────────────────────────────
# 1. LOAD REAL DATA FROM HUGGINGFACE
# ─────────────────────────────────────────────────────────────────

def load_real_dataset():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    print(f"[HF] Connecting to {DATASET_REPO} ...")
    loaded = None

    # Try each split and common file formats
    for split in ["train", "validation", "test"]:
        for ext, fn in [("parquet", pd.read_parquet), ("csv", pd.read_csv), ("json", pd.read_json)]:
            for fname in [
                f"data/{split}-00000-of-00001.{ext}",
                f"{split}.{ext}",
            ]:
                try:
                    path = hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset")
                    part = fn(path)
                    loaded = part if loaded is None else pd.concat([loaded, part], ignore_index=True)
                    print(f"  [HF] {split}/{fname}: {len(part)} rows")
                    break
                except Exception:
                    continue

    # Try root-level files
    if loaded is None:
        for fname, fn in [
            ("data.parquet", pd.read_parquet), ("data.csv", pd.read_csv),
            ("dataset.csv",  pd.read_csv),     ("train.csv", pd.read_csv),
            ("ConfiDetect.csv", pd.read_csv),
        ]:
            try:
                path = hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset")
                loaded = fn(path)
                print(f"  [HF] root/{fname}: {len(loaded)} rows")
                break
            except Exception:
                continue

    if loaded is None or len(loaded) == 0:
        raise RuntimeError("No data could be downloaded from HuggingFace.")

    print(f"[HF] Downloaded {len(loaded)} total rows")
    print(f"[HF] Columns: {loaded.columns.tolist()}")
    return map_columns(loaded)


def map_columns(raw):
    col = {c.lower().strip(): c for c in raw.columns}

    def get(variants, default):
        for v in variants:
            if v in col:
                return pd.to_numeric(raw[col[v]], errors="coerce").fillna(default)
        return pd.Series(default, index=raw.index)

    def gets(variants, default):
        for v in variants:
            if v in col:
                return raw[col[v]].astype(str).str.strip()
        return pd.Series(default, index=raw.index)

    # Target
    confidence = gets(["confidence_label", "confidence", "label", "target"], "Medium")
    confidence = confidence.str.capitalize().replace({"0": "Low", "1": "Medium", "2": "High"})

    # Numeric features
    wc   = get(["word_count", "words", "total_words"], 100)
    fc   = get(["filler_word_count", "fillers", "filler_count"], 3)
    ec   = get(["eye_contact_score", "eye_contact", "gaze_score"], 0.5)
    sent = get(["sentiment_score", "sentiment", "polarity"], 0.0)
    wpm  = get(["speech_rate_wpm", "wpm", "speech_rate"], 120)

    # Posture categorical -> numeric
    posture_raw = gets(["posture_label", "posture", "body_posture"], "Neutral")
    pmap = {"Good": 1.0, "Neutral": 0.5, "Poor": 0.0,
            "good": 1.0, "neutral": 0.5, "poor": 0.0, "1": 1.0, "0": 0.0}
    posture = posture_raw.map(pmap).fillna(0.5)

    df = pd.DataFrame({
        "total_words":     wc.values,
        "total_fillers":   fc.values,
        "face_detected":   (ec > 0.3).astype(int).values,
        "avg_sentiment":   sent.values,
        "avg_words_per_q": (wc / NUM_QUESTIONS).round(1).values,
        "filler_ratio":    (fc / wc.clip(lower=1)).round(4).values,
        "eye_contact":     ec.values,
        "posture_score":   posture.values,
        "speech_rate_wpm": wpm.values,
        "confidence":      confidence.values,
    })
    df = df[df["confidence"].isin(["Low", "Medium", "High"])].reset_index(drop=True)
    print(f"[MAP] After label filter: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. SYNTHETIC FALLBACK  (mirrors ConfiDetect schema exactly)
# ─────────────────────────────────────────────────────────────────

def build_synthetic(n=200):
    rng = np.random.default_rng(RANDOM_STATE)

    def cls(label, wc, fc, ec, wpm, post, sent):
        n_ = n
        tw   = rng.integers(*wc,   size=n_).astype(float)
        tf   = rng.integers(*fc,   size=n_).astype(float)
        eye  = rng.uniform(*ec,    size=n_).round(3)
        spd  = rng.uniform(*wpm,   size=n_).round(1)
        pst  = rng.uniform(*post,  size=n_).round(3)
        snt  = rng.uniform(*sent,  size=n_).round(3)
        return pd.DataFrame({
            "total_words":     tw,
            "total_fillers":   tf,
            "face_detected":   (eye > 0.3).astype(int),
            "avg_sentiment":   snt,
            "avg_words_per_q": (tw / NUM_QUESTIONS).round(1),
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
    ], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 3. LOCAL RESPONSES (collected from real sessions)
# ─────────────────────────────────────────────────────────────────

def load_local(path="responses.csv"):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "confidence" not in df.columns:
            return None
        for col, default in [
            ("eye_contact",0.5),("posture_score",0.5),("speech_rate_wpm",120),
            ("avg_sentiment",0.0),("filler_ratio",0.05),("face_detected",1),
            ("avg_words_per_q",15.0),("total_words",120),("total_fillers",4),
        ]:
            if col not in df.columns:
                df[col] = default
        df = df[df["confidence"].isin(["Low","Medium","High"])]
        if len(df) > 0:
            print(f"[LOCAL] {len(df)} rows from {path}")
            return df[FEATURES + ["confidence"]]
    except Exception as e:
        print(f"[LOCAL] Could not read {path}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────────────────────────

def train(df):
    X = df[FEATURES]
    y = df["confidence"]
    print("\nLabel distribution:")
    print(y.value_counts().to_string())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred   = model.predict(X_te)
    acc      = (y_pred == y_te).mean()
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_sc    = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    report   = classification_report(y_te, y_pred)
    cm       = confusion_matrix(y_te, y_pred, labels=["Low","Medium","High"])
    imps     = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    print(f"\nTest Accuracy     : {acc:.2%}")
    print(f"Cross-val (5-fold): {cv_sc.mean():.2%} +/- {cv_sc.std():.2%}")
    print("\nClassification Report:\n" + report)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=["Low","Medium","High"], columns=["Low","Medium","High"]).to_string())
    print("\nFeature Importances:")
    for f, i in imps.items():
        print(f"  {f:<20} {i:.3f}  {'|' * int(i*50)}")

    return model, acc, cv_sc, report, imps


# ─────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────

def save(model, acc, cv_sc, report, imps, source):
    joblib.dump(model, MODEL_PATH)
    print(f"\n[OK] Model -> {MODEL_PATH}")

    lines = [
        "SwaKrut Confidence Model — Training Report",
        "=" * 52, "",
        f"Data source    : {source}",
        f"Features       : {', '.join(FEATURES)}",
        f"Test Accuracy  : {acc:.2%}",
        f"CV Accuracy    : {cv_sc.mean():.2%} +/- {cv_sc.std():.2%}", "",
        "Classification Report:", report,
        "Feature Importances:",
    ] + [f"  {f:<20} {i:.3f}" for f, i in imps.items()] + [
        "", "Usage in app.py:",
        "  model = joblib.load('confidence_model.pkl')",
        "  features = " + str(FEATURES),
        "  pred = model.predict(pd.DataFrame([row])[features])[0]",
    ]
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report -> {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────
# 6. QUICK INFERENCE TEST
# ─────────────────────────────────────────────────────────────────

def inference_test(model):
    print("\n--- Quick Inference Test ---")
    cases = [
        ("Low (expected)",    dict(total_words=28, total_fillers=14, face_detected=0,
             avg_sentiment=-0.2, avg_words_per_q=3.5, filler_ratio=0.50,
             eye_contact=0.12, posture_score=0.10, speech_rate_wpm=65)),
        ("Medium (expected)", dict(total_words=160, total_fillers=5, face_detected=1,
             avg_sentiment=0.2,  avg_words_per_q=20.0, filler_ratio=0.031,
             eye_contact=0.58, posture_score=0.55, speech_rate_wpm=122)),
        ("High (expected)",   dict(total_words=420, total_fillers=1, face_detected=1,
             avg_sentiment=0.55, avg_words_per_q=52.5, filler_ratio=0.002,
             eye_contact=0.88, posture_score=0.90, speech_rate_wpm=155)),
    ]
    for lbl, row in cases:
        pred = model.predict(pd.DataFrame([row])[FEATURES])[0]
        prob = model.predict_proba(pd.DataFrame([row])[FEATURES])[0]
        pstr = " | ".join(f"{c}:{p:.0%}" for c,p in zip(model.classes_, prob))
        ok   = "[OK]" if pred.lower() in lbl.lower() else "[!!]"
        print(f"  {ok} {lbl:<25} -> {pred:<8}  ({pstr})")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  SwaKrut Confidence Model — Training")
    print("=" * 52)

    df, source = None, ""

    # 1. Try HuggingFace
    try:
        df     = load_real_dataset()
        source = f"HuggingFace: {DATASET_REPO} ({len(df)} rows)"
    except ImportError as e:
        print(f"[WARN] {e}")
    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")

    # 2. Merge local real responses
    local = load_local("responses.csv")
    if local is not None:
        df     = pd.concat([df, local], ignore_index=True) if df is not None else local
        source += f" + local ({len(local)} rows)"

    # 3. Synthetic fallback
    if df is None or len(df) < 30:
        print("\n[FALLBACK] Using synthetic dataset (600 samples).")
        print("[FALLBACK] Install huggingface_hub + internet access to use real data.")
        df     = build_synthetic(n=200)
        source = f"Synthetic fallback ({len(df)} rows, ConfiDetect schema)"

    print(f"\nSource  : {source}")
    print(f"Samples : {len(df)}")

    model, acc, cv_sc, report, imps = train(df)
    save(model, acc, cv_sc, report, imps, source)
    inference_test(model)

    print("""
How to use in app.py:
  import joblib, pandas as pd
  model = joblib.load("confidence_model.pkl")
  row = {
      "total_words": 300, "total_fillers": 2, "face_detected": 1,
      "avg_sentiment": 0.4, "avg_words_per_q": 37.5, "filler_ratio": 0.007,
      "eye_contact": 0.75, "posture_score": 0.80, "speech_rate_wpm": 140,
  }
  pred = model.predict(pd.DataFrame([row])[model.feature_names_in_])[0]
""")


if __name__ == "__main__":
    main()
