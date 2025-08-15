"""
EV Threat Detection - Detect attacks from dataset CSV without label column
Usage: python detect.py dataset.csv
"""

import sys
import os
import time
import pandas as pd
import joblib
import warnings
import winsound
from termcolor import colored
from colorama import init

# Init colorama
init(autoreset=True)

# Suppress feature-name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load models
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("best_model.joblib")
label_enc = joblib.load("label_encoder.joblib")

required_cols = list(preprocessor.feature_names_in_)
LABEL_COLS = {"label", "Label", "attack_type", "Attack", "target"}

def prepare_df(df):
    df = df.drop(columns=[c for c in df.columns if c in LABEL_COLS], errors="ignore")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "hour" in required_cols and "hour" not in df.columns:
            df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
        if "minute" in required_cols and "minute" not in df.columns:
            df["minute"] = df["timestamp"].dt.minute.fillna(0).astype(int)
        if "interval_ms" in required_cols and "interval_ms" not in df.columns:
            if "charger_id" in df.columns:
                df = df.sort_values(["charger_id", "timestamp"])
                ts_diff = df.groupby("charger_id")["timestamp"].diff()
                df["interval_ms"] = ts_diff.dt.total_seconds().fillna(0) * 1000
            else:
                df["interval_ms"] = 0
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    return df[required_cols]

def play_attack_sound(label):
    sounds = {
        "spoofing": "spoofing.wav",
        "mitm": "mitm.wav",
        "dos": "dos.wav"
    }
    sound_file = sounds.get(label.lower(), None)
    if sound_file and os.path.exists(sound_file):
        end_time = time.time() + 5  # 5 seconds of playback
        while time.time() < end_time:
            winsound.PlaySound(sound_file, winsound.SND_FILENAME)
    else:
        # fallback beep for missing sound
        print("\a", end="", flush=True)
        time.sleep(1)

def detect_from_df(df):
    df = prepare_df(df)
    X = preprocessor.transform(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    for i, (p, prob_row) in enumerate(zip(preds, probs), start=1):
        label = label_enc.inverse_transform([p])[0]
        prob = max(prob_row)
        color = "green" if label.lower() == "normal" else "red" if label.lower() in ["spoofing","mitm","dos"] else "yellow"
        print(f"Row {i}: Prediction = {colored(label, color)} | Probability = {prob:.4f}")
        if label.lower() in ["spoofing", "mitm", "dos"]:
            play_attack_sound(label)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect.py dataset.csv")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    df = pd.read_csv(path)
    detect_from_df(df)
