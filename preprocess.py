# preprocess.py
import pandas as pd
import json

RAW = "dataset.csv"
FEATURES_JSON = "feature_names.json"
TARGET = "label"

df = pd.read_csv(RAW)
print("Raw shape:", df.shape)
print(df.columns.tolist())

# Derive time features
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "charger_id" in df.columns:
        df = df.sort_values(["charger_id", "timestamp"])
        inter = df.groupby("charger_id")["timestamp"].diff().dt.total_seconds() * 1000
        df["interval_ms"] = inter.fillna(inter.median())
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

drop_cols = [TARGET]
if "timestamp" in df.columns:
    drop_cols.append("timestamp")

num_cols = df.drop(columns=drop_cols).select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.drop(columns=drop_cols).select_dtypes(include=["object", "category"]).columns.tolist()

feature_info = {"numeric": num_cols, "categorical": cat_cols}

with open(FEATURES_JSON, "w") as f:
    json.dump(feature_info, f)

print("Feature info saved:", FEATURES_JSON)
