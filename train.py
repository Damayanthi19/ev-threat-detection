# train.py
import matplotlib
matplotlib.use('Agg')  # non-GUI backend

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

# =========================
# 1) Load data
# =========================
DATA = "preprocessed_dataset.csv"
df = pd.read_csv(DATA)

# Plot label distribution (before balancing)
plt.figure(figsize=(6, 4))
df["label"].value_counts().sort_index().plot(kind="bar")
plt.title("Label Distribution (Before Balancing)")
plt.xlabel("Encoded label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution_before.png", bbox_inches="tight")
plt.close()

# =========================
# 2) Features / target
# =========================
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 3) Balance with SMOTE
# =========================
print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Plot label distribution after balancing
plt.figure(figsize=(6, 4))
pd.Series(y_train).value_counts().sort_index().plot(kind="bar")
plt.title("Label Distribution (After SMOTE)")
plt.xlabel("Encoded label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution_after.png", bbox_inches="tight")
plt.close()

# =========================
# 4) Probability calibration helper
# =========================
# If per-class counts are small, isotonic can overfit → use sigmoid (Platt).
min_per_class = min(Counter(y_train).values())
calib_method = "isotonic" if min_per_class >= 30 else "sigmoid"
calib_cv = 3  # keep small due to small dataset size

def calibrate(est):
    """Wrap estimator in a CalibratedClassifierCV (skip for LogisticRegression)."""
    if isinstance(est, LogisticRegression):
        # LR already outputs well-calibrated probabilities typically
        return est
    return CalibratedClassifierCV(estimator=est, method=calib_method, cv=calib_cv)

# =========================
# 5) Candidate models (with smoothing/regularization)
# =========================
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=3000, random_state=42, class_weight="balanced", n_jobs=None
    ),
    "RandomForest": calibrate(RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_depth=12,            # limit depth to avoid pure leaves
        min_samples_leaf=5,      # smooth leaves
        class_weight="balanced_subsample"
    )),
    "GradientBoosting": calibrate(GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3
    )),
    "DecisionTree": calibrate(DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced"
    )),
}

# =========================
# 6) Train & evaluate
# =========================
results = {}
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", round(acc, 4))
    rep = classification_report(y_test, preds, zero_division=0)
    print(rep)
    # macro F1 for selection
    f1_macro = classification_report(y_test, preds, output_dict=True)["macro avg"]["f1-score"]
    results[name] = {"model": model, "f1_macro": f1_macro}

best_name = max(results, key=lambda k: results[k]["f1_macro"])
best_model = results[best_name]["model"]
print("\nBest model:", best_name)
joblib.dump(best_model, "best_model.joblib")

# =========================
# 7) Confusion matrix
# =========================
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

# =========================
# 8) Feature importances (if available)
# CalibratedClassifierCV wraps the estimator; try to extract inner estimator.
# =========================
raw_estimator = best_model
try:
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(best_model, CalibratedClassifierCV):
        # take the first calibrated clone's estimator
        raw_estimator = best_model.calibrated_classifiers_[0].estimator
except Exception:
    pass

if hasattr(raw_estimator, "feature_importances_"):
    importances = raw_estimator.feature_importances_
    feat_names = np.array(X.columns)
    top_idx = np.argsort(importances)[-20:][::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(feat_names[top_idx][::-1], importances[top_idx][::-1])
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png", bbox_inches="tight")
    plt
