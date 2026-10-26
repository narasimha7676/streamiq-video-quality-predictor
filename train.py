"""Train and evaluate the demo model. Run: python -m src.train"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import CLASSES, FEATURES, TARGET, make_dataset

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def train() -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    df = make_dataset()
    df.to_csv(ROOT / "data" / "streaming_sessions.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=.2, random_state=42,
        stratify=df[TARGET]
    )
    numeric = [c for c in FEATURES if c != "device_type"]
    prep = ColumnTransformer([
        ("numbers", "passthrough", numeric),
        ("device", OneHotEncoder(handle_unknown="ignore"), ["device_type"]),
    ])
    model = Pipeline([
        ("features", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3, class_weight="balanced",
            max_features="sqrt", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "class_order": CLASSES,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
        "classification_report": report,
        "validation_note": "Held-out split of synthetic data; not real-world performance.",
    }
    joblib.dump(model, ARTIFACTS / "quality_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Trained on {len(X_train):,} sessions; tested on {len(X_test):,}.")
    print(f"Accuracy: {metrics['accuracy']:.3f} | Macro F1: {metrics['macro_f1']:.3f}")
    print("Class recall:", {k: round(v["recall"], 3) for k, v in report.items() if k in CLASSES})
    return metrics


if __name__ == "__main__":
    train()
