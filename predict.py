"""Prediction and explanation helpers shared by the CLI and Streamlit UI."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "quality_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        from src.train import train
        train()
    return joblib.load(MODEL_PATH)


def predict_session(model, values: dict) -> tuple[str, dict[str, float]]:
    row = pd.DataFrame([{key: values[key] for key in FEATURES}])
    label = str(model.predict(row)[0])
    probs = model.predict_proba(row)[0]
    return label, {str(c): float(p) for c, p in zip(model.classes_, probs)}
