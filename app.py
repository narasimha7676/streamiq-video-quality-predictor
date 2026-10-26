"""Interactive streaming quality demo: streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import CLASSES, DEVICES, FEATURES, make_dataset
from src.predict import load_model, predict_session

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="StreamIQ | Quality Predictor", page_icon="📺", layout="wide")

st.title("📺 StreamIQ — Video Session Quality Predictor")
st.caption("Estimate playback risk from network, bitrate, device, and time-of-day signals.")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()
left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("Session conditions")
    with st.form("session_form"):
        c1, c2 = st.columns(2)
        latency = c1.slider("Network latency (ms)", 5, 500, 55)
        bandwidth = c2.slider("Available bandwidth (Mbps)", .3, 100.0, 18.0, .1)
        jitter = c1.slider("Jitter (ms)", .1, 120.0, 8.0, .1)
        loss = c2.slider("Packet loss (%)", 0.0, 12.0, .2, .05)
        bitrate = c1.slider("Requested bitrate (Mbps)", .2, 35.0, 8.0, .1)
        device = c2.selectbox("Device", DEVICES)
        hour = st.slider("Hour of day", 0, 23, 20)
        submitted = st.form_submit_button("Predict session quality", type="primary")

with right:
    st.subheader("Prediction")
    if submitted:
        values = {"latency_ms": latency, "bandwidth_mbps": bandwidth,
                  "packet_loss_pct": loss, "jitter_ms": jitter,
                  "bitrate_mbps": bitrate, "device_type": device,
                  "hour_of_day": hour}
        label, probs = predict_session(model, values)
        emoji = {"Healthy": "✅", "Buffering": "⏳", "Poor Quality": "📉", "Playback Failure": "🛑"}[label]
        st.markdown(f"### {emoji} {label}")
        st.write(f"Estimated chance of this outcome: **{probs[label]:.0%}**")
        st.progress(probs[label])
        st.caption("Probabilities are model estimates on synthetic demo data; use as directional signals only.")
        st.markdown("**Class probabilities**")
        st.bar_chart(pd.Series(probs).reindex(CLASSES).rename("Probability"))
        headroom = bandwidth / max(bitrate, .1)
        st.info(f"Bandwidth / bitrate headroom: **{headroom:.2f}×**. A value near or below 1× leaves little room for stable playback.")
    else:
        st.info("Adjust the session inputs, then predict to see the most likely playback outcome.")
        st.markdown("**What the model considers**")
        st.write("Latency, jitter, packet loss, available bandwidth relative to requested bitrate, device type, and time of day.")

st.divider()
overview, quality, metrics_tab = st.tabs(["Dataset overview", "Quality drivers", "Model evaluation"])
with overview:
    df = make_dataset()
    counts = df["quality_outcome"].value_counts().reindex(CLASSES)
    a, b = st.columns([1, 2])
    a.metric("Synthetic sessions", f"{len(df):,}")
    a.metric("Input signals", len(FEATURES))
    b.bar_chart(counts.rename("Sessions"))
    st.caption("The project creates a repeatable synthetic dataset; it does not contain subscriber or company data.")
with quality:
    st.write("The feature importance chart shows which inputs most influenced splits in the fitted random forest. Importance is predictive, not causal.")
    try:
        names = model.named_steps["features"].get_feature_names_out()
        vals = model.named_steps["classifier"].feature_importances_
        imp = pd.Series(vals, index=names).sort_values().tail(12)
        st.bar_chart(imp.rename("Feature importance"))
    except Exception as exc:
        st.warning(f"Could not load feature importances: {exc}")
with metrics_tab:
    metric_path = ROOT / "artifacts" / "metrics.json"
    if metric_path.exists():
        metrics = json.loads(metric_path.read_text())
        a, b = st.columns(2)
        a.metric("Held-out accuracy", f"{metrics['accuracy']:.1%}")
        b.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=CLASSES, columns=CLASSES)
        st.write("Confusion matrix (rows = actual, columns = predicted)")
        st.dataframe(cm, use_container_width=True)
        st.caption(metrics["validation_note"])
    else:
        st.warning("Run `python -m src.train` to create the model evaluation report.")

st.caption("Portfolio prototype • Synthetic data • Not a production QoE monitoring system")
