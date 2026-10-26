# StreamIQ: Intelligent Video Streaming Quality Predictor

An interview-ready portfolio prototype that predicts the likely outcome of a video streaming session from network conditions, requested bitrate, device type, and time of day. It includes a reproducible synthetic dataset, a scikit-learn classification pipeline, an interactive Streamlit demo, and a held-out evaluation report.

> **Data honesty:** The included records are generated for demonstration. They are not measurements from TO THE NEW, a streaming provider, or real subscribers. The reported metrics measure performance on a held-out split of this synthetic dataset and must not be presented as real-world performance.

## Demo features

- Four outcomes: **Healthy**, **Buffering**, **Poor Quality**, and **Playback Failure**.
- Inputs: latency, bandwidth, packet loss, jitter, requested bitrate, device, and hour.
- Displays predicted class probabilities and bandwidth-to-bitrate headroom.
- Dataset overview, feature importance, confusion matrix, accuracy, and macro F1.
- Deterministic data generation (seed 42) makes the demo repeatable.

## Run locally

Python 3.12 is recommended (`.python-version` pins this for hosted deployments). On Windows, double-click `run_demo.bat` to create a virtual environment, install dependencies, train if needed, and launch the app. The first run needs internet access to install packages.

To run manually:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Open the local URL printed by Streamlit. The training command writes the generated CSV to `data/streaming_sessions.csv` and the model and metrics to `artifacts/` (the model file is git-ignored and is generated automatically by the app if missing).

## Repository layout

```text
app.py                   Streamlit interface
src/data.py              Synthetic session data generator
src/train.py             preprocessing, training, evaluation, artifacts
src/predict.py           shared inference helper
data/streaming_sessions.csv   included synthetic demo sessions
artifacts/metrics.json        included held-out evaluation summary
artifacts/quality_model.joblib generated model (git-ignored)
INTERVIEW_GUIDE.md       setup, demo cases, speaking script, troubleshooting
run_demo.bat             Windows one-click setup and launch
```

## Modeling approach

The feature pipeline one-hot encodes device type and passes numeric network/session measurements to a class-balanced Random Forest classifier. Stratified train/test splitting keeps all four outcomes represented in evaluation. Accuracy gives an overall score; macro F1 weights each class equally, which is useful when class counts differ. The confusion matrix highlights which neighboring outcomes the model confuses.

Random Forest is used in the runnable baseline because it handles nonlinear feature interactions with little tuning and does not require feature scaling. XGBoost or LightGBM can be compared as a follow-on experiment once real labeled telemetry is available. Feature importance describes model reliance, not causality.

## How to explain it in an interview

**30-second version:** “I built a small QoE prediction prototype for video streaming. It takes session signals such as latency, jitter, packet loss, bandwidth, requested bitrate, and device type, and predicts whether playback is healthy, buffering, degraded, or likely to fail. I made the pipeline reproducible, evaluated it on a stratified holdout, and put the model behind a Streamlit demo. Since I did not have real provider telemetry, I clearly marked the dataset as synthetic and treat the scores as a functional demo, not a production claim.”

**Walkthrough:** Start the app, predict a healthy session (e.g. 55 ms latency, 18 Mbps bandwidth, 8 Mbps bitrate), then constrain bandwidth below the requested bitrate and raise jitter/loss. Show how the class probabilities change. Open **Quality drivers** to discuss feature importance, then **Model evaluation** to explain the holdout metrics and confusion matrix.

**Likely questions:**

1. **Why classify outcomes?** It gives operations a concise risk category; in production I would also consider predicting continuous QoE metrics or time-to-rebuffer.
2. **Why macro F1?** Each outcome gets equal weight, so a model cannot look good by only predicting the majority class.
3. **What would change for production?** Use consented, privacy-reviewed session telemetry with clear label definitions; split by time and possibly user/device to avoid leakage; compare calibrated XGBoost/LightGBM baselines; monitor drift, class recall, calibration, and provider/device slices.
4. **How could this help a streaming team?** It can flag at-risk sessions early, guide adaptive bitrate decisions, or help triage regional/device issues. Those are proposed uses, not validated outcomes of this demo.

## Limitations and next steps

- Synthetic labels are generated from a heuristic; models can learn that heuristic rather than real playback behavior.
- No real QoE labels, session sequences, CDN/region signal, content encoding, or adaptive bitrate history are included.
- A random split is appropriate for this reproducible demo, but a production evaluation should use a time-based split and test generalization across devices, users, and regions.
- Probability calibration, threshold selection, latency monitoring, fairness/accessibility review, and privacy controls would be required before operational use.
- Next: replace synthetic data with authorized telemetry, define measurable labels (for example rebuffer ratio and startup delay), establish a temporal baseline, then compare tree boosting models.

## GitHub upload

Create an empty public or private GitHub repository. This folder is already initialized as a Git repository. Configure your own Git author details if Git requests them, then from this folder:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
git add .
git commit -m "Build streaming quality prediction demo"
git remote add origin https://github.com/YOUR_USERNAME/streamiq-quality-predictor.git
git push -u origin main
```

Before sharing, personalize the repository URL and add your name/contact if desired. Do not claim access to company or subscriber data.

### Optional: deploy the demo

After pushing to a public GitHub repository, create a Streamlit Community Cloud app connected to that repository. Set the app's main file to `app.py`; `.python-version` and `requirements.txt` provide the runtime and dependencies. Keep the model out of GitHub—the app trains it on first launch. Review the synthetic-data disclosure before sharing the demo link.
