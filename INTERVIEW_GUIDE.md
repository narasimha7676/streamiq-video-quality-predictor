# StreamIQ interview demo guide

## Before the interview

1. Install Python 3.12 and Git.
2. Open the project folder and double-click `run_demo.bat`. On first launch it creates a local virtual environment, installs the packages, trains the model, and opens the Streamlit app in your browser. Keep the terminal window open.
3. Try the two cases below and confirm the app loads its **Dataset overview**, **Quality drivers**, and **Model evaluation** tabs.
4. If the app is already running, open the address printed in the terminal (usually `http://localhost:8501`).

## Live demo cases

Use these values in the form. They are known examples from the included trained demo model, so the predicted classes should be stable after retraining with the committed seed.

| Signal | Healthy example | High-risk example |
| --- | ---: | ---: |
| Latency | 55 ms | 220 ms |
| Bandwidth | 18 Mbps | 2 Mbps |
| Packet loss | 0.2% | 3% |
| Jitter | 8 ms | 45 ms |
| Requested bitrate | 8 Mbps | 12 Mbps |
| Device | Laptop | Smart TV |
| Hour | 14 | 20 |
| Expected prediction | Healthy | Playback Failure |

For the high-risk case, point out the bandwidth-to-bitrate headroom: `2 / 12 = 0.17×`. The session is requesting substantially more bitrate than the available capacity, and latency, jitter, and packet loss are also elevated. The model's output is a risk estimate, not a certainty.

## Short presentation script

“StreamIQ is a prototype for predicting video session quality. A session can be affected by network delay, jitter, packet loss, available bandwidth, the requested bitrate, device, and time of day. I generate a reproducible demo dataset, encode the categorical device feature, and fit a class-balanced Random Forest to predict four outcomes. The app shows class probabilities, a capacity-to-bitrate check, feature importance, and held-out metrics. I did not have access to real streaming telemetry, so the data and scores are explicitly synthetic. For production, I would validate label definitions with QoE teams, use authorized session telemetry, evaluate on future time windows and unseen devices or regions, and compare a calibrated boosting model.”

## What to emphasize

- **Problem:** Surface sessions at risk of degraded playback early enough to investigate or adapt bitrate.
- **Inputs:** Network and playback context collected per session.
- **Evaluation:** Stratified holdout; macro F1 complements accuracy when class counts differ; inspect per-class recall and confusion matrix.
- **Tradeoff:** The current baseline is intentionally easy to run and explain. XGBoost/LightGBM is a sensible comparison after obtaining real labels; a more complex model cannot fix synthetic or poorly defined labels.
- **Responsible claims:** This is a portfolio prototype, not a deployed TO THE NEW solution, and it uses no company or subscriber data.

## Troubleshooting

- **`py -3.12` is not found:** Install Python 3.12 from python.org and enable the Python launcher, then run `run_demo.bat` again.
- **Port 8501 is busy:** Stop the other Streamlit process or run `.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` from the project folder.
- **Need to retrain:** Run `.venv\Scripts\python.exe -m src.train`; this deterministically regenerates the dataset, model, and metrics.
- **App opens but predictions fail:** Re-train using the same environment that runs Streamlit so the installed scikit-learn version matches the saved model.
