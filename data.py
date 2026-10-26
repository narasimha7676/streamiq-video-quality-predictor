"""Generate a reproducible, realistic demo dataset (not production telemetry)."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEVICES = ["Smart TV", "Mobile", "Laptop", "Tablet", "Streaming Stick"]
DEVICE_RELIABILITY = {"Smart TV": 0.86, "Mobile": 0.90, "Laptop": 0.94,
                      "Tablet": 0.91, "Streaming Stick": 0.84}
FEATURES = ["latency_ms", "bandwidth_mbps", "packet_loss_pct", "jitter_ms",
            "bitrate_mbps", "device_type", "hour_of_day"]
TARGET = "quality_outcome"
CLASSES = ["Healthy", "Buffering", "Poor Quality", "Playback Failure"]


def make_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Synthesize session-level measurements and outcomes for a portfolio demo."""
    rng = np.random.default_rng(seed)
    device = rng.choice(DEVICES, n_rows, p=[.20, .32, .24, .12, .12])
    hour = rng.integers(0, 24, n_rows)
    busy_hour = np.isin(hour, [18, 19, 20, 21]).astype(float)
    bandwidth = np.clip(rng.lognormal(2.25 - .20 * busy_hour, .55, n_rows), .3, 150)
    latency = np.clip(rng.gamma(2.0, 24, n_rows) + busy_hour * 20, 5, 500)
    jitter = np.clip(rng.gamma(2.0, 5, n_rows) + busy_hour * 4, .1, 120)
    loss = np.clip(rng.beta(1.2, 30, n_rows) * 8 + busy_hour * .16, 0, 12)
    # The requested bitrate is partly adapted to available bandwidth, but some
    # sessions are deliberately over-ambitious (e.g., stale ABR estimates).
    bitrate = np.clip(bandwidth * rng.uniform(.45, 1.45, n_rows), .2, 35)

    # Risk score models intuitive streaming constraints with modest noise.
    headroom = bandwidth / np.maximum(bitrate, .1)
    risk = (1.25 * np.maximum(0, 1.25 - headroom)
            + .012 * np.maximum(0, latency - 80)
            + .055 * jitter + .22 * loss
            + np.array([1 - DEVICE_RELIABILITY[d] for d in device]) * 2.0
            + rng.normal(0, .38, n_rows))
    outcome = np.select(
        [risk < .75, risk < 1.55, risk < 2.75],
        [CLASSES[0], CLASSES[1], CLASSES[2]], default=CLASSES[3]
    )
    # Some high-latency sessions lower quality without failing; small label
    # noise keeps this synthetic demo from being a trivial threshold lookup.
    flip = rng.random(n_rows) < .035
    idx = rng.integers(0, len(CLASSES), n_rows)
    outcome = np.where(flip, np.array(CLASSES)[idx], outcome)
    return pd.DataFrame({
        "latency_ms": latency.round(1), "bandwidth_mbps": bandwidth.round(2),
        "packet_loss_pct": loss.round(3), "jitter_ms": jitter.round(2),
        "bitrate_mbps": bitrate.round(2), "device_type": device,
        "hour_of_day": hour, TARGET: outcome,
    })
