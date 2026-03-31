# ⚡ Industrial IoT Anomaly Detection — Hackathon MVP

A fully local, real-time anomaly detection system for industrial sensor data.
No Kafka. No heavy databases. Runs on a laptop.

---

## Project Structure

```
iot_anomaly/
├── data.py       # Synthetic streaming data generator + cleaning
├── models.py     # Three anomaly detectors + explainability
├── app.py        # Streamlit real-time dashboard
└── retrain_log.csv  (auto-created when you mark false positives)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install streamlit pandas scikit-learn tensorflow
```

> **Python 3.9–3.11 recommended.** TensorFlow requires 64-bit Python.

### 2. Run the dashboard

```bash
# From inside the iot_anomaly/ folder:
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Architecture

### `data.py` — Data Pipeline
| Function | Purpose |
|---|---|
| `generate_row()` | Produces 1 row of 50-sensor data per call. Injects random anomaly spikes (2%) and NaN dropouts (3%) |
| `stream_data(buffer, row)` | Appends row to rolling buffer (max 3 600 rows ≈ 1 hour) |
| `clean_data(df)` | `ffill` → `bfill` → baseline-mean fallback to eliminate all NaNs |

### `models.py` — AI Models
| Class | Anomaly Type | Technique |
|---|---|---|
| `RollingZScore` | **Contextual** | Per-sensor z-score over 60s rolling window; alert if \|z\| > 3 |
| `IsoForestDetector` | **Point** | `IsolationForest` across all 50 sensors simultaneously |
| `AutoencoderDetector` | **Collective** | Dense AE (50→32→16→32→50); alert if mean RE > threshold |

**Root-Cause Explainability**: when the autoencoder fires, the sensor with the highest individual reconstruction error is returned as the `root_cause`.

`AnomalyEngine` wraps all three into a single `.run(df)` call.

### `app.py` — Streamlit Dashboard
- **Real-time line charts** for 3 representative sensors (120s rolling window)
- **Per-sensor RE bar chart** showing which sensors are driving autoencoder alerts
- **Big system status indicator** (green 🟢 / red 🔴 with animation)
- **4 live metrics**: max z-score, IsoForest score, autoencoder RE, root-cause sensor
- **Alert log** — last 200 anomaly events with type and root cause
- **"Mark False Positive" button** — appends the current row to `retrain_log.csv`

---

## Adaptive Learning

Every time you click **"Mark Current Row as False Positive"**, the cleaned
sensor row is appended to `retrain_log.csv`.  This file accumulates labelled
normal data that can be used to:

1. **Filter training data** — exclude logged rows from IsolationForest training
2. **Fine-tune the autoencoder** — re-train on rows confirmed as normal
3. **Adjust thresholds** — statistically derive better RE/z-score cutoffs

---

## Tuning Tips

| Parameter | Location | Effect |
|---|---|---|
| `anomaly_probability` | `app.py → tick()` | ↑ = more synthetic anomalies injected |
| `window`, `threshold` | `RollingZScore.__init__` | Shorter window = more sensitive to recent spikes |
| `contamination` | `IsoForestDetector.__init__` | Expected anomaly fraction (default 2%) |
| `re_threshold` | `AutoencoderDetector.__init__` | Lower = more AE alerts |
| `retrain_every` | `AnomalyEngine.__init__` | How often (in ticks) the AE is retrained |
