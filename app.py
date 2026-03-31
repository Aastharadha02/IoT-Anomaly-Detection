import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import time

from data import generate_row, stream_data, clean_data, SENSOR_COLS, N_SENSORS
from models import AnomalyEngine

# Page configuration
st.set_page_config(page_title="IoT Anomaly Detection", layout="wide", initial_sidebar_state="expanded")

# Simple Dark Theme
st.markdown("""
<style>
    * { margin: 0; padding: 0; }
    
    body { 
        background-color: #0F1419;
        color: #E0E0E0;
    }
    
    .main {
        background-color: #0F1419;
        color: #E0E0E0;
    }
    
    h1, h2, h3 {
        color: #00D9FF;
        font-weight: 600;
    }
    
    [data-testid="metric-container"] {
        background-color: #1A1F28;
        border: 1px solid #00D9FF;
        border-radius: 8px;
    }
    
    [data-testid="stDataFrame"] {
        background-color: #1A1F28 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #121619;
    }
    
    hr { border: 1px solid #00D9FF; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame()
if "engine" not in st.session_state:
    st.session_state.engine = AnomalyEngine()
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "anomaly_count" not in st.session_state:
    st.session_state.anomaly_count = 0
if "normal_count" not in st.session_state:
    st.session_state.normal_count = 0

# Sidebar
with st.sidebar:
    st.markdown("### System Info")
    st.markdown(f"**Active Sensors:** {N_SENSORS}")
    st.markdown("**Buffer Capacity:** 100 readings")
    st.markdown("**Detection Models:** 3 (Ensemble)")
    st.markdown("**Update Rate:** 1 sec")

# Main header
st.title("Industrial IoT Anomaly Detection")
st.write("Real-time sensor monitoring with AI-powered anomaly detection")

# Generate new data on each rerun
new_row = generate_row()
st.session_state.buffer = stream_data(st.session_state.buffer, new_row)

# Clean and run detection
buffer_clean = clean_data(st.session_state.buffer)
results = st.session_state.engine.run(buffer_clean)

# Check for anomalies
z_anom = results["zscore"]["anomaly"]
if_anom = results["isoforest"]["anomaly"]
ae_anom = results["autoencoder"]["anomaly"]

is_anomaly = sum([z_anom, if_anom, ae_anom]) >= 2

if is_anomaly:
    st.session_state.anomaly_count += 1
    alert = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type": "ANOMALY",
        "Z-Score": "DETECTED" if z_anom else "OK",
        "Isolation Forest": "DETECTED" if if_anom else "OK",
        "Autoencoder": "DETECTED" if ae_anom else "OK",
        "Root Cause": results["zscore"]["sensor"] or results["autoencoder"]["root_cause"] or "Unknown"
    }
    st.session_state.alerts.append(alert)
else:
    st.session_state.normal_count += 1

# Display section
# Top metrics row
st.markdown("## System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Anomalies Detected", st.session_state.anomaly_count, delta=1 if is_anomaly else 0)
with col2:
    st.metric("Normal Readings", st.session_state.normal_count, delta=1 if not is_anomaly else 0)
with col3:
    st.metric("Buffer Size", len(st.session_state.buffer), delta=1)
with col4:
    status_text = "ANOMALY" if is_anomaly else "NORMAL"
    st.metric("Status", status_text)

st.divider()

# Sensor readings chart section
st.markdown("## Live Sensor Readings")

latest = buffer_clean[SENSOR_COLS].iloc[-1]
mean_val = latest.mean()
std_val = latest.std()

# Color logic: anomalous if > 2 std from mean
colors = []
for x in latest.values:
    if x > mean_val + 2*std_val or x < mean_val - 2*std_val:
        colors.append('#FF4444')  # Red for anomaly
    elif x > mean_val + std_val or x < mean_val - std_val:
        colors.append('#FFA500')  # Orange for warning
    else:
        colors.append('#00D9FF')  # Cyan for normal

fig = go.Figure(data=[
    go.Bar(
        x=SENSOR_COLS, 
        y=latest.values,
        marker=dict(
            color=colors,
            line=dict(color='#00FFFF', width=1)
        ),
        name='Sensor Value',
        text=np.round(latest.values, 2),
        textposition='outside',
        textfont=dict(color='#E0E0E0', size=8)
    )
])

fig.update_layout(
    height=350,
    xaxis_tickangle=-45,
    showlegend=False,
    plot_bgcolor='#1A1F28',
    paper_bgcolor='#0F1419',
    font=dict(color='#E0E0E0', family='Arial'),
    xaxis=dict(
        gridcolor='rgba(0, 217, 255, 0.1)',
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='rgba(0, 217, 255, 0.1)',
        showgrid=True,
        zeroline=False
    ),
    margin=dict(b=100, l=50, r=30, t=30),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# Two column layout for data table and alerts
col_data, col_alerts = st.columns([1, 1])

with col_data:
    st.markdown("### Latest Sensor Data")
    display_df = buffer_clean[SENSOR_COLS].tail(10).copy()
    display_df.index = [f"Reading {i+1}" for i in range(len(display_df))]
    st.dataframe(display_df, use_container_width=True, height=280)

with col_alerts:
    st.markdown("### Alert Log")
    if st.session_state.alerts:
        alerts_df = pd.DataFrame(st.session_state.alerts[-10:])
        st.dataframe(alerts_df, use_container_width=True, height=280)
    else:
        st.info("No anomalies detected - All systems running normally")

# Auto-refresh using Streamlit's rerun mechanism
time.sleep(1)
st.rerun()
