"""
Metrics and analysis module for IoT Anomaly Detection
Includes: F1 scores, accuracy calculations, and heatmaps
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data import SENSOR_COLS
from models import AnomalyEngine


class MetricsCalculator:
    """Calculate and track model performance metrics"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.detection_history = []
        
    def add_detection(self, z_score, isolation_forest, autoencoder, ensemble, is_true_anomaly=None):
        """Record a detection result"""
        self.detection_history.append({
            "timestamp": datetime.now(),
            "z_score": z_score,
            "isolation_forest": isolation_forest,
            "autoencoder": autoencoder,
            "ensemble": ensemble,
            "ground_truth": is_true_anomaly if is_true_anomaly is not None else ensemble
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.window_size:
            self.detection_history = self.detection_history[-self.window_size:]
    
    def calculate_metrics(self, predictions, ground_truth):
        """Calculate precision, recall, F1 score"""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
        
        tp = sum((p == 1) and (g == 1) for p, g in zip(predictions, ground_truth))
        fp = sum((p == 1) and (g == 0) for p, g in zip(predictions, ground_truth))
        tn = sum((p == 0) and (g == 0) for p, g in zip(predictions, ground_truth))
        fn = sum((p == 0) and (g == 1) for p, g in zip(predictions, ground_truth))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "accuracy": round(accuracy, 3),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }
    
    def get_all_metrics(self):
        """Get metrics for all detector models"""
        if len(self.detection_history) == 0:
            return None
        
        df = pd.DataFrame(self.detection_history)
        ground_truth = df["ground_truth"].astype(int).tolist()
        
        metrics = {
            "Z-Score": self.calculate_metrics(df["z_score"].astype(int).tolist(), ground_truth),
            "Isolation Forest": self.calculate_metrics(df["isolation_forest"].astype(int).tolist(), ground_truth),
            "Autoencoder": self.calculate_metrics(df["autoencoder"].astype(int).tolist(), ground_truth),
            "Ensemble": self.calculate_metrics(df["ensemble"].astype(int).tolist(), ground_truth),
        }
        
        return metrics
    
    def get_f1_heatmap_data(self, num_chunks=20):
        """Get F1 scores over time for heatmap visualization"""
        if len(self.detection_history) < 10:
            return None
        
        df = pd.DataFrame(self.detection_history)
        chunk_size = max(1, len(df) // num_chunks)
        
        f1_data = {
            "Z-Score": [],
            "Isolation Forest": [],
            "Autoencoder": [],
            "Ensemble": []
        }
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            ground_truth = chunk["ground_truth"].astype(int).tolist()
            
            f1_data["Z-Score"].append(self.calculate_metrics(chunk["z_score"].astype(int).tolist(), ground_truth)["f1"])
            f1_data["Isolation Forest"].append(self.calculate_metrics(chunk["isolation_forest"].astype(int).tolist(), ground_truth)["f1"])
            f1_data["Autoencoder"].append(self.calculate_metrics(chunk["autoencoder"].astype(int).tolist(), ground_truth)["f1"])
            f1_data["Ensemble"].append(self.calculate_metrics(chunk["ensemble"].astype(int).tolist(), ground_truth)["f1"])
        
        return f1_data
    
    def get_sensor_anomaly_heatmap(self, buffer_df, num_bins=20):
        """Get sensor readings heatmap showing anomalies"""
        if len(buffer_df) < num_bins:
            return None
        
        # Normalize sensor data
        sensor_data = buffer_df[SENSOR_COLS].tail(num_bins * 5).values
        normalized = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-6)
        
        # Bin by time
        bin_size = max(1, len(normalized) // num_bins)
        binned_means = []
        for i in range(0, len(normalized), bin_size):
            binned_means.append(normalized[i:i+bin_size].mean(axis=0))
        
        return np.array(binned_means)


def plot_f1_heatmap(f1_data):
    """Plot F1 score heatmap"""
    if f1_data is None:
        return None
    
    df = pd.DataFrame(f1_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values.T,
        x=range(len(df)),
        y=df.columns,
        colorscale="RdYlGn",
        zmid=0.5
    ))
    
    fig.update_layout(
        title="F1 Score Trends by Detector",
        xaxis_title="Time Chunks",
        yaxis_title="Detector",
        height=300,
        plot_bgcolor='#1A1F28',
        paper_bgcolor='#0F1419',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='rgba(0, 217, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(0, 217, 255, 0.1)'),
        title_font_color='#00D9FF'
    )
    
    return fig


def plot_sensor_heatmap(sensor_data, sensor_cols):
    """Plot sensor anomaly heatmap"""
    if sensor_data is None:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=sensor_data.T,
        y=sensor_cols[:sensor_data.shape[1]],
        colorscale="RdBu",
        zmid=0
    ))
    
    fig.update_layout(
        title="Normalized Sensor Readings Over Time",
        xaxis_title="Time",
        yaxis_title="Sensor",
        height=400,
        plot_bgcolor='#1A1F28',
        paper_bgcolor='#0F1419',
        font=dict(color='#E0E0E0'),
        xaxis=dict(gridcolor='rgba(0, 217, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(0, 217, 255, 0.1)'),
        title_font_color='#00D9FF'
    )
    
    return fig


def plot_metrics_table(metrics):
    """Create a metrics comparison table"""
    if metrics is None:
        return None
    
    data = {
        "Model": list(metrics.keys()),
        "Precision": [metrics[k]["precision"] for k in metrics.keys()],
        "Recall": [metrics[k]["recall"] for k in metrics.keys()],
        "F1 Score": [metrics[k]["f1"] for k in metrics.keys()],
        "Accuracy": [metrics[k]["accuracy"] for k in metrics.keys()],
    }
    
    return pd.DataFrame(data)


def create_metrics_dashboard(metrics_calc, buffer_df):
    """Create a full metrics dashboard in Streamlit"""
    st.set_page_config(page_title="Metrics Dashboard", layout="wide")
    
    # Dark theme styling
    st.markdown("""
    <style>
        body {
            background-color: #0F1419;
            color: #E0E0E0;
        }
        
        h1, h2, h3 {
            color: #00D9FF;
        }
        
        [data-testid="stDataFrame"] {
            background-color: #1A1F28;
            border: 1px solid #00D9FF;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Model Performance Metrics")
    st.write("Real-time performance tracking for all AI detection models")
    
    metrics = metrics_calc.get_all_metrics()
    if metrics:
        # Metrics table
        st.markdown("### Performance Metrics")
        metrics_df = plot_metrics_table(metrics)
        st.dataframe(metrics_df, use_container_width=True)
        
        # F1 heatmap
        st.markdown("### F1 Score Heatmap")
        f1_data = metrics_calc.get_f1_heatmap_data()
        f1_fig = plot_f1_heatmap(f1_data)
        if f1_fig:
            st.plotly_chart(f1_fig, use_container_width=True)
        
        # Sensor heatmap
        st.markdown("### Sensor Anomaly Heatmap")
        sensor_heatmap = metrics_calc.get_sensor_anomaly_heatmap(buffer_df)
        sensor_fig = plot_sensor_heatmap(sensor_heatmap, SENSOR_COLS)
        if sensor_fig:
            st.plotly_chart(sensor_fig, use_container_width=True)
    else:
        st.info("Collecting data... Not enough detections to display metrics yet")


if __name__ == "__main__":
    # Example usage
    calc = MetricsCalculator()
    
    # Simulate some detections
    for i in range(100):
        calc.add_detection(
            z_score=np.random.random() > 0.9,
            isolation_forest=np.random.random() > 0.95,
            autoencoder=np.random.random() > 0.92,
            ensemble=np.random.random() > 0.93
        )
    
    metrics = calc.get_all_metrics()
    print("Metrics:", metrics)
