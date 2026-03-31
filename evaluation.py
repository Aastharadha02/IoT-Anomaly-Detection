import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from data import generate_sensor_data
from models import AnomalyEngine
import time

# Set up dark theme
st.set_page_config(
    page_title="Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    body {
        background-color: #0F1419;
        color: #E0E0E0;
    }
    .stMetric {
        background-color: #1A1F28;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00D9FF;
    }
    h1, h2, h3 {
        color: #00D9FF;
    }
</style>
""", unsafe_allow_html=True)

st.title("Model Evaluation Dashboard")
st.subheader("F1 Score, Accuracy & Confusion Matrix Analysis")

# Parameters
st.sidebar.header("Configuration")
num_samples = st.sidebar.slider("Number of samples to analyze", 100, 5000, 1000, 100)
contamination_rate = st.sidebar.slider("Anomaly contamination rate", 0.01, 0.5, 0.05, 0.01)

if st.sidebar.button("Run Evaluation", key="eval_button"):
    
    st.info("Generating test data and running models...")
    
    # Generate data
    with st.spinner("Generating sensor data..."):
        data_list = []
        anomalies_list = []
        
        for i in range(num_samples):
            data, anomalies = generate_sensor_data(contamination=contamination_rate)
            data_list.append(data)
            anomalies_list.append(anomalies)
        
        all_data = pd.concat(data_list, ignore_index=True)
        all_anomalies = np.concatenate(anomalies_list)
    
    st.success(f"Generated {len(all_data)} samples with {sum(all_anomalies)} anomalies ({sum(all_anomalies)/len(all_anomalies)*100:.2f}%)")
    
    # Run anomaly detection
    with st.spinner("Running anomaly detection models..."):
        engine = AnomalyEngine()
        predictions_list = []
        
        for idx, row in all_data.iterrows():
            sensor_values = row.drop(['timestamp']).values
            preds = engine.detect(sensor_values)
            predictions_list.append(preds)
        
        predictions_df = pd.DataFrame(predictions_list)
    
    st.success("Anomaly detection completed!")
    
    # Calculate metrics for each model
    col1, col2, col3, col4 = st.columns(4)
    
    models = ['z_anom', 'if_anom', 'ae_anom']
    model_names = ['Z-Score', 'Isolation Forest', 'Autoencoder']
    metrics_data = []
    
    for model_col, model_name in zip(models, model_names):
        y_pred = predictions_df[model_col].values
        
        f1 = f1_score(all_anomalies, y_pred, zero_division=0)
        accuracy = accuracy_score(all_anomalies, y_pred)
        precision = precision_score(all_anomalies, y_pred, zero_division=0)
        recall = recall_score(all_anomalies, y_pred, zero_division=0)
        
        metrics_data.append({
            'Model': model_name,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })
    
    # Ensemble predictions (2+ models agree)
    ensemble_pred = (
        (predictions_df['z_anom'].astype(int) + 
         predictions_df['if_anom'].astype(int) + 
         predictions_df['ae_anom'].astype(int)) >= 2
    ).astype(int).values
    
    f1_ens = f1_score(all_anomalies, ensemble_pred, zero_division=0)
    accuracy_ens = accuracy_score(all_anomalies, ensemble_pred)
    precision_ens = precision_score(all_anomalies, ensemble_pred, zero_division=0)
    recall_ens = recall_score(all_anomalies, ensemble_pred, zero_division=0)
    
    metrics_data.append({
        'Model': 'Ensemble (2+ vote)',
        'F1 Score': f1_ens,
        'Accuracy': accuracy_ens,
        'Precision': precision_ens,
        'Recall': recall_ens
    })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
    
    # Display individual metrics as cards
    col1, col2, col3, col4 = st.columns(4)
    best_f1_idx = metrics_df['F1 Score'].idxmax()
    best_f1_model = metrics_df.loc[best_f1_idx, 'Model']
    
    with col1:
        st.metric("Best F1 Score", f"{metrics_df['F1 Score'].max():.4f}", f"{best_f1_model}")
    with col2:
        st.metric("Avg Accuracy", f"{metrics_df['Accuracy'].mean():.4f}")
    with col3:
        st.metric("Avg Precision", f"{metrics_df['Precision'].mean():.4f}")
    with col4:
        st.metric("Avg Recall", f"{metrics_df['Recall'].mean():.4f}")
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#0F1419')
    
    # Z-Score
    cm_zscore = confusion_matrix(all_anomalies, predictions_df['z_anom'].values)
    sns.heatmap(cm_zscore, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
    axes[0, 0].set_title('Z-Score Confusion Matrix', color='#00D9FF', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('True Label', color='#E0E0E0')
    axes[0, 0].set_xlabel('Predicted Label', color='#E0E0E0')
    axes[0, 0].tick_params(colors='#E0E0E0')
    
    # Isolation Forest
    cm_if = confusion_matrix(all_anomalies, predictions_df['if_anom'].values)
    sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
    axes[0, 1].set_title('Isolation Forest Confusion Matrix', color='#00D9FF', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('True Label', color='#E0E0E0')
    axes[0, 1].set_xlabel('Predicted Label', color='#E0E0E0')
    axes[0, 1].tick_params(colors='#E0E0E0')
    
    # Autoencoder
    cm_ae = confusion_matrix(all_anomalies, predictions_df['ae_anom'].values)
    sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title('Autoencoder Confusion Matrix', color='#00D9FF', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('True Label', color='#E0E0E0')
    axes[1, 0].set_xlabel('Predicted Label', color='#E0E0E0')
    axes[1, 0].tick_params(colors='#E0E0E0')
    
    # Ensemble
    cm_ens = confusion_matrix(all_anomalies, ensemble_pred)
    sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title('Ensemble Confusion Matrix', color='#00D9FF', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('True Label', color='#E0E0E0')
    axes[1, 1].set_xlabel('Predicted Label', color='#E0E0E0')
    axes[1, 1].tick_params(colors='#E0E0E0')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Comparison Chart
    st.subheader("F1 Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0F1419')
    
    models_list = metrics_df['Model'].tolist()
    f1_scores = metrics_df['F1 Score'].tolist()
    
    bars = ax.bar(models_list, f1_scores, color='#00D9FF', edgecolor='#00D9FF', linewidth=2)
    ax.set_ylabel('F1 Score', color='#E0E0E0', fontsize=12)
    ax.set_title('F1 Score by Model', color='#00D9FF', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_facecolor('#1A1F28')
    ax.tick_params(colors='#E0E0E0')
    ax.grid(axis='y', alpha=0.3, color='#00D9FF')
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', color='#00D9FF', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed Statistics Table
    st.subheader("Detailed Statistics")
    
    detailed_stats = []
    for model_col, model_name in zip(models + ['ensemble'], model_names + ['Ensemble (2+ vote)']):
        if model_col == 'ensemble':
            y_pred = ensemble_pred
        else:
            y_pred = predictions_df[model_col].values
        
        cm = confusion_matrix(all_anomalies, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, sum(all_anomalies == y_pred))
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        detailed_stats.append({
            'Model': model_name,
            'True Positives': int(tp),
            'False Positives': int(fp),
            'True Negatives': int(tn),
            'False Negatives': int(fn),
            'Sensitivity': f"{sensitivity:.4f}",
            'Specificity': f"{specificity:.4f}"
        })
    
    detailed_df = pd.DataFrame(detailed_stats)
    st.dataframe(detailed_df, use_container_width=True)
