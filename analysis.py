import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import json
from datetime import datetime
from data import generate_row, stream_data, clean_data, SENSOR_COLS
from models import AnomalyEngine

# Page configuration
st.set_page_config(page_title="Anomaly Detection Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom dark theme CSS
st.markdown("""
    <style>
        :root {
            --primary-bg: #0F1419;
            --secondary-bg: #1A1F28;
            --accent-color: #00D9FF;
            --text-color: #E0E0E0;
            --danger: #FF4444;
            --success: #00FF88;
            --warning: #FFA500;
        }
        
        body {
            background-color: var(--primary-bg);
            color: var(--text-color);
        }
        
        .stMetric {
            background-color: var(--secondary-bg);
            border: 1px solid var(--accent-color);
            border-radius: 8px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
    st.session_state.data_buffer = pd.DataFrame()
    st.session_state.engine = AnomalyEngine()

# Sidebar
with st.sidebar:
    st.markdown("### Analysis Configuration")
    simulation_length = st.slider("Simulation Iterations", 100, 1000, 500)
    run_simulation = st.button("Run Full Simulation", key="run_sim")

# Header
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #00D9FF; margin-bottom: 10px;">Anomaly Detection Analysis</h1>
        <p style="color: #FFA500; font-size: 18px;">F1 Score, Accuracy & Confusion Matrix Analysis</p>
    </div>
""", unsafe_allow_html=True)

def calculate_ground_truth(detection_results):
    """
    Use ensemble voting as ground truth.
    If 2+ detectors agree on anomaly, it's treated as TRUE anomaly.
    """
    zscore_anom = detection_results.get('zscore', {}).get('anomaly', False)
    iforest_anom = detection_results.get('isoforest', {}).get('anomaly', False)
    autoenc_anom = detection_results.get('autoencoder', {}).get('anomaly', False)
    
    votes = sum([zscore_anom, iforest_anom, autoenc_anom])
    return votes >= 2

def run_analysis_simulation(num_iterations):
    """Run simulation and collect detection history"""
    st.info(f"Running simulation for {num_iterations} iterations...")
    progress_bar = st.progress(0)
    
    for i in range(num_iterations):
        # Generate new data
        new_row = generate_row()
        st.session_state.data_buffer = stream_data(st.session_state.data_buffer, new_row)
        
        # Clean and detect
        clean_df = clean_data(st.session_state.data_buffer)
        results = st.session_state.engine.run(clean_df)
        
        # Calculate ground truth
        ground_truth = calculate_ground_truth(results)
        
        # Store detection data
        detection_record = {
            'timestamp': new_row['timestamp'].iloc[0],
            'iteration': i,
            'zscore_pred': results['zscore']['anomaly'],
            'iforest_pred': results['isoforest']['anomaly'],
            'autoenc_pred': results['autoencoder']['anomaly'],
            'ground_truth': ground_truth,
            'zscore_score': results['zscore'].get('max_z', 0),
            'iforest_score': results['isoforest'].get('score', 0),
            'autoenc_score': results['autoencoder'].get('total_re', 0),
        }
        
        st.session_state.detection_history.append(detection_record)
        progress_bar.progress((i + 1) / num_iterations)
    
    st.success("Simulation complete!")
    return pd.DataFrame(st.session_state.detection_history)

# Run simulation if button clicked
if run_simulation and len(st.session_state.detection_history) < simulation_length:
    history_df = run_analysis_simulation(simulation_length - len(st.session_state.detection_history))
else:
    history_df = pd.DataFrame(st.session_state.detection_history)

# Display metrics if history exists
if len(history_df) > 0:
    # Tab interface
    tab1, tab2, tab3, tab4 = st.tabs(["Overall Metrics", "Confusion Matrices", "Score Distributions", "Detector Comparison"])
    
    with tab1:
        st.markdown("### Overall Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # Z-Score Metrics
        with col1:
            st.markdown("#### Z-Score Detector")
            zscore_f1 = f1_score(history_df['ground_truth'], history_df['zscore_pred'])
            zscore_acc = accuracy_score(history_df['ground_truth'], history_df['zscore_pred'])
            zscore_prec = precision_score(history_df['ground_truth'], history_df['zscore_pred'], zero_division=0)
            zscore_rec = recall_score(history_df['ground_truth'], history_df['zscore_pred'], zero_division=0)
            
            st.metric("F1 Score", f"{zscore_f1:.4f}", delta=f"{zscore_f1*100:.2f}%")
            st.metric("Accuracy", f"{zscore_acc:.4f}", delta=f"{zscore_acc*100:.2f}%")
            st.metric("Precision", f"{zscore_prec:.4f}", delta=f"{zscore_prec*100:.2f}%")
            st.metric("Recall", f"{zscore_rec:.4f}", delta=f"{zscore_rec*100:.2f}%")
        
        # Isolation Forest Metrics
        with col2:
            st.markdown("#### Isolation Forest Detector")
            iforest_f1 = f1_score(history_df['ground_truth'], history_df['iforest_pred'])
            iforest_acc = accuracy_score(history_df['ground_truth'], history_df['iforest_pred'])
            iforest_prec = precision_score(history_df['ground_truth'], history_df['iforest_pred'], zero_division=0)
            iforest_rec = recall_score(history_df['ground_truth'], history_df['iforest_pred'], zero_division=0)
            
            st.metric("F1 Score", f"{iforest_f1:.4f}", delta=f"{iforest_f1*100:.2f}%")
            st.metric("Accuracy", f"{iforest_acc:.4f}", delta=f"{iforest_acc*100:.2f}%")
            st.metric("Precision", f"{iforest_prec:.4f}", delta=f"{iforest_prec*100:.2f}%")
            st.metric("Recall", f"{iforest_rec:.4f}", delta=f"{iforest_rec*100:.2f}%")
        
        # Autoencoder Metrics
        with col3:
            st.markdown("#### Autoencoder Detector")
            autoenc_f1 = f1_score(history_df['ground_truth'], history_df['autoenc_pred'])
            autoenc_acc = accuracy_score(history_df['ground_truth'], history_df['autoenc_pred'])
            autoenc_prec = precision_score(history_df['ground_truth'], history_df['autoenc_pred'], zero_division=0)
            autoenc_rec = recall_score(history_df['ground_truth'], history_df['autoenc_pred'], zero_division=0)
            
            st.metric("F1 Score", f"{autoenc_f1:.4f}", delta=f"{autoenc_f1*100:.2f}%")
            st.metric("Accuracy", f"{autoenc_acc:.4f}", delta=f"{autoenc_acc*100:.2f}%")
            st.metric("Precision", f"{autoenc_prec:.4f}", delta=f"{autoenc_prec*100:.2f}%")
            st.metric("Recall", f"{autoenc_rec:.4f}", delta=f"{autoenc_rec*100:.2f}%")
        
        # F1 Score Comparison
        st.markdown("### F1 Score Comparison")
        f1_data = {
            'Detector': ['Z-Score', 'Isolation Forest', 'Autoencoder'],
            'F1 Score': [zscore_f1, iforest_f1, autoenc_f1],
            'Accuracy': [zscore_acc, iforest_acc, autoenc_acc],
            'Precision': [zscore_prec, iforest_prec, autoenc_prec],
            'Recall': [zscore_rec, iforest_rec, autoenc_rec]
        }
        
        fig_f1 = go.Figure(data=[
            go.Bar(name='F1 Score', x=f1_data['Detector'], y=f1_data['F1 Score'], marker_color='#00D9FF'),
            go.Bar(name='Accuracy', x=f1_data['Detector'], y=f1_data['Accuracy'], marker_color='#00FF88'),
            go.Bar(name='Precision', x=f1_data['Detector'], y=f1_data['Precision'], marker_color='#FFA500'),
            go.Bar(name='Recall', x=f1_data['Detector'], y=f1_data['Recall'], marker_color='#FF4444')
        ])
        
        fig_f1.update_layout(
            barmode='group',
            title='Performance Metrics Comparison',
            xaxis_title='Detector',
            yaxis_title='Score',
            template='plotly_dark',
            plot_bgcolor='#0F1419',
            paper_bgcolor='#0F1419',
            font=dict(color='#E0E0E0'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_f1, use_container_width=True)
    
    with tab2:
        st.markdown("### Confusion Matrices")
        
        col1, col2, col3 = st.columns(3)
        
        # Z-Score Confusion Matrix
        with col1:
            st.markdown("#### Z-Score")
            cm_zscore = confusion_matrix(history_df['ground_truth'], history_df['zscore_pred'])
            
            fig_cm_z = go.Figure(data=go.Heatmap(
                z=cm_zscore,
                x=['Predicted Normal', 'Predicted Anomaly'],
                y=['True Normal', 'True Anomaly'],
                colorscale='Viridis',
                text=cm_zscore,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorbar=dict(title="Count")
            ))
            fig_cm_z.update_layout(
                title='Z-Score Confusion Matrix',
                template='plotly_dark',
                plot_bgcolor='#0F1419',
                paper_bgcolor='#0F1419',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_cm_z, use_container_width=True)
        
        # Isolation Forest Confusion Matrix
        with col2:
            st.markdown("#### Isolation Forest")
            cm_iforest = confusion_matrix(history_df['ground_truth'], history_df['iforest_pred'])
            
            fig_cm_if = go.Figure(data=go.Heatmap(
                z=cm_iforest,
                x=['Predicted Normal', 'Predicted Anomaly'],
                y=['True Normal', 'True Anomaly'],
                colorscale='Viridis',
                text=cm_iforest,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorbar=dict(title="Count")
            ))
            fig_cm_if.update_layout(
                title='Isolation Forest Confusion Matrix',
                template='plotly_dark',
                plot_bgcolor='#0F1419',
                paper_bgcolor='#0F1419',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_cm_if, use_container_width=True)
        
        # Autoencoder Confusion Matrix
        with col3:
            st.markdown("#### Autoencoder")
            cm_autoenc = confusion_matrix(history_df['ground_truth'], history_df['autoenc_pred'])
            
            fig_cm_ae = go.Figure(data=go.Heatmap(
                z=cm_autoenc,
                x=['Predicted Normal', 'Predicted Anomaly'],
                y=['True Normal', 'True Anomaly'],
                colorscale='Viridis',
                text=cm_autoenc,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorbar=dict(title="Count")
            ))
            fig_cm_ae.update_layout(
                title='Autoencoder Confusion Matrix',
                template='plotly_dark',
                plot_bgcolor='#0F1419',
                paper_bgcolor='#0F1419',
                font=dict(color='#E0E0E0')
            )
            st.plotly_chart(fig_cm_ae, use_container_width=True)
    
    with tab3:
        st.markdown("### Score Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Normal vs Anomaly Scores")
            
            normal_mask = history_df['ground_truth'] == False
            anomaly_mask = history_df['ground_truth'] == True
            
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Box(
                y=history_df.loc[normal_mask, 'zscore_score'],
                name='Z-Score (Normal)',
                marker_color='#00D9FF'
            ))
            fig_dist.add_trace(go.Box(
                y=history_df.loc[anomaly_mask, 'zscore_score'],
                name='Z-Score (Anomaly)',
                marker_color='#FF4444'
            ))
            
            fig_dist.update_layout(
                title='Z-Score Distribution by Ground Truth',
                yaxis_title='Z-Score',
                template='plotly_dark',
                plot_bgcolor='#0F1419',
                paper_bgcolor='#0F1419',
                font=dict(color='#E0E0E0'),
                hovermode='y unified'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.markdown("#### Autoencoder Reconstruction Error")
            
            fig_re = go.Figure()
            
            fig_re.add_trace(go.Box(
                y=history_df.loc[normal_mask, 'autoenc_score'],
                name='RE (Normal)',
                marker_color='#00D9FF'
            ))
            fig_re.add_trace(go.Box(
                y=history_df.loc[anomaly_mask, 'autoenc_score'],
                name='RE (Anomaly)',
                marker_color='#FF4444'
            ))
            
            fig_re.update_layout(
                title='Reconstruction Error Distribution by Ground Truth',
                yaxis_title='Reconstruction Error',
                template='plotly_dark',
                plot_bgcolor='#0F1419',
                paper_bgcolor='#0F1419',
                font=dict(color='#E0E0E0'),
                hovermode='y unified'
            )
            st.plotly_chart(fig_re, use_container_width=True)
    
    with tab4:
        st.markdown("### Detector Comparison Over Time")
        
        # F1 Score Rolling Average
        window = 50
        history_df['zscore_f1_rolling'] = history_df['zscore_pred'].rolling(window).apply(
            lambda x: f1_score([True]*len(x), x, zero_division=0) if len(x) > 0 else 0
        )
        history_df['iforest_f1_rolling'] = history_df['iforest_pred'].rolling(window).apply(
            lambda x: f1_score([True]*len(x), x, zero_division=0) if len(x) > 0 else 0
        )
        history_df['autoenc_f1_rolling'] = history_df['autoenc_pred'].rolling(window).apply(
            lambda x: f1_score([True]*len(x), x, zero_division=0) if len(x) > 0 else 0
        )
        
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['zscore_score'],
            name='Z-Score',
            line=dict(color='#00D9FF', width=2),
            mode='lines'
        ))
        fig_timeline.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['iforest_score'],
            name='Isolation Forest',
            line=dict(color='#00FF88', width=2),
            mode='lines'
        ))
        fig_timeline.add_trace(go.Scatter(
            x=history_df['iteration'],
            y=history_df['autoenc_score'],
            name='Autoencoder',
            line=dict(color='#FFA500', width=2),
            mode='lines'
        ))
        
        fig_timeline.update_layout(
            title='Anomaly Scores Over Time',
            xaxis_title='Iteration',
            yaxis_title='Score',
            template='plotly_dark',
            plot_bgcolor='#0F1419',
            paper_bgcolor='#0F1419',
            font=dict(color='#E0E0E0'),
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Summary Statistics
        st.markdown("### Summary Statistics")
        
        summary_data = {
            'Metric': ['Total Samples', 'Detected Anomalies', 'True Positives (Z-Score)', 'True Positives (IF)', 'True Positives (AE)'],
            'Value': [
                len(history_df),
                history_df['ground_truth'].sum(),
                ((history_df['zscore_pred'] == True) & (history_df['ground_truth'] == True)).sum(),
                ((history_df['iforest_pred'] == True) & (history_df['ground_truth'] == True)).sum(),
                ((history_df['autoenc_pred'] == True) & (history_df['ground_truth'] == True)).sum()
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    st.warning("No data available. Click 'Run Full Simulation' in the sidebar to generate analysis data.")
