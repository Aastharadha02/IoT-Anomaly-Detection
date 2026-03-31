import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from data import generate_row, stream_data, clean_data, SENSOR_COLS
from models import AnomalyEngine
import warnings
warnings.filterwarnings('ignore')

# Set dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

def generate_sensor_data(n_samples=100, contamination=0.05):
    """Generate sensor data with anomalies"""
    data_list = []
    labels = []
    
    for i in range(n_samples):
        row = generate_row()
        
        # Inject anomalies with specified contamination
        if np.random.rand() < contamination:
            # Add significant anomaly
            row_values = row[SENSOR_COLS].values[0].copy()
            idx = np.random.randint(0, len(SENSOR_COLS))
            row_values[idx] = np.random.uniform(10, 20)
            for col, val in zip(SENSOR_COLS, row_values):
                row[col] = val
            labels.append(1)
        else:
            labels.append(0)
        
        # Clean data to remove NaN values
        row = clean_data(row)
        data_list.append(row)
    
    data = pd.concat(data_list, ignore_index=True)
    # Final cleaning to ensure no NaN values
    data[SENSOR_COLS] = data[SENSOR_COLS].fillna(data[SENSOR_COLS].mean())
    return data, np.array(labels)

def run_evaluation(n_baseline=2000, n_test=2000, contamination=0.08):
    """Run improved evaluation with proper train/test split and precision tuning"""
    
    print(f"\n{'='*60}")
    print("PRECISION-OPTIMIZED ANOMALY DETECTION EVALUATION")
    print(f"{'='*60}")
    print("Improvements: Higher Z-score threshold (3.2), Stricter contamination (2%),")
    print("Higher autoencoder percentile (98%), Stronger ensemble voting (2.5)")
    
    # Step 1: Generate baseline (normal) data
    print("\n1. Generating baseline training data (normal operations)...")
    baseline_data, _ = generate_sensor_data(n_samples=n_baseline, contamination=0.0)
    baseline_X = baseline_data[SENSOR_COLS].values
    print(f"   Generated {len(baseline_X)} baseline samples")
    
    # Step 2: Generate test data with anomalies
    print("\n2. Generating test data with anomalies...")
    test_data, test_y = generate_sensor_data(n_samples=n_test, contamination=contamination)
    test_X = test_data[SENSOR_COLS].values
    
    n_anomalies = np.sum(test_y)
    print(f"   Generated {len(test_X)} test samples with {n_anomalies} anomalies ({n_anomalies/len(test_X)*100:.1f}%)")
    
    # Step 3: Train anomaly detection engine
    print("\n3. Training anomaly detection models...")
    engine = AnomalyEngine(contamination=contamination)
    engine.train(baseline_X)
    print("   Models trained successfully!")
    
    # Step 4: Run detection on test data
    print("\n4. Running anomaly detection on test data...")
    predictions_list = []
    
    for i, sensor_values in enumerate(test_X):
        preds = engine.detect(sensor_values)
        predictions_list.append(preds)
        if (i + 1) % (n_test // 4) == 0:
            print(f"   Progress: {i+1}/{len(test_X)} samples processed")
    
    predictions_df = pd.DataFrame(predictions_list)
    print("   Detection completed!")
    
    # Step 5: Calculate metrics for each model
    print("\n5. Calculating performance metrics...\n")
    
    models = ['z_anom', 'if_anom', 'ae_anom', 'ee_anom', 'ensemble']
    model_names = ['Z-Score', 'Isolation Forest', 'Autoencoder', 'Elliptic Envelope', 'Ensemble']
    
    metrics_data = []
    confusion_matrices = {}
    
    for model_col, model_name in zip(models, model_names):
        y_pred = predictions_df[model_col].values
        
        f1 = f1_score(test_y, y_pred, zero_division=0)
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, zero_division=0)
        recall = recall_score(test_y, y_pred, zero_division=0)
        
        cm = confusion_matrix(test_y, y_pred, labels=[0, 1])
        confusion_matrices[model_name] = cm
        
        metrics_data.append({
            'Model': model_name,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'TN': cm[0, 0],
            'FP': cm[0, 1],
            'FN': cm[1, 0],
            'TP': cm[1, 1]
        })
        
        print(f"{model_name:20s} | F1: {f1:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Step 6: Visualizations
    print("\n6. Generating visualizations...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0F1419')
    
    # 1. F1 Score Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(metrics_df['Model'], metrics_df['F1 Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#00D9FF'])
    ax1.set_title('F1 Score Comparison', fontsize=12, fontweight='bold', color='#00D9FF')
    ax1.set_ylabel('F1 Score', color='#E0E0E0')
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='#00D9FF')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Bar Chart
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(metrics_df['Model'], metrics_df['Accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#00D9FF'])
    ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold', color='#00D9FF')
    ax2.set_ylabel('Accuracy', color='#E0E0E0')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='#00D9FF')
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics Radar comparison for best model
    ax3 = plt.subplot(2, 3, 3)
    best_idx = metrics_df['F1 Score'].idxmax()
    best_model = metrics_df.iloc[best_idx]
    ax3.bar(['F1', 'Accuracy', 'Precision', 'Recall'], 
           [best_model['F1 Score'], best_model['Accuracy'], best_model['Precision'], best_model['Recall']],
           color='#00D9FF')
    ax3.set_title(f"Best Model: {best_model['Model']}", fontsize=12, fontweight='bold', color='#00D9FF')
    ax3.set_ylabel('Score', color='#E0E0E0')
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    
    # 4-8. Confusion Matrices for each model (first 3 only in main chart)
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        if idx < 3:
            ax = plt.subplot(2, 3, 4 + idx)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            ax.set_title(f"{model_name} Confusion Matrix", fontsize=11, fontweight='bold', color='#00D9FF')
            ax.set_ylabel('True Label', color='#E0E0E0')
            ax.set_xlabel('Predicted Label', color='#E0E0E0')
    
    # Replace the last position with an overall metrics table
    ax_table = plt.subplot(2, 3, 6)
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = metrics_df[['Model', 'F1 Score', 'Accuracy', 'Precision', 'Recall']].copy()
    for col in ['F1 Score', 'Accuracy', 'Precision', 'Recall']:
        table_data[col] = table_data[col].apply(lambda x: f'{x:.4f}')
    
    table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns,
                          cellLoc='center', loc='center',
                          colColours=['#1A1F28']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    plt.suptitle('Improved Anomaly Detection - Performance Analysis', 
                fontsize=14, fontweight='bold', color='#00D9FF', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('improved_model_evaluation.png', dpi=150, facecolor='#0F1419', edgecolor='none')
    print("   Saved: improved_model_evaluation.png")
    
    # Save results to CSV
    metrics_df.to_csv('improved_model_metrics.csv', index=False)
    print("   Saved: improved_model_metrics.csv")
    
    # Step 7: Precision-Recall Analysis
    print("\n7. Analyzing Precision-Recall improvement...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0F1419')
    
    for idx, (model_name, col) in enumerate(zip(model_names[:4], models[:4])):
        ax = axes[idx // 2, idx % 2]
        
        y_pred = predictions_df[col].values
        cm = confusion_matrix(test_y, y_pred, labels=[0, 1])
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision = precision_score(test_y, y_pred, zero_division=0)
        recall = recall_score(test_y, y_pred, zero_division=0)
        
        # Create confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                   cbar=True, square=True, annot_kws={'size': 12})
        
        ax.set_title(f'{model_name}\nPrecision: {precision:.3f} | Recall: {recall:.3f}', 
                    fontsize=11, fontweight='bold', color='#00D9FF')
        ax.set_ylabel('True Label', color='#E0E0E0')
        ax.set_xlabel('Predicted Label', color='#E0E0E0')
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
    
    plt.suptitle('Confusion Matrices - Precision Optimized Models', 
                fontsize=14, fontweight='bold', color='#00D9FF', y=0.995)
    plt.tight_layout()
    plt.savefig('confusion_matrices_detailed.png', dpi=150, facecolor='#0F1419', edgecolor='none')
    print("   Saved: confusion_matrices_detailed.png")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return metrics_df, predictions_df, test_y
    
    # Detailed confusion matrix visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0F1419')
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar=True,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold', color='#00D9FF')
        ax.set_ylabel('True Label', color='#E0E0E0')
        ax.set_xlabel('Predicted Label', color='#E0E0E0')
    
    # Remove extra subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', color='#00D9FF')
    plt.tight_layout()
    plt.savefig('confusion_matrices_detailed.png', dpi=150, facecolor='#0F1419', edgecolor='none')
    print("   Saved: confusion_matrices_detailed.png")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60 + "\n")
    
    return metrics_df, predictions_df, test_y

if __name__ == "__main__":
    metrics_df, predictions_df, test_y = run_evaluation(
        n_baseline=2000,
        n_test=1500,
        contamination=0.12
    )
    
    print("\nFinal Results Summary:")
    print(metrics_df.to_string(index=False))
