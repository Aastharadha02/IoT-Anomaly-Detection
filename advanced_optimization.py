import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from data import generate_row, clean_data, SENSOR_COLS
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
            row_values = row[SENSOR_COLS].values[0].copy()
            idx = np.random.randint(0, len(SENSOR_COLS))
            row_values[idx] = np.random.uniform(10, 20)
            for col, val in zip(SENSOR_COLS, row_values):
                row[col] = val
            labels.append(1)
        else:
            labels.append(0)
        
        row = clean_data(row)
        data_list.append(row)
    
    data = pd.concat(data_list, ignore_index=True)
    data[SENSOR_COLS] = data[SENSOR_COLS].fillna(data[SENSOR_COLS].mean())
    return data, np.array(labels)

def get_confidence_scores(engine, test_X):
    """Get confidence scores from each detector (0-1 scale)"""
    confidences = []
    predictions = []
    
    for sensor_values in test_X:
        # Get raw predictions and detect anomalies
        preds = engine.detect(sensor_values)
        predictions.append(preds)
        
        # Calculate confidence as combination of detector agreement
        votes = (preds['z_anom'] * 1.0 + 
                preds['if_anom'] * 1.2 + 
                preds['ae_anom'] * 1.1 + 
                preds['ee_anom'] * 1.1)
        confidence = votes / 4.4  # Normalize to 0-1
        confidences.append(confidence)
    
    return np.array(confidences), predictions

def run_precision_optimization(n_baseline=2000, n_test=1500, contamination=0.08):
    """Run advanced precision optimization with confidence thresholding"""
    
    print(f"\n{'='*70}")
    print("ADVANCED PRECISION OPTIMIZATION USING CONFIDENCE THRESHOLDING")
    print(f"{'='*70}")
    
    # Step 1: Generate and train
    print("\n1. Generating and training models...")
    baseline_data, _ = generate_sensor_data(n_samples=n_baseline, contamination=0.0)
    baseline_X = baseline_data[SENSOR_COLS].values
    
    test_data, test_y = generate_sensor_data(n_samples=n_test, contamination=contamination)
    test_X = test_data[SENSOR_COLS].values
    
    engine = AnomalyEngine(contamination=contamination)
    engine.train(baseline_X)
    print(f"   Training complete. Test set: {len(test_X)} samples, {np.sum(test_y)} anomalies")
    
    # Step 2: Get confidence scores
    print("\n2. Computing confidence scores for each prediction...")
    confidences, predictions = get_confidence_scores(engine, test_X)
    
    predictions_df = pd.DataFrame(predictions)
    
    # Step 3: Test different decision thresholds
    print("\n3. Testing different confidence thresholds for precision optimization...\n")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Accuracy':<12}")
    print("-" * 60)
    
    for threshold in thresholds:
        # Apply confidence threshold
        y_pred_thresholded = (confidences >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(test_y, y_pred_thresholded, zero_division=0)
        recall = recall_score(test_y, y_pred_thresholded, zero_division=0)
        f1 = f1_score(test_y, y_pred_thresholded, zero_division=0)
        accuracy = accuracy_score(test_y, y_pred_thresholded)
        
        results.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy
        })
        
        print(f"{threshold:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:<12.4f}")
    
    results_df = pd.DataFrame(results)
    best_f1_idx = results_df['F1 Score'].idxmax()
    best_threshold = thresholds[best_f1_idx]
    
    print(f"\nOptimal Threshold: {best_threshold} (F1: {results_df.loc[best_f1_idx, 'F1 Score']:.4f})")
    
    # Step 4: Visualizations
    print("\n4. Generating precision-recall analyses...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0F1419')
    
    # Plot 1: Precision-Recall Trade-off
    ax1 = axes[0, 0]
    ax1.plot(results_df['Threshold'], results_df['Precision'], marker='o', label='Precision', color='#FF6B6B', linewidth=2)
    ax1.plot(results_df['Threshold'], results_df['Recall'], marker='s', label='Recall', color='#4ECDC4', linewidth=2)
    ax1.plot(results_df['Threshold'], results_df['F1 Score'], marker='^', label='F1 Score', color='#00D9FF', linewidth=2)
    ax1.axvline(best_threshold, color='#FFD700', linestyle='--', alpha=0.7, label=f'Optimal ({best_threshold})')
    ax1.set_xlabel('Confidence Threshold', color='#E0E0E0')
    ax1.set_ylabel('Score', color='#E0E0E0')
    ax1.set_title('Precision-Recall Trade-off Analysis', fontsize=12, fontweight='bold', color='#00D9FF')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score vs Accuracy
    ax2 = axes[0, 1]
    ax2.scatter(results_df['Accuracy'], results_df['F1 Score'], s=300, c=results_df['Threshold'], 
               cmap='plasma', alpha=0.7, edgecolors='#00D9FF', linewidth=2)
    for idx, row in results_df.iterrows():
        ax2.annotate(f"{row['Threshold']:.1f}", (row['Accuracy'], row['F1 Score']), 
                    ha='center', va='center', fontsize=9, color='#E0E0E0')
    ax2.set_xlabel('Accuracy', color='#E0E0E0')
    ax2.set_ylabel('F1 Score', color='#E0E0E0')
    ax2.set_title('F1 Score vs Accuracy at Different Thresholds', fontsize=12, fontweight='bold', color='#00D9FF')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix at Optimal Threshold
    ax3 = axes[1, 0]
    y_pred_optimal = (confidences >= best_threshold).astype(int)
    cm = confusion_matrix(test_y, y_pred_optimal, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, cbar=True, square=True,
               xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    ax3.set_title(f'Confusion Matrix (Threshold={best_threshold})', fontsize=12, fontweight='bold', color='#00D9FF')
    ax3.set_ylabel('True Label', color='#E0E0E0')
    ax3.set_xlabel('Predicted Label', color='#E0E0E0')
    
    # Plot 4: Confidence Score Distribution
    ax4 = axes[1, 1]
    ax4.hist(confidences[test_y == 0], bins=30, alpha=0.6, label='Normal', color='#4ECDC4', edgecolor='#00D9FF')
    ax4.hist(confidences[test_y == 1], bins=30, alpha=0.6, label='Anomaly', color='#FF6B6B', edgecolor='#FFD700')
    ax4.axvline(best_threshold, color='#FFD700', linestyle='--', linewidth=2, label=f'Optimal Threshold ({best_threshold})')
    ax4.set_xlabel('Confidence Score', color='#E0E0E0')
    ax4.set_ylabel('Count', color='#E0E0E0')
    ax4.set_title('Confidence Score Distribution by Class', fontsize=12, fontweight='bold', color='#00D9FF')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Advanced Precision Optimization - Confidence Thresholding', 
                fontsize=14, fontweight='bold', color='#00D9FF', y=0.995)
    plt.tight_layout()
    plt.savefig('precision_optimization_analysis.png', dpi=150, facecolor='#0F1419', edgecolor='none')
    print("   Saved: precision_optimization_analysis.png")
    
    # Save results
    results_df.to_csv('precision_optimization_results.csv', index=False)
    print("   Saved: precision_optimization_results.csv")
    
    print("\n" + "="*70)
    print("PRECISION OPTIMIZATION COMPLETE")
    print("="*70)
    
    return results_df, best_threshold

if __name__ == "__main__":
    results_df, best_threshold = run_precision_optimization(
        n_baseline=2000,
        n_test=1500,
        contamination=0.08
    )
    print("\nKey Findings:")
    print(f"- Best threshold for F1 Score: {best_threshold}")
    print(f"- At this threshold, you can achieve the best balance between precision & recall")
    print(f"- Increasing threshold further improves precision but reduces recall")
