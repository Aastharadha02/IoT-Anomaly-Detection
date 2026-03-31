import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data import SENSOR_COLS, N_SENSORS
import warnings
warnings.filterwarnings('ignore')

# Set dark theme for plots
plt.style.use('dark_background')
sns.set_palette("husl")

print("=" * 80)
print("IoT ANOMALY DETECTION - MODEL EVALUATION")
print("=" * 80)

# Parameters - smaller dataset for faster computation
num_samples = 500
anomaly_rate = 0.05

print("\nGenerating test data...")
print(f"  - Samples: {num_samples}")
print(f"  - Anomaly Rate: {anomaly_rate*100}%")

# Generate synthetic data
np.random.seed(42)
data_list = []
true_labels = []

for i in range(num_samples):
    # Normal data
    values = np.random.normal(0, 1, N_SENSORS)
    
    # Add anomaly with specified probability
    is_anomaly = np.random.rand() < anomaly_rate
    if is_anomaly:
        idx = np.random.randint(0, N_SENSORS)
        values[idx] += np.random.uniform(5, 10)
    
    data_list.append(values)
    true_labels.append(1 if is_anomaly else 0)

test_data = np.array(data_list)
true_labels = np.array(true_labels)

print(f"\nGenerated data: {test_data.shape}")
print(f"Anomalies: {sum(true_labels)} ({sum(true_labels)/len(true_labels)*100:.2f}%)")
print(f"Normal: {len(true_labels) - sum(true_labels)} ({(1 - sum(true_labels)/len(true_labels))*100:.2f}%)")

# Pre-train models once on first 100 samples
print("\nPre-training anomaly detection models...")
train_data = test_data[:100]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
iso_forest = IsolationForest(contamination=anomaly_rate, n_estimators=50, random_state=0)
iso_forest.fit(X_train_scaled)

# Calculate statistics for Z-Score
mean_vals = train_data.mean(axis=0)
std_vals = train_data.std(axis=0)
std_vals[std_vals == 0] = 1e-9

# Run predictions using simple fast methods
print("Running anomaly detection on all samples...")
predictions = {
    'z_score': [],
    'isolation_forest': [],
    'autoencoder': [],
    'ensemble': []
}

for i in range(num_samples):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i+1}/{num_samples}")
    
    sample = test_data[i]
    
    # Z-Score method (threshold = 3)
    z_scores = np.abs((sample - mean_vals) / std_vals)
    z_anom = int(np.max(z_scores) > 3.0)
    predictions['z_score'].append(z_anom)
    
    # Isolation Forest
    sample_scaled = scaler.transform([sample])
    if_pred = iso_forest.predict(sample_scaled)[0]
    if_anom = int(if_pred == -1)
    predictions['isolation_forest'].append(if_anom)
    
    # Simple Autoencoder (reconstruction error based on mean)
    reconstruction_error = np.mean((sample - mean_vals) ** 2)
    ae_anom = int(reconstruction_error > 0.02)
    predictions['autoencoder'].append(ae_anom)
    
    # Ensemble: vote of at least 2 models
    ensemble = int((z_anom + if_anom + ae_anom) >= 2)
    predictions['ensemble'].append(ensemble)

# Convert to arrays
for key in predictions:
    predictions[key] = np.array(predictions[key])

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)

# Calculate metrics for each model
models = ['z_score', 'isolation_forest', 'autoencoder', 'ensemble']
model_display_names = ['Z-Score', 'Isolation Forest', 'Autoencoder', 'Ensemble']
metrics_results = []

for model, display_name in zip(models, model_display_names):
    y_pred = predictions[model]
    
    f1 = f1_score(true_labels, y_pred, zero_division=0)
    accuracy = accuracy_score(true_labels, y_pred)
    precision = precision_score(true_labels, y_pred, zero_division=0)
    recall = recall_score(true_labels, y_pred, zero_division=0)
    
    metrics_results.append({
        'Model': display_name,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })
    
    print(f"\n{display_name}:")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")

metrics_df = pd.DataFrame(metrics_results)

# Create visualizations
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0F1419')

# 1. Metrics comparison
ax = axes[0, 0]
x = np.arange(len(model_display_names))
width = 0.2

ax.bar(x - 1.5*width, metrics_df['F1 Score'], width, label='F1 Score', color='#00D9FF')
ax.bar(x - 0.5*width, metrics_df['Accuracy'], width, label='Accuracy', color='#FF6B6B')
ax.bar(x + 0.5*width, metrics_df['Precision'], width, label='Precision', color='#4ECDC4')
ax.bar(x + 1.5*width, metrics_df['Recall'], width, label='Recall', color='#95E1D3')

ax.set_ylabel('Score', fontsize=11, color='#E0E0E0')
ax.set_title('Model Performance Comparison', fontsize=12, color='#00D9FF', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_display_names, color='#E0E0E0')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#1A1F28')

# Add value labels on bars
for bars in [ax.patches[i:i+4] for i in range(0, len(ax.patches), 4)]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8, color='#E0E0E0')

# 2. Confusion Matrices
cm_positions = [(0, 1), (1, 0), (1, 1)]
for idx, (model, display_name) in enumerate(zip(models, model_display_names)):
    if idx >= 3:
        break
    
    y_pred = predictions[model]
    cm = confusion_matrix(true_labels, y_pred)
    
    ax = axes[cm_positions[idx]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Count'}, annot_kws={'size': 11})
    
    ax.set_title(f'{display_name}\nConfusion Matrix', fontsize=11, color='#00D9FF', fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, color='#E0E0E0')
    ax.set_xlabel('Predicted Label', fontsize=10, color='#E0E0E0')
    ax.set_xticklabels(['Normal', 'Anomaly'], color='#E0E0E0')
    ax.set_yticklabels(['Normal', 'Anomaly'], color='#E0E0E0')
    ax.set_facecolor('#1A1F28')

plt.tight_layout()
plt.savefig('model_evaluation_metrics.png', dpi=300, facecolor='#0F1419', edgecolor='none', bbox_inches='tight')
print("  Saved: model_evaluation_metrics.png")

# Create detailed confusion matrix for ensemble
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0F1419')

for idx, (model, display_name) in enumerate(zip(models, model_display_names)):
    ax = axes[idx // 2, idx % 2]
    y_pred = predictions[model]
    cm = confusion_matrix(true_labels, y_pred)
    
    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': 'Normalized Count'}, annot_kws={'size': 12},
                vmin=0, vmax=1)
    
    ax.set_title(f'{display_name}\nConfusion Matrix (Normalized)', fontsize=12, color='#00D9FF', fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, color='#E0E0E0')
    ax.set_xlabel('Predicted Label', fontsize=11, color='#E0E0E0')
    ax.set_xticklabels(['Normal', 'Anomaly'], color='#E0E0E0')
    ax.set_yticklabels(['Normal', 'Anomaly'], color='#E0E0E0')
    ax.set_facecolor('#1A1F28')

plt.tight_layout()
plt.savefig('confusion_matrices_detailed.png', dpi=300, facecolor='#0F1419', edgecolor='none', bbox_inches='tight')
print("  Saved: confusion_matrices_detailed.png")

# F1 Score comparison across models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0F1419')

# Bar chart
ax1.bar(model_display_names, metrics_df['F1 Score'], color=['#00D9FF', '#FF6B6B', '#4ECDC4', '#95E1D3'])
ax1.set_ylabel('F1 Score', fontsize=12, color='#E0E0E0')
ax1.set_title('F1 Score Comparison', fontsize=13, color='#00D9FF', fontweight='bold')
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)
ax1.set_facecolor('#1A1F28')

for i, v in enumerate(metrics_df['F1 Score']):
    ax1.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=11, color='#E0E0E0', fontweight='bold')

# Radar chart
from math import pi
categories = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax2 = plt.subplot(122, projection='polar', facecolor='#1A1F28')

colors = ['#00D9FF', '#FF6B6B', '#4ECDC4', '#95E1D3']
for idx, (model, display_name) in enumerate(zip(models, model_display_names)):
    values = [
        metrics_df.loc[idx, 'F1 Score'],
        metrics_df.loc[idx, 'Accuracy'],
        metrics_df.loc[idx, 'Precision'],
        metrics_df.loc[idx, 'Recall']
    ]
    values += values[:1]
    
    ax2.plot(angles, values, 'o-', linewidth=2, label=display_name, color=colors[idx])
    ax2.fill(angles, values, alpha=0.15, color=colors[idx])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, color='#E0E0E0')
ax2.set_ylim(0, 1)
ax2.set_title('Model Performance Radar', fontsize=13, color='#00D9FF', fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
ax2.grid(True, color='#333333', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('f1_score_analysis.png', dpi=300, facecolor='#0F1419', edgecolor='none', bbox_inches='tight')
print("  Saved: f1_score_analysis.png")

# Create detailed classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)

for model, display_name in zip(models, model_display_names):
    y_pred = predictions[model]
    print(f"\n{display_name}:")
    print(classification_report(true_labels, y_pred, target_names=['Normal', 'Anomaly'], digits=4))

# Save results to CSV
metrics_df.to_csv('model_metrics.csv', index=False)
print("\n  Saved: model_metrics.csv")

# Summary stats
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nBest F1 Score:", metrics_df.loc[metrics_df['F1 Score'].idxmax(), 'Model'])
print("Best Accuracy:", metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model'])
print("Best Precision:", metrics_df.loc[metrics_df['Precision'].idxmax(), 'Model'])
print("Best Recall:", metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model'])

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - model_evaluation_metrics.png")
print("  - confusion_matrices_detailed.png")
print("  - f1_score_analysis.png")
print("  - model_metrics.csv")
print("=" * 80)

plt.show()
