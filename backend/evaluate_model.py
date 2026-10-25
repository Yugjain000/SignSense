import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

def evaluate_system():
    print("=" * 60)
    print(" SIGNSENSE: ACADEMIC MODEL EVALUATION & BENCHMARK SUITE")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists('actions.pkl') or not os.path.exists('X_memory.npy') or not os.path.exists('y_memory.npy'):
        print("[ERROR] Required dataset files (actions.pkl, X_memory.npy, y_memory.npy) not found!")
        return

    with open('actions.pkl', 'rb') as f:
        actions = pickle.load(f)
    X = np.load('X_memory.npy')
    y = np.load('y_memory.npy').astype(int)

    num_samples = len(X)
    num_features = X.shape[1]
    unique_classes, counts = np.unique(y, return_counts=True)
    num_classes = len(unique_classes)

    print(f"\n[1] Dataset Statistics:")
    print(f"  - Total Keypoint Samples : {num_samples}")
    print(f"  - Feature Dimension      : {num_features} (126 coordinates: 2 hands x 21 x 3D)")
    print(f"  - Total Classes/Vocab    : {num_classes}")
    print(f"  - Min Samples per Class  : {counts.min()}")
    print(f"  - Max Samples per Class  : {counts.max()}")
    print(f"  - Mean Samples per Class : {counts.mean():.1f}")

    # 2. Benchmark Train/Test Split (80/20)
    print(f"\n[2] Performing 80/20 Train-Test Evaluation...")
    # Handle classes with only 1 sample gracefully for train_test_split
    valid_classes = unique_classes[counts >= 2]
    mask = np.isin(y, valid_classes)
    X_eval = X[mask]
    y_eval = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Benchmark Inference Latency
    num_latency_trials = 500
    sample_indices = np.random.choice(len(X_test), size=min(num_latency_trials, len(X_test)))
    latency_records = []
    for idx in sample_indices:
        sample = X_test[idx:idx+1]
        t0 = time.perf_counter()
        _ = clf.predict(sample)
        latency_records.append((time.perf_counter() - t0) * 1000.0) # in ms

    avg_latency = np.mean(latency_records)
    p95_latency = np.percentile(latency_records, 95)
    fps_estimate = 1000.0 / avg_latency

    # Predictions
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    train_acc = accuracy_score(y_train, y_train_pred) * 100.0
    test_acc = accuracy_score(y_test, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_test, y_pred) * 100.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n[3] Model Performance Metrics:")
    print(f"  - Training Accuracy      : {train_acc:.2f}%")
    print(f"  - Test Accuracy          : {test_acc:.2f}%")
    print(f"  - Balanced Accuracy      : {bal_acc:.2f}%")
    print(f"  - Precision (Macro)      : {p_macro * 100:.2f}%")
    print(f"  - Recall (Macro)         : {r_macro * 100:.2f}%")
    print(f"  - F1-Score (Macro)       : {f1_macro * 100:.2f}%")
    print(f"  - Precision (Weighted)   : {p_weight * 100:.2f}%")
    print(f"  - Recall (Weighted)      : {r_weight * 100:.2f}%")
    print(f"  - F1-Score (Weighted)    : {f1_weight * 100:.2f}%")

    print(f"\n[4] Computational & Latency Benchmark:")
    print(f"  - Training Time (100 trees): {train_time:.3f} s")
    print(f"  - Average Inference Latency: {avg_latency:.2f} ms / frame")
    print(f"  - 95th Percentile Latency  : {p95_latency:.2f} ms / frame")
    print(f"  - Real-Time Throughput     : ~{fps_estimate:.1f} FPS (Model alone)")

    # 3. Generate Visual Charts for Report & PPT
    os.makedirs('evaluation_artifacts', exist_ok=True)

    # Chart 1: Performance Metrics Summary Bar Chart
    plt.figure(figsize=(9, 5))
    metrics_names = ['Test Accuracy', 'Precision (W)', 'Recall (W)', 'F1-Score (W)', 'Balanced Acc']
    metrics_values = [test_acc, p_weight * 100, r_weight * 100, f1_weight * 100, bal_acc]
    colors = ['#4A62FF', '#2ECC71', '#3498DB', '#9B59B6', '#E67E22']

    bars = plt.bar(metrics_names, metrics_values, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 110)
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('SignSense: Model Performance Metrics Summary', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart1_path = os.path.join('evaluation_artifacts', 'metrics_summary.png')
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"\n[5] Saved visual chart: {chart1_path}")

    # Chart 2: Top Active Classes Confusion Matrix
    # Select top 15 most frequent classes in the test set to produce an aesthetic, readable confusion matrix
    test_class_counts = {c: np.sum(y_test == c) for c in np.unique(y_test)}
    top_classes = sorted(test_class_counts.keys(), key=lambda c: test_class_counts[c], reverse=True)[:15]

    top_mask = np.isin(y_test, top_classes) & np.isin(y_pred, top_classes)
    y_test_top = y_test[top_mask]
    y_pred_top = y_pred[top_mask]
    top_action_names = [actions[c] for c in top_classes]

    cm = confusion_matrix(y_test_top, y_pred_top, labels=top_classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: Representative ISL Classes', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(top_action_names))
    plt.xticks(tick_marks, top_action_names, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, top_action_names, fontsize=9)

    thresh = cm.max() / 2. if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=10, fontweight='bold')

    plt.ylabel('True ISL Sign Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted ISL Sign Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    chart2_path = os.path.join('evaluation_artifacts', 'confusion_matrix_top.png')
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"    Saved visual chart: {chart2_path}")

    # 4. Save Textual Benchmark Report
    report_text_path = os.path.join('evaluation_artifacts', 'evaluation_report.txt')
    with open(report_text_path, 'w', encoding='utf-8') as rf:
        rf.write("============================================================\n")
        rf.write("      SIGNSENSE: ACADEMIC MODEL EVALUATION REPORT\n")
        rf.write("============================================================\n\n")
        rf.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"Total Dataset Samples: {num_samples}\n")
        rf.write(f"Feature Dimension: {num_features} normalized keypoints\n")
        rf.write(f"Total Classes in Vocabulary: {num_classes}\n\n")
        rf.write("--- METRICS BREAKDOWN ---\n")
        rf.write(f"Training Accuracy   : {train_acc:.2f}%\n")
        rf.write(f"Testing Accuracy    : {test_acc:.2f}%\n")
        rf.write(f"Balanced Accuracy   : {bal_acc:.2f}%\n")
        rf.write(f"Precision (Weighted): {p_weight * 100:.2f}%\n")
        rf.write(f"Recall (Weighted)   : {r_weight * 100:.2f}%\n")
        rf.write(f"F1-Score (Weighted) : {f1_weight * 100:.2f}%\n\n")
        rf.write("--- LATENCY & THROUGHPUT ---\n")
        rf.write(f"Mean Inference Latency: {avg_latency:.2f} ms\n")
        rf.write(f"95th Percentile Latency: {p95_latency:.2f} ms\n")
        rf.write(f"Estimated Model Throughput: {fps_estimate:.1f} FPS\n\n")
        rf.write("--- CLASSIFICATION REPORT (Top Evaluated Classes) ---\n")
        rf.write(classification_report(y_test_top, y_pred_top, target_names=top_action_names, zero_division=0))

    print(f"    Saved report text : {report_text_path}")
    print("\n[SUCCESS] Evaluation complete! All metrics & publication-ready charts generated successfully.")

if __name__ == '__main__':
    evaluate_system()
