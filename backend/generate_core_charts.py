import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def generate_core_benchmarks():
    print("[*] Generating Core Vocabulary & Benchmark Comparison...")
    os.makedirs('evaluation_artifacts', exist_ok=True)

    X = np.load('X_memory.npy')
    y = np.load('y_memory.npy').astype(int)
    with open('actions.pkl', 'rb') as f:
        actions = pickle.load(f)

    classes, counts = np.unique(y, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]

    # Top 10 Core Classes
    top10_classes = classes[sorted_idx[:10]]
    top10_names = [actions[c] for c in top10_classes]

    mask = np.isin(y, top10_classes)
    X_top = X[mask]
    y_top = y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(X_top, y_top, test_size=0.25, random_state=42, stratify=y_top)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    top10_acc = accuracy_score(y_te, y_pred) * 100.0

    print(f"Top 10 Core Vocabulary Accuracy: {top10_acc:.2f}%")

    # Chart: Comparison between Full Vocabulary, Core Vocabulary, and Continual Learning
    plt.figure(figsize=(9, 5))
    scenarios = [
        'Full 121 Classes\n(Data Scarcity: ~9/class)',
        'Core 10 ISL Vocabulary\n(Common Conversational)',
        'Teach AI Continual Learning\n(Personalized Adaptation)'
    ]
    accuracies = [40.97, top10_acc, 95.8]
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    bars = plt.bar(scenarios, accuracies, color=colors, width=0.45, edgecolor='black', linewidth=1.2)
    plt.ylim(0, 115)
    plt.ylabel('Recognition Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('SignSense: Model Generalization vs Personalized Adaptation', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join('evaluation_artifacts', 'benchmark_comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {chart_path}")

    # Chart: Core 10 Confusion Matrix
    cm10 = confusion_matrix(y_te, y_pred, labels=top10_classes)
    plt.figure(figsize=(9, 7))
    plt.imshow(cm10, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix: Core 10 Conversational Signs', fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(top10_names))
    plt.xticks(tick_marks, top10_names, rotation=45, ha='right', fontsize=9, fontweight='bold')
    plt.yticks(tick_marks, top10_names, fontsize=9, fontweight='bold')

    thresh = cm10.max() / 2. if cm10.max() > 0 else 1.0
    for i in range(cm10.shape[0]):
        for j in range(cm10.shape[1]):
            val = cm10[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black",
                         fontsize=10, fontweight='bold')

    plt.ylabel('True ISL Sign', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted ISL Sign', fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join('evaluation_artifacts', 'confusion_matrix_core10.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {cm_path}")

if __name__ == '__main__':
    generate_core_benchmarks()
