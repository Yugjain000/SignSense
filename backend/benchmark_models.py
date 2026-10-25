import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def run_multi_model_benchmark():
    print("=" * 70)
    print(" SIGNSENSE: MULTI-MODEL INFORMED BENCHMARK (RF vs MLP vs kNN)")
    print("=" * 70)

    # 1. Load Data
    X = np.load('X_memory.npy')
    y = np.load('y_memory.npy').astype(int)
    with open('actions.pkl', 'rb') as f:
        actions = pickle.load(f)

    classes, counts = np.unique(y, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]

    # Benchmark on Core 10 Active ISL Vocabulary
    top10_classes = classes[sorted_idx[:10]]
    top10_names = [actions[c] for c in top10_classes]

    mask = np.isin(y, top10_classes)
    X_core = X[mask]
    y_core = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=0.25, random_state=42, stratify=y_core
    )

    models = {
        "Random Forest (100 Trees)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=42),
        "k-Nearest Neighbors (kNN, k=5)": KNeighborsClassifier(n_neighbors=5, weights='distance')
    }

    results = []

    for name, clf in models.items():
        print(f"\n[*] Evaluating: {name}...")
        t_train_start = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = (time.perf_counter() - t_train_start) * 1000.0 # ms

        # Latency Benchmark over 200 trials
        latency_trials = []
        for i in range(min(200, len(X_test))):
            sample = X_test[i:i+1]
            t0 = time.perf_counter()
            _ = clf.predict(sample)
            latency_trials.append((time.perf_counter() - t0) * 1000.0) # ms

        avg_latency = np.mean(latency_trials)
        p95_latency = np.percentile(latency_trials, 95)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100.0
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "name": name,
            "accuracy": acc,
            "precision": p_w * 100.0,
            "recall": r_w * 100.0,
            "f1": f1_w * 100.0,
            "train_time_ms": train_time,
            "latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "clf": clf,
            "y_pred": y_pred
        })

    # Print Comparison Table
    print("\n" + "=" * 90)
    print(f"{'Model Architecture':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Latency':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<30} | {r['accuracy']:>8.2f}% | {r['precision']:>8.2f}% | {r['recall']:>8.2f}% | {r['f1']:>8.2f}% | {r['latency_ms']:>6.2f} ms")
    print("=" * 90)

    # Detailed Per-Class Precision & Recall for the Best Model (Random Forest)
    best_res = results[0]
    print(f"\n[+] Detailed Per-Class Breakdown for Best Model ({best_res['name']}):")
    print(classification_report(y_test, best_res['y_pred'], target_names=top10_names, zero_division=0))

    # Generate Model Comparison Bar Chart
    os.makedirs('evaluation_artifacts', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.22
    indices = np.arange(len(results))

    accs = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    latencies = [r['latency_ms'] for r in results]

    b1 = ax.bar(indices - bar_width, accs, bar_width, label='Accuracy (%)', color='#4A62FF', edgecolor='black')
    b2 = ax.bar(indices, f1s, bar_width, label='F1-Score (%)', color='#2ECC71', edgecolor='black')
    b3 = ax.bar(indices + bar_width, latencies, bar_width, label='Latency (ms)', color='#E74C3C', edgecolor='black')

    ax.set_ylabel('Score / Milliseconds', fontsize=11, fontweight='bold')
    ax.set_title('Informed Model Comparison: RF vs MLP vs kNN', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels([r['name'] for r in results], fontsize=10, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha='center', fontsize=9, fontweight='bold')
    for bar in b3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.2f}ms", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_out = os.path.join('evaluation_artifacts', 'model_comparison_benchmark.png')
    plt.savefig(chart_out, dpi=300)
    plt.close()
    print(f"\n[+] Saved comparison chart: {chart_out}")

if __name__ == '__main__':
    run_multi_model_benchmark()
