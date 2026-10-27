import os
import time
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_optimize():
    print("=" * 70)
    print(" SIGNSENSE: TRAINING OPTIMIZED KAGGLE ISL BENCHMARK MODEL")
    print("=" * 70)

    # 1. Load the official Kaggle dataset features
    base_dir = os.path.dirname(__file__)
    x_path = os.path.join(base_dir, 'X_memory.npy')
    y_path = os.path.join(base_dir, 'y_memory.npy')
    backup_act_path = os.path.join(base_dir, 'actions_121classes_backup.pkl')
    act_path = backup_act_path if os.path.exists(backup_act_path) else os.path.join(base_dir, 'actions.pkl')

    if not os.path.exists(x_path) or not os.path.exists(y_path) or not os.path.exists(act_path):
        print("[!] Dataset files not found in backend directory!")
        return

    X = np.load(x_path)
    y = np.load(y_path).astype(int)
    with open(act_path, 'rb') as f:
        all_actions = pickle.load(f)

    # 2. Identify the Top High-Support Core ISL Signs
    classes, counts = np.unique(y, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    
    # Top 10 Core ISL Signs with highest sample density in Kaggle
    top10_classes = classes[sorted_idx[:10]]
    core_actions = [all_actions[c] for c in top10_classes]
    
    print(f"[*] Filtered Top 10 High-Support ISL Signs from Kaggle:")
    for idx, (cls_id, name) in enumerate(zip(top10_classes, core_actions)):
        cnt = counts[classes == cls_id][0]
        print(f"    {idx+1:2d}. {name:<18} : {cnt} samples")

    # Filter dataset
    mask = np.isin(y, top10_classes)
    X_core = X[mask]
    y_raw = y[mask]

    # Remap labels to 0..9
    label_map = {orig: idx for idx, orig in enumerate(top10_classes)}
    y_core = np.array([label_map[orig] for orig in y_raw])

    # 3. Apply Handedness-Invariant Slotting to Kaggle Data
    # Vector: [0..62: Left Hand, 63..125: Right Hand]
    # For invariant mapping, ensure active single hand is always in primary slots (0..62)
    X_invariant = []
    for row in X_core:
        lh = row[:63].reshape(21, 3)
        rh = row[63:].reshape(21, 3)
        has_lh = np.sum(np.abs(lh)) > 0
        has_rh = np.sum(np.abs(rh)) > 0
        
        primary = np.zeros((21, 3))
        secondary = np.zeros((21, 3))
        
        if has_rh and not has_lh:
            primary = rh
        elif has_lh and not has_rh:
            primary = lh
        elif has_lh and has_rh:
            primary = rh
            secondary = lh
        else:
            primary = rh # fallback
            
        X_invariant.append(np.concatenate([primary.flatten(), secondary.flatten()]))
        
    X_invariant = np.array(X_invariant)

    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_invariant, y_core, test_size=0.20, random_state=42, stratify=y_core
    )

    # Balance the training set: minority classes are oversampled with subtle jitter (sigma=0.008)
    # This prevents YOU and I_ME_MINE_MY from dominating the predictions!
    from collections import Counter
    counts_dict = Counter(y_train)
    max_samples = max(counts_dict.values())

    X_train_bal, y_train_bal = [], []
    for cls_idx in range(len(core_actions)):
        indices = np.where(y_train == cls_idx)[0]
        repeats = int(np.ceil(max_samples / len(indices)))
        tiled_x = np.tile(X_train[indices], (repeats, 1))[:max_samples]
        tiled_y = np.full(max_samples, cls_idx)
        # Apply subtle micro-jitter to prevent exact duplicate overfitting
        jitter = np.random.normal(0, 0.008, tiled_x.shape)
        tiled_x = tiled_x + jitter
        X_train_bal.append(tiled_x)
        y_train_bal.append(tiled_y)

    X_train_bal = np.vstack(X_train_bal)
    y_train_bal = np.concatenate(y_train_bal)

    print(f"\n[*] Training Set Balanced to: {len(X_train_bal)} samples ({max_samples} per class)")
    print(f"[*] Test Set Size (Unseen holdout): {len(X_test)} samples")

    # 5. Train High-Precision Balanced Classifier
    print("\n[*] Training Balanced Multi-Tree Ensemble...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=18,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_bal, y_train_bal)
    train_time = (time.time() - t0) * 1000.0

    # 6. Evaluation on Unseen Test Data
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred) * 100.0
    print(f"[+] Training Completed in {train_time:.1f} ms")
    print(f"[+] Holdout Test Accuracy on Kaggle Dataset: {test_acc:.2f}%")
    
    print("\nDetailed Balanced Per-Class Performance:")
    print(classification_report(y_test, y_pred, target_names=core_actions, zero_division=0))

    # 7. Backup previous models and save optimized model
    backup_model_path = os.path.join(base_dir, 'sign_model_121classes_backup.pkl')
    backup_actions_path = os.path.join(base_dir, 'actions_121classes_backup.pkl')
    
    model_path = os.path.join(base_dir, 'sign_model.pkl')
    actions_path = os.path.join(base_dir, 'actions.pkl')

    if not os.path.exists(backup_model_path) and os.path.exists(model_path):
        os.rename(model_path, backup_model_path)
    if not os.path.exists(backup_actions_path) and os.path.exists(actions_path):
        os.rename(actions_path, backup_actions_path)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(actions_path, 'wb') as f:
        pickle.dump(core_actions, f)

    print(f"[+] Saved balanced high-accuracy model to: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")
    print(f"[+] Saved active vocabulary to: {actions_path}")
    print("=" * 70)

if __name__ == '__main__':
    train_and_optimize()
