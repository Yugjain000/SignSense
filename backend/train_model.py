import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

dataset_path = 'dataset/landmark_files_all'

X = []
y = []
actions = []

print("Loading dataset...")
if not os.path.exists(dataset_path):
    print("Dataset path not found!")
    exit(1)

words = os.listdir(dataset_path)
for word in words:
    word_path = os.path.join(dataset_path, word)
    if not os.path.isdir(word_path):
        continue
    
    actions.append(word)
    label = actions.index(word)
    
    # Process parquet files for this word
    for parquet_file in os.listdir(word_path):
        if not parquet_file.endswith('.parquet'):
            continue
            
        file_path = os.path.join(word_path, parquet_file)
        
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
            
        # Group by frame to process frame-by-frame
        grouped = df.groupby('frame')
        
        for frame, frame_data in grouped:
            if frame % 5 != 0:
                continue
            
            # Filter for left and right hand
            lh_data = frame_data[frame_data['type'] == 'left_hand'].sort_values('landmark_index')
            rh_data = frame_data[frame_data['type'] == 'right_hand'].sort_values('landmark_index')
            
            # If both hands are completely NaN (not present), skip this frame
            if lh_data['x'].isna().all() and rh_data['x'].isna().all():
                continue
                
            lh_keypoints = np.zeros((21, 3))
            rh_keypoints = np.zeros((21, 3))
            
            has_lh = not lh_data['x'].isna().all() and len(lh_data) == 21
            has_rh = not rh_data['x'].isna().all() and len(rh_data) == 21
            
            if has_lh:
                lh_coords = lh_data[['x', 'y', 'z']].values
                lh_keypoints = lh_coords
                
            if has_rh:
                rh_coords = rh_data[['x', 'y', 'z']].values
                rh_keypoints = rh_coords
                
            if has_lh or has_rh:
                anchor = lh_keypoints[0] if has_lh else rh_keypoints[0]
                if has_lh: lh_keypoints = lh_keypoints - anchor
                if has_rh: rh_keypoints = rh_keypoints - anchor
                
                all_kps = []
                if has_lh: all_kps.extend(lh_keypoints)
                if has_rh: all_kps.extend(rh_keypoints)
                
                max_val = np.max(np.abs(all_kps))
                if max_val > 0:
                    if has_lh: lh_keypoints = lh_keypoints / max_val
                    if has_rh: rh_keypoints = rh_keypoints / max_val
                    
                lh_keypoints = np.nan_to_num(lh_keypoints)
                rh_keypoints = np.nan_to_num(rh_keypoints)
                
            frame_features = np.concatenate([lh_keypoints.flatten(), rh_keypoints.flatten()])
            
            X.append(frame_features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Data shape: {X.shape}, labels shape: {y.shape}")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=1)
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

print("Saving model and actions...")
with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('actions.pkl', 'wb') as f:
    pickle.dump(actions, f)
    
print("Done!")
