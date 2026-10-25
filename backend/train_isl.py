import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Initializing MediaPipe Hand Landmarker...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def extract_keypoints(detection_result):
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    has_lh = False
    has_rh = False
    
    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0].category_name
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            
            if handedness == 'Left':
                lh = keypoints
                has_lh = True
            else:
                rh = keypoints
                has_rh = True
                
    if has_lh or has_rh:
        anchor = lh[0] if has_lh else rh[0]
        if has_lh: lh = lh - anchor
        if has_rh: rh = rh - anchor
        
        all_kps = []
        if has_lh: all_kps.extend(lh)
        if has_rh: all_kps.extend(rh)
        
        max_val = np.max(np.abs(all_kps))
        if max_val > 0:
            if has_lh: lh = lh / max_val
            if has_rh: rh = rh / max_val

    return np.concatenate([lh.flatten(), rh.flatten()]), (has_lh or has_rh)


zip_path = os.getenv("ISL_DATASET_ZIP", os.path.join(os.path.dirname(__file__), "archive.zip"))
X = []
y = []
actions = []

print("Opening ISL Dataset ZIP file...")
with zipfile.ZipFile(zip_path, 'r') as z:
    image_files = [f for f in z.namelist() if "Frames_Word_Level/" in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} ISL images!")
    
    for i, file_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(image_files)}...")
            
        # Extract word label from folder name: "Frames_Word_Level/WORD/image.jpg"
        parts = file_path.split("Frames_Word_Level/")
        if len(parts) < 2: continue
        word = parts[1].split("/")[0].strip()
        
        if word not in actions:
            actions.append(word)
        label = actions.index(word)
        
        # Read image directly from zip into memory
        with z.open(file_path) as f:
            img_data = f.read()
            
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: continue
        
        # Mirror the frame if the dataset was facing the camera naturally (assume not mirrored, but backend flips it! Wait.)
        # The ISL dataset images are recorded by someone facing the camera. 
        # When a user faces the camera, their anatomical right hand is on the left of the image.
        # This matches live webcam (before flipping). 
        # In main.py, we do `cv2.flip(frame, 1)`. So we MUST flip the dataset images here too to match main.py!
        frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = detector.detect(mp_image)
        keypoints, valid = extract_keypoints(detection_result)
        
        if valid:
            X.append(keypoints)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Extracted {len(X)} valid hand frames across {len(actions)} ISL words.")

if len(X) == 0:
    print("No valid hands detected in dataset!")
    exit(1)

print("Saving ISL experience replay memory bank...")
np.save('X_memory.npy', X)
np.save('y_memory.npy', y)

print("Training ISL Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {acc * 100:.2f}%")

print("Saving final ISL model...")
with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('actions.pkl', 'wb') as f:
    pickle.dump(actions, f)

print("ISL Model Training Complete!")
