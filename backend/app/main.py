from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import google.generativeai as genai
import os

# Automatically load .env if present
for env_path in [".env", "../.env", "backend/.env"]:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")
        break

# Configure Gemini strictly via environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel('gemini-1.5-flash')
else:
    llm = None

# Load the trained Machine Learning model
try:
    with open('sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('actions.pkl', 'rb') as f:
        actions = pickle.load(f)
except Exception as e:
    try:
        with open('../sign_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../actions.pkl', 'rb') as f:
            actions = pickle.load(f)
    except:
        model = None
        actions = []
        print("Warning: sign_model.pkl or actions.pkl not found!")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ISL Communication Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LandmarkData(BaseModel):
    keypoints: List[float] # Expecting 3780 floats

class SignSequence(BaseModel):
    words: List[str]

@app.get("/")
def read_root():
    return {"status": "Agentic AI Backend is fully operational!"}

# Predict sign (Legacy endpoint removed to prevent 3780-feature mismatch crashes)

import base64
import cv2

import collections

class FrameData(BaseModel):
    image_base64: str

# Setup MediaPipe for live backend processing
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)
except:
    detector = None

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

    return np.concatenate([lh.flatten(), rh.flatten()])

# Sequence management for live camera
prediction_buffer = collections.deque(maxlen=5)
last_confirmed_word = ""

@app.post("/process_frame")
def process_frame(data: FrameData):
    global prediction_buffer, last_confirmed_word
    if detector is None or model is None:
        return {"predicted_word": None}

    try:
        img_data = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Resize to match training data aspect ratio (4:3)
        frame = cv2.resize(frame, (640, 480))
        
        # Mirror the frame to match the training data
        frame = cv2.flip(frame, 1)
        
        # Convert for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = detector.detect(mp_image)
        keypoints = extract_keypoints(detection_result)
        
        # If no hands are detected, ignore this frame
        if np.sum(keypoints) == 0:
            last_confirmed_word = "" # Reset when hands drop
            return {"predicted_word": None, "has_hands": False, "landmarks": []}
        
        # Extract 2D normalized landmark points for video skeleton overlay
        raw_landmarks = []
        if detection_result.hand_landmarks:
            for hand_lms in detection_result.hand_landmarks:
                raw_landmarks.append([[round(lm.x, 3), round(lm.y, 3)] for lm in hand_lms])

        # Predict on this single frame (126 features)
        input_data = keypoints.reshape(1, -1)
        prediction_index = model.predict(input_data)[0]
        current_prediction = actions[prediction_index]
        
        prediction_buffer.append(current_prediction)
        
        # Debounce: If we have 5 frames and the most common prediction appears at least 3 times
        confirmed_word = None
        if len(prediction_buffer) == 5:
            most_common = collections.Counter(prediction_buffer).most_common(1)[0]
            if most_common[1] >= 3:
                confirmed = most_common[0]
                prediction_buffer.clear()
                if confirmed != last_confirmed_word:
                    last_confirmed_word = confirmed
                    confirmed_word = confirmed
            
        return {
            "predicted_word": confirmed_word,
            "raw_prediction": current_prediction,
            "has_hands": True,
            "landmarks": raw_landmarks
        }
    except Exception as e:
        return {"predicted_word": None, "has_hands": False, "landmarks": [], "error": str(e)}

class TeachData(BaseModel):
    word: str
    frames_base64: List[str] # Expecting 30 frames

@app.post("/teach_ai")
def teach_ai(data: TeachData):
    global model, actions
    
    if detector is None:
        raise HTTPException(status_code=500, detail="MediaPipe detector not initialized.")
    
    # Process the 30 frames
    new_features = []
    for img_base64 in data.frames_base64:
        try:
            img_data = base64.b64decode(img_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)
            keypoints = extract_keypoints(detection_result)
            
            if np.sum(keypoints) != 0:
                new_features.append(keypoints)
        except:
            continue
            
    if len(new_features) < 3:
        raise HTTPException(status_code=400, detail="Not enough valid frames detected. Ensure hands are clearly visible.")
        
    word = data.word.strip().upper()
    if word not in actions:
        actions.append(word)
    label = actions.index(word)
    
    # Load memory bank
    try:
        X_memory = np.load('X_memory.npy')
        y_memory = np.load('y_memory.npy')
    except:
        X_memory = np.zeros((0, 126))
        y_memory = np.zeros((0,))
        
    # Append new features
    new_X = np.array(new_features)
    new_y = np.array([label] * len(new_features))
    
    X_memory = np.concatenate([X_memory, new_X])
    y_memory = np.concatenate([y_memory, new_y])
    
    # Cap frames per class to 15 to prevent Random Forest Class Flooding/Imbalance
    balanced_X = []
    balanced_y = []
    for cls in np.unique(y_memory):
        cls_indices = np.where(y_memory == cls)[0]
        # Keep only the latest 15 frames for this class (FIFO)
        if len(cls_indices) > 15:
            cls_indices = cls_indices[-15:]
        balanced_X.append(X_memory[cls_indices])
        balanced_y.append(y_memory[cls_indices])
        
    X_memory = np.vstack(balanced_X)
    y_memory = np.concatenate(balanced_y)
    
    # Save memory bank
    np.save('X_memory.npy', X_memory)
    np.save('y_memory.npy', y_memory)
    
    # Save actions so the word label is persistent
    with open('actions.pkl', 'wb') as f:
        import pickle
        pickle.dump(actions, f)
        
    # Prevent Scikit-Learn 500 Crash (Random Forest requires >= 2 classes to split)
    if len(np.unique(y_memory)) < 2:
        return {"message": f"Captured '{word}'! Please teach at least 1 more sign to unlock AI."}
        
    # Backup existing model before overwriting
    import shutil
    try:
        shutil.copy('sign_model.pkl', 'sign_model_backup.pkl')
    except Exception:
        pass
    
    # Retrain model
    from sklearn.ensemble import RandomForestClassifier
    print(f"Retraining model with {len(X_memory)} perfectly balanced frames...")
    new_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    new_model.fit(X_memory, y_memory)
    
    # Save new model
    import pickle
    with open('sign_model.pkl', 'wb') as f:
        pickle.dump(new_model, f)
        
    # Update live variables
    model = new_model
    
    return {"message": f"Successfully learned '{word}'!"}
    
@app.post("/generate_sentence")
def generate_sentence(data: SignSequence):
    if not data.words:
        raise HTTPException(status_code=400, detail="No words provided.")
    
    # Agentic Prompting
    prompt = f"""
    You are an AI assistant designed to translate rough sign language glosses into natural spoken English sentences.
    The user performed the following sequence of signs: {data.words}
    
    Convert this into a grammatically correct, natural-sounding sentence. 
    Maintain the intended meaning exactly. Do not add extra information.
    Just output the final sentence and nothing else.
    
    Example: ['i', 'want', 'water'] -> I want water.
    Example: ['hello', 'thanks'] -> Hello, thank you.
    """
    
    try:
        if llm is None:
            raise ValueError("Gemini LLM is unconfigured.")
        response = llm.generate_content(prompt)
        sentence = response.text.strip()
        return {"original_signs": data.words, "natural_sentence": sentence}
    except Exception as e:
        # Graceful fallback: capitalize raw tokens into coherent default
        fallback = " ".join(data.words).capitalize() + "."
        return {
            "original_signs": data.words,
            "natural_sentence": fallback,
            "fallback_used": True,
            "error_detail": str(e)
        }
