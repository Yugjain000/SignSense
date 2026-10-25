import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# --- Configuration ---
DATA_PATH = os.path.join('dataset')
actions = np.array(['hello', 'thanks', 'i', 'want', 'water'])
no_sequences = 30    
sequence_length = 30 

# Create folders for data
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# --- Setup MediaPipe Tasks API ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Middle
    (9,13), (13,14), (14,15), (15,16),# Ring
    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky and palm
]

def draw_landmarks(frame, hand_landmarks):
    h, w, c = frame.shape
    # Draw points
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
        x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def extract_keypoints(detection_result):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Get handedness (Left or Right)
            handedness = detection_result.handedness[idx][0].category_name
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            if handedness == 'Left':
                lh = keypoints
            else:
                rh = keypoints
    return np.concatenate([lh, rh])

# --- Main Data Collection Loop ---
cap = cv2.VideoCapture(0)

print("Starting data collection...")
for action in actions:
    print(f"\n--- Get ready to perform: {action.upper()} ---")
    cv2.waitKey(3000) 
    
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # MediaPipe tasks require their specific Image format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect landmarks
            detection_result = detector.detect(mp_image)
            
            # Draw landmarks
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)
            
            # User instructions on screen
            if frame_num == 0:
                cv2.putText(frame, 'STARTING COLLECTION', (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Sign: {action} | Video: {sequence}/{no_sequences}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(2000)
            else:
                cv2.putText(frame, f'Sign: {action} | Video: {sequence}/{no_sequences}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
            
            # Export keypoints
            keypoints = extract_keypoints(detection_result)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            np.save(npy_path, keypoints)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
                
cap.release()
cv2.destroyAllWindows()
print("Data collection completed successfully!")
