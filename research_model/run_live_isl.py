"""
Real-Time Dynamic ISL Word-Level Recognition System
Powered by PyTorch & MediaPipe Holistic
Based on Springer Nature Research Paper:
'An Efficient Real-Time Word-Level Recognition of Indian Sign Language' (2024)
Authors: Chaitanya Madhav R, Prasad S Shet, Mohan Murali K S, Likhith R (PES University)
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES_17 = [
    'Fedup', 'Fever', 'Giraffe', 'Give', 'Hello', 'Hug',
    'Injury', 'Interview', 'Jump', 'Key', 'Knife', 'Lemon',
    'Lion', 'Man', 'Maths', 'Maybe', 'Monkey'
]

CLASSES_61 = [
    'Bear', 'Break', 'Brinjal', 'Budget', 'Busy', 'Cabbage',
    'Carrot', 'Cauliflower', 'Chilli', 'Clean', 'Close', 'Come',
    'Cook', 'Crocodile', 'Cry', 'Cucumber', 'Deer', 'Drink',
    'Elephant', 'Exam', 'Fedup', 'Fever', 'Giraffe', 'Give',
    'Good afternoon', 'Good Morning', 'Hello', 'Hug', 'Injury',
    'Interview', 'Jump', 'Karnataka', 'Key', 'Knife', 'Lemon',
    'Lion', 'Man', 'Maths', 'Maybe', 'Monkey', 'Onion',
    'Peacock', 'Pigeon', 'Pour', 'Radish', 'Sparrow', 'Still',
    'Switch', 'Tea', 'Temple', 'Thank you', 'Tiger', 'Turtle',
    'Umbrella', 'Uncle', 'Vegetables', 'Volcano', 'What is your Name',
    'Wife', 'Writer', 'Wrong'
]

# ------------------------------------------------------------------
# Model Architectures
# ------------------------------------------------------------------
class CNN_LSTM_17(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(225, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 17)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        out, _ = self.lstm3(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return self.softmax(logits)

class StackLSTM_61(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(225, 64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 61)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        out, _ = self.lstm5(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return self.softmax(logits)

# Load Models into Memory
print("=" * 65)
print("  Springer ISL Real-Time Recognition Engine (PyTorch CPU)")
print("=" * 65)

model_17 = CNN_LSTM_17()
model_17.load_state_dict(torch.load(os.path.join(BASE_DIR, "cnn_lstm_17.pt"), map_location='cpu'))
model_17.eval()
print("[+] Loaded CNN+LSTM (17 Classes, 99.88% Accuracy)")

model_61 = StackLSTM_61()
model_61.load_state_dict(torch.load(os.path.join(BASE_DIR, "stack_lstm_61.pt"), map_location='cpu'))
model_61.eval()
print("[+] Loaded StackLSTM (61 Classes, 98.5% Accuracy)")

# Active Model Selector
active_model = model_17
active_classes = CLASSES_17
active_model_name = "CNN+LSTM (17 Words - 99.88% Accuracy)"

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_keypoints(results):
    pose = [[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] if results.pose_landmarks else [[0.0] * 3] * 33
    lh = [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [[0.0] * 3] * 21
    rh = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [[0.0] * 3] * 21
    landmarks = []
    landmarks.extend(pose)
    landmarks.extend(lh)
    landmarks.extend(rh)
    return np.array(landmarks, dtype=np.float32).flatten()

def draw_styled_landmarks(image, results):
    # Right hand: Coral Red
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(60, 60, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1)
    )
    # Left hand: Cyan Blue
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 180, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1)
    )
    # Upper Body: Green
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 180, 0), thickness=2, circle_radius=1)
    )

FRAME_LENGTH = 50
SLIDE = 12
CONF_THRESHOLD = 0.85

def switch_model(mode):
    global active_model, active_classes, active_model_name
    if mode == 1:
        active_model = model_17
        active_classes = CLASSES_17
        active_model_name = "CNN+LSTM (17 Words - 99.88% Accuracy)"
        print("\n[*] Switched to CNN+LSTM (17 Classes)")
    elif mode == 2:
        active_model = model_61
        active_classes = CLASSES_61
        active_model_name = "StackLSTM (61 ISLRTC Words - 98.5% Accuracy)"
        print("\n[*] Switched to StackLSTM (61 Classes)")

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not access webcam. Please make sure no other program is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence = []
    detected_word = "Ready - Perform a sign"
    confidence_score = 0.0

    print("\n[+] Camera stream started! Controls:")
    print("    - Press '1': Switch to CNN+LSTM (17 Core Signs)")
    print("    - Press '2': Switch to StackLSTM (61 Total Signs)")
    print("    - Press 'q': Exit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[-] Camera frame dropped.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # MediaPipe Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        draw_styled_landmarks(frame, results)
        keypoints = process_keypoints(results)
        sequence.append(keypoints)

        # Predict when buffer reaches 50 frames
        if len(sequence) == FRAME_LENGTH:
            seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                preds = active_model(seq_tensor).squeeze(0).numpy()
            
            max_prob = float(np.max(preds))
            best_idx = int(np.argmax(preds))

            if max_prob >= CONF_THRESHOLD:
                detected_word = active_classes[best_idx]
                confidence_score = max_prob
            else:
                detected_word = "Tracking motion..."
                confidence_score = max_prob

            sequence = sequence[SLIDE:]

        # Header HUD
        cv2.rectangle(frame, (0, 0), (w, 85), (15, 15, 25), -1)
        cv2.putText(frame, active_model_name, (25, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (220, 220, 255), 1, cv2.LINE_AA)
        
        status_text = f"Buffer: {len(sequence)}/{FRAME_LENGTH} | Confidence: {confidence_score*100:.1f}% | Keys: [1] 17 Words [2] 61 Words [q] Exit"
        cv2.putText(frame, status_text, (25, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        # Bottom Detection Banner
        cv2.rectangle(frame, (0, h - 90), (w, h), (15, 15, 25), -1)
        banner_color = (0, 255, 128) if confidence_score >= CONF_THRESHOLD else (180, 180, 180)
        cv2.putText(frame, f"SIGN: {detected_word.upper()}", (30, h - 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, banner_color, 2, cv2.LINE_AA)

        cv2.imshow("SignSense Dynamic ISL - Springer Benchmark Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            switch_model(1)
        elif key == ord('2'):
            switch_model(2)

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("[+] Engine closed cleanly.")

if __name__ == "__main__":
    main()
