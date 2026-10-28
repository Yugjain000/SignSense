"""
Real-Time Dynamic ISL Word-Level Recognition System
Optimized for Instant Low-Latency Live CPU Tracking (PyTorch + MediaPipe Holistic Lite)
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

# Load PyTorch Models
print("=" * 65)
print("  Springer ISL Real-Time Recognition Engine (High-Speed CPU)")
print("=" * 65)

model_17 = CNN_LSTM_17()
model_17.load_state_dict(torch.load(os.path.join(BASE_DIR, "cnn_lstm_17.pt"), map_location='cpu'))
model_17.eval()
print("[+] Loaded CNN+LSTM (17 Classes, 99.88% Accuracy)")

model_61 = StackLSTM_61()
model_61.load_state_dict(torch.load(os.path.join(BASE_DIR, "stack_lstm_61.pt"), map_location='cpu'))
model_61.eval()
print("[+] Loaded StackLSTM (61 Classes, 98.5% Accuracy)")

active_model = model_17
active_classes = CLASSES_17
active_model_name = "CNN+LSTM (17 Words - 99.88% Accuracy)"

# MediaPipe Setup (Lite complexity for max FPS on CPU)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    model_complexity=0
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
    # Pose: Green
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 180, 0), thickness=2, circle_radius=1)
    )

FRAME_LENGTH = 50
SLIDE = 3 # Fast response: predicts every 3 frames (~10 times/second)!
CONF_THRESHOLD = 0.70

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
        print("[!] Error: Could not access webcam. Please verify camera permissions.")
        return

    # Use standard 640x480 for 30 FPS smooth tracking
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence = []
    detected_word = "Ready - Show hand gesture"
    confidence_score = 0.0

    fps_time = time.time()
    fps = 0.0

    print("\n[+] Camera stream started! Controls:")
    print("    - Press '1': Switch to CNN+LSTM (17 Core Signs)")
    print("    - Press '2': Switch to StackLSTM (61 Total Signs)")
    print("    - Press 'q': Exit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - fps_time)))
        fps_time = curr_time

        # Check for pitch black camera (closed privacy shutter)
        mean_brightness = float(np.mean(frame))
        is_dark = mean_brightness < 8.0

        # MediaPipe Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        draw_styled_landmarks(frame, results)
        keypoints = process_keypoints(results)

        # Instant warm-up: pre-fill buffer so user NEVER waits for 50 frames!
        if len(sequence) == 0:
            sequence = [keypoints] * (FRAME_LENGTH - 1)

        sequence.append(keypoints)

        has_hands = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

        # Predict every SLIDE frames
        if len(sequence) >= FRAME_LENGTH:
            if has_hands and not is_dark:
                seq_tensor = torch.tensor(np.array(sequence[-FRAME_LENGTH:]), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    preds = active_model(seq_tensor).squeeze(0).numpy()
                
                max_prob = float(np.max(preds))
                best_idx = int(np.argmax(preds))

                if max_prob >= CONF_THRESHOLD:
                    detected_word = active_classes[best_idx]
                    confidence_score = max_prob
                else:
                    detected_word = "Recognizing movement..."
                    confidence_score = max_prob
            elif not has_hands:
                detected_word = "No hands in frame"
                confidence_score = 0.0

            # Slide window
            sequence = sequence[SLIDE:]

        # Header HUD
        cv2.rectangle(frame, (0, 0), (w, 75), (15, 15, 25), -1)
        cv2.putText(frame, active_model_name, (20, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 255), 1, cv2.LINE_AA)
        
        status_text = f"FPS: {fps:.0f} | Conf: {confidence_score*100:.1f}% | [1] 17 Words [2] 61 Words [q] Exit"
        cv2.putText(frame, status_text, (20, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        # Warning banner if camera shutter is physically closed
        if is_dark:
            cv2.rectangle(frame, (30, h // 2 - 40), (w - 30, h // 2 + 40), (0, 0, 180), -1)
            cv2.putText(frame, "WEBCAM IS DARK / SHUTTER CLOSED!", (45, h // 2 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Please slide open your laptop camera privacy shutter", (45, h // 2 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Bottom Detection Banner
        cv2.rectangle(frame, (0, h - 80), (w, h), (15, 15, 25), -1)
        banner_color = (0, 255, 128) if confidence_score >= CONF_THRESHOLD else (180, 180, 180)
        cv2.putText(frame, f"SIGN: {detected_word.upper()}", (25, h - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, banner_color, 2, cv2.LINE_AA)

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
