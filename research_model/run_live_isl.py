"""
SignSense Real-Time Dynamic ISL Word-Level Recognition System
- PyTorch 99.88% CNN+LSTM (17 Classes) & 98.5% StackLSTM (61 Classes)
- Dual-Mode: Continuous Live Streaming + Spacebar Precision Gesture Recording
- Upper Body Pose + Hands Skeleton Overlay with Distance/Framing Guidance
- Peak-Hold Latch (Holds confirmed gesture on-screen for 3.0s)
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

ACTION_HINTS = [
    ("FEVER", "Bring hand up to forehead"),
    ("HELLO", "Salute outward from temple"),
    ("GIVE", "Push open palms forward"),
    ("KEY", "Twist key in air twice"),
    ("HUG", "Cross both arms over chest"),
    ("FEDUP", "Hand sweeps under chin"),
    ("KNIFE", "Chop right fingers on left palm"),
    ("MAN", "Twirl moustache at upper lip"),
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
        fwd_last = out[:, -1, :64]
        bwd_last = out[:, 0, 64:]
        last_step = torch.cat([fwd_last, bwd_last], dim=-1)
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
        fwd_last = out[:, -1, :64]
        bwd_last = out[:, 0, 64:]
        last_step = torch.cat([fwd_last, bwd_last], dim=-1)
        logits = self.fc(last_step)
        return self.softmax(logits)

print("=" * 65)
print("  SignSense Real-Time Dynamic ISL Engine (PyTorch CPU)")
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
active_model_name = "CNN+LSTM 17 Classes (99.88%)"

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    model_complexity=0
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Upper body pose connections (mirrored)
POSE_ARM_PAIRS = [
    (11, 12), # Left shoulder to Right shoulder
    (11, 13), # Left shoulder to Left elbow
    (13, 15), # Left elbow to Left wrist
    (12, 14), # Right shoulder to Right elbow
    (14, 16), # Right elbow to Right wrist
]

def process_keypoints(results):
    pose = [[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] if results.pose_landmarks else [[0.0] * 3] * 33
    lh = [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [[0.0] * 3] * 21
    rh = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [[0.0] * 3] * 21
    landmarks = []
    landmarks.extend(pose)
    landmarks.extend(lh)
    landmarks.extend(rh)
    return np.array(landmarks, dtype=np.float32).flatten()

def draw_mirrored_hand(frame, landmarks, color, w, h):
    pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in landmarks.landmark]
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        cv2.line(frame, pts[p1_idx], pts[p2_idx], (40, 40, 40), 3, cv2.LINE_AA)
        cv2.line(frame, pts[p1_idx], pts[p2_idx], color, 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

def draw_mirrored_pose(frame, pose_landmarks, w, h):
    lms = pose_landmarks.landmark
    pts = {i: (int((1.0 - lms[i].x) * w), int(lms[i].y * h)) for i in range(len(lms))}
    for p1, p2 in POSE_ARM_PAIRS:
        if p1 in pts and p2 in pts:
            # Check visibility
            if lms[p1].visibility > 0.4 and lms[p2].visibility > 0.4:
                cv2.line(frame, pts[p1], pts[p2], (0, 255, 128), 2, cv2.LINE_AA)
                cv2.circle(frame, pts[p1], 4, (0, 200, 100), -1, cv2.LINE_AA)
                cv2.circle(frame, pts[p2], 4, (0, 200, 100), -1, cv2.LINE_AA)

FRAME_LENGTH = 50

def switch_model(mode):
    global active_model, active_classes, active_model_name
    if mode == 1:
        active_model = model_17
        active_classes = CLASSES_17
        active_model_name = "CNN+LSTM 17 Classes (99.88%)"
        print("\n[*] Switched to CNN+LSTM (17 Classes)")
    elif mode == 2:
        active_model = model_61
        active_classes = CLASSES_61
        active_model_name = "StackLSTM 61 Classes (98.5%)"
        print("\n[*] Switched to StackLSTM (61 Classes)")

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not access webcam. Please ensure camera privacy shutter is open and no other app is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sequence = []
    detected_word = "STANDBY"
    confidence_score = 0.0
    top_predictions = []
    motion_energy = 0.0
    prev_hand_pts = None

    # Latch for confirmed predictions
    confirmed_word = ""
    confirmed_conf = 0.0
    confirmed_timestamp = 0.0

    # Spacebar Gesture Recording State: 'IDLE', 'COUNTDOWN', 'RECORDING'
    rec_state = 'IDLE'
    rec_buffer = []
    countdown_start = 0.0

    fps_time = time.time()
    fps = 0.0

    CANVAS_W = 1020
    CANVAS_H = 520
    CAM_W = 640
    CAM_H = 480

    cv2.namedWindow("SignSense Dynamic ISL Pro", cv2.WINDOW_AUTOSIZE)

    print("\n[+] Engine running!")
    print("    - Tap [SPACE]: Precision 50-Frame Sign Recording (3.. 2.. 1.. GO!)")
    print("    - Press '1': 17 Words (CNN+LSTM 99.88%)")
    print("    - Press '2': 61 Words (StackLSTM 98.5%)")
    print("    - Press 'c': Clear buffer / Reset")
    print("    - Press 'q': Exit\n")

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break

        # MediaPipe native raw frame
        rgb_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        rgb_raw.flags.writeable = False
        results = holistic.process(rgb_raw)
        rgb_raw.flags.writeable = True

        display_frame = cv2.flip(raw_frame, 1)
        h, w, _ = display_frame.shape

        # Draw Pose (Shoulders & Arms)
        shoulders_visible = False
        if results.pose_landmarks:
            draw_mirrored_pose(display_frame, results.pose_landmarks, w, h)
            # Check shoulder visibility and framing
            l_sh = results.pose_landmarks.landmark[11]
            r_sh = results.pose_landmarks.landmark[12]
            if l_sh.visibility > 0.4 and r_sh.visibility > 0.4 and l_sh.y < 0.90 and r_sh.y < 0.90:
                shoulders_visible = True

        # Draw Hands
        curr_pts = []
        has_hands = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

        if results.left_hand_landmarks:
            draw_mirrored_hand(display_frame, results.left_hand_landmarks, (255, 180, 0), w, h)
            curr_pts.extend([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            draw_mirrored_hand(display_frame, results.right_hand_landmarks, (60, 80, 255), w, h)
            curr_pts.extend([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark])

        if len(curr_pts) > 0 and prev_hand_pts is not None and len(prev_hand_pts) == len(curr_pts):
            diff = np.mean(np.abs(np.array(curr_pts) - np.array(prev_hand_pts))) * 500
            motion_energy = 0.7 * motion_energy + 0.3 * diff
        else:
            motion_energy = 0.8 * motion_energy
        prev_hand_pts = curr_pts

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - fps_time)))
        fps_time = curr_time

        keypoints = process_keypoints(results)

        # -------------------------------------------------------------
        # Precision Recording via Spacebar
        # -------------------------------------------------------------
        if rec_state == 'COUNTDOWN':
            elapsed = curr_time - countdown_start
            rem = int(3.0 - elapsed) + 1
            if rem > 0:
                cv2.rectangle(display_frame, (w//2 - 140, h//2 - 60), (w//2 + 140, h//2 + 40), (20, 20, 30), -1)
                cv2.putText(display_frame, f"STARTING IN: {rem}", (w//2 - 120, h//2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 230, 255), 2, cv2.LINE_AA)
            else:
                rec_state = 'RECORDING'
                rec_buffer = []

        elif rec_state == 'RECORDING':
            rec_buffer.append(keypoints)
            progress = len(rec_buffer) / FRAME_LENGTH
            # Overlay recording banner
            cv2.rectangle(display_frame, (20, h - 60), (w - 20, h - 20), (30, 30, 45), -1)
            cv2.rectangle(display_frame, (20, h - 60), (int(20 + (w - 40) * progress), h - 20), (0, 140, 255), -1)
            cv2.putText(display_frame, f"RECORDING SIGN... ({len(rec_buffer)}/{FRAME_LENGTH})", (30, h - 33),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if len(rec_buffer) >= FRAME_LENGTH:
                # Run model on exact 50 recorded frames
                seq_tensor = torch.tensor(np.array(rec_buffer), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    preds = active_model(seq_tensor).squeeze(0).numpy()
                top_indices = np.argsort(preds)[::-1][:4]
                top_predictions = [(active_classes[i], float(preds[i])) for i in top_indices]
                
                best_word, best_conf = top_predictions[0]
                confirmed_word = best_word
                confirmed_conf = best_conf
                confirmed_timestamp = curr_time
                detected_word = best_word
                confidence_score = best_conf
                rec_state = 'IDLE'

        # -------------------------------------------------------------
        # Continuous Live Rolling Stream (when not in recording mode)
        # -------------------------------------------------------------
        if rec_state == 'IDLE':
            if len(sequence) == 0:
                for _ in range(FRAME_LENGTH):
                    sequence.append(keypoints)
            else:
                sequence.append(keypoints)
                if len(sequence) > FRAME_LENGTH:
                    sequence.pop(0)

            if has_hands:
                seq_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    preds = active_model(seq_tensor).squeeze(0).numpy()

                top_indices = np.argsort(preds)[::-1][:4]
                top_predictions = [(active_classes[i], float(preds[i])) for i in top_indices]

                top1_word, top1_conf = top_predictions[0]
                confidence_score = top1_conf

                # Peak-hold latch: If confidence hits >= 45%, latch it for 3 seconds
                if confidence_score >= 0.45:
                    detected_word = top1_word
                    confirmed_word = top1_word
                    confirmed_conf = top1_conf
                    confirmed_timestamp = curr_time
                elif curr_time - confirmed_timestamp < 3.0:
                    detected_word = confirmed_word
                    confidence_score = confirmed_conf
                elif motion_energy > 1.2:
                    detected_word = f"{top1_word}?"
                else:
                    detected_word = "PERFORM SIGN"
            else:
                if curr_time - confirmed_timestamp < 3.0:
                    detected_word = confirmed_word
                    confidence_score = confirmed_conf
                else:
                    detected_word = "SHOW HANDS"
                    confidence_score = 0.0

        # -------------------------------------------------------------
        # Canvas Construction (1020 x 520)
        # -------------------------------------------------------------
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 26)

        cam_resized = cv2.resize(display_frame, (CAM_W, CAM_H))
        canvas[20:20+CAM_H, 20:20+CAM_W] = cam_resized
        cv2.rectangle(canvas, (19, 19), (21+CAM_W, 21+CAM_H), (55, 60, 72), 1)

        # FPS badge
        cv2.rectangle(canvas, (30, 30), (160, 58), (15, 18, 22), -1)
        cv2.putText(canvas, f"FPS: {fps:.0f} | 640x480", (38, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (160, 220, 160), 1, cv2.LINE_AA)

        # Shoulder framing reminder if sitting too close
        if not shoulders_visible:
            cv2.rectangle(canvas, (175, 30), (450, 58), (30, 40, 60), -1)
            cv2.putText(canvas, "TIP: Sit back slightly to show shoulders", (182, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 210, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # Right Sidebar
        # -------------------------------------------------------------
        sb_x = 680

        cv2.putText(canvas, "SIGNSENSE PRO", (sb_x, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, active_model_name, (sb_x, 62),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (180, 190, 210), 1, cv2.LINE_AA)
        cv2.line(canvas, (sb_x, 72), (CANVAS_W - 25, 72), (45, 50, 60), 1)

        # Detection Box
        box_y = 85
        cv2.rectangle(canvas, (sb_x, box_y), (CANVAS_W - 25, box_y + 75), (28, 32, 40), -1)
        cv2.rectangle(canvas, (sb_x, box_y), (CANVAS_W - 25, box_y + 75), (60, 68, 82), 1)

        header_label = "CONFIRMED SIGN" if (curr_time - confirmed_timestamp < 3.0 and confirmed_conf >= 0.45) else "DETECTED GESTURE"
        cv2.putText(canvas, header_label, (sb_x + 12, box_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 230, 255) if header_label == "CONFIRMED SIGN" else (140, 150, 170), 1, cv2.LINE_AA)

        if confidence_score >= 0.70:
            word_color = (60, 240, 120)
        elif confidence_score >= 0.45:
            word_color = (0, 210, 255)
        else:
            word_color = (180, 190, 205)

        cv2.putText(canvas, detected_word.upper(), (sb_x + 12, box_y + 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, word_color, 2, cv2.LINE_AA)

        bar_y = box_y + 68
        bar_w_max = (CANVAS_W - 25) - (sb_x + 12) - 12
        conf_fill = int(bar_w_max * max(0.0, min(1.0, confidence_score)))
        cv2.rectangle(canvas, (sb_x + 12, bar_y), (sb_x + 12 + bar_w_max, bar_y + 3), (40, 45, 55), -1)
        if conf_fill > 0:
            cv2.rectangle(canvas, (sb_x + 12, bar_y), (sb_x + 12 + conf_fill, bar_y + 3), word_color, -1)

        # Top-3 Predictions Meter
        top_y = 180
        cv2.putText(canvas, "LIVE PROBABILITIES", (sb_x, top_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (150, 160, 180), 1, cv2.LINE_AA)

        meter_start_y = top_y + 14
        if len(top_predictions) > 0:
            for idx, (c_name, c_prob) in enumerate(top_predictions[:3]):
                row_y = meter_start_y + (idx * 22)
                cv2.putText(canvas, f"{c_name:<11}", (sb_x, row_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
                m_bar_x = sb_x + 95
                m_bar_w = 150
                m_fill = int(m_bar_w * max(0.0, min(1.0, c_prob)))
                cv2.rectangle(canvas, (m_bar_x, row_y + 3), (m_bar_x + m_bar_w, row_y + 12), (35, 40, 50), -1)
                cv2.rectangle(canvas, (m_bar_x, row_y + 3), (m_bar_x + m_fill, row_y + 12), (0, 200, 255) if idx == 0 else (100, 110, 130), -1)
                cv2.putText(canvas, f"{c_prob*100:.0f}%", (m_bar_x + m_bar_w + 10, row_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 180, 195), 1, cv2.LINE_AA)

        # Action Guide
        guide_y = 265
        cv2.line(canvas, (sb_x, guide_y), (CANVAS_W - 25, guide_y), (45, 50, 60), 1)
        cv2.putText(canvas, "TRY THESE EASY SIGNS:", (sb_x, guide_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        start_hint_y = guide_y + 38
        for i, (sign_label, sign_desc) in enumerate(ACTION_HINTS[:5]):
            h_y = start_hint_y + (i * 30)
            cv2.circle(canvas, (sb_x + 4, h_y + 4), 3, (80, 190, 255), -1)
            cv2.putText(canvas, f"{sign_label}:", (sb_x + 14, h_y + 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (240, 240, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, sign_desc, (sb_x + 85, h_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 160, 175), 1, cv2.LINE_AA)

        # Bottom Instructions & Shortcuts
        ctrl_y = CANVAS_H - 30
        cv2.line(canvas, (sb_x, ctrl_y - 12), (CANVAS_W - 25, ctrl_y - 12), (45, 50, 60), 1)
        cv2.putText(canvas, "[SPACE] Record Sign   [1] 17W   [2] 61W   [q] Quit", (sb_x, ctrl_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 130, 145), 1, cv2.LINE_AA)

        cv2.imshow("SignSense Dynamic ISL Pro", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # Spacebar
            rec_state = 'COUNTDOWN'
            countdown_start = time.time()
        elif key == ord('1'):
            switch_model(1)
        elif key == ord('2'):
            switch_model(2)
        elif key == ord('c'):
            sequence = []
            rec_state = 'IDLE'
            confirmed_word = ""
            confirmed_conf = 0.0
            detected_word = "BUFFER RESET"
            confidence_score = 0.0

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("[+] Engine closed cleanly.")

if __name__ == "__main__":
    main()
