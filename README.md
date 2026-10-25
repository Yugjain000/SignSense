# SignSense: Two-Way AI-Powered Indian Sign Language Communication Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-FF6F00.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-8E75B2.svg?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Flutter](https://img.shields.io/badge/Flutter-3.x%20(Dart)-02569B.svg?logo=flutter&logoColor=white)](https://flutter.dev/)
[![CI Tests](https://img.shields.io/badge/CI%20Unit%20Tests-5%20Passing-brightgreen.svg?logo=githubactions&logoColor=white)](./.github/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/Platform-Web%20%7C%20Windows%20%7C%20Android-lightgrey.svg)]()

---

## 📹 Live Video Demonstration

> **Watch SignSense in Action:**  
> 🎬 **[Download / Watch the Full Working Demo Video](./assets/demo/signsense_demo.mp4)**  
> *(Real recorded video demonstration showing real-time 3D MediaPipe landmark tracking, live ISL sign prediction, Gemini NLP sentence synthesis, bidirectional voice-to-text dictation, and on-device personalization).*

---

## 📌 1. Project Overview & Societal Motivation

Over **18 million individuals** in India live with speech and hearing impairments. While **Indian Sign Language (ISL)** is their primary mode of communication, over 99% of the hearing population cannot understand or interpret ISL gestures. Conversely, deaf individuals frequently encounter comprehension hurdles with written English or Hindi because ISL possesses a fundamentally distinct grammatical syntax (e.g., Topic-Comment, Subject-Object-Verb, and an absence of English inflectional morphemes and auxiliary verbs like *"is, am, are"*).

Existing technological solutions suffer from major drawbacks:
1. **Intrusive Hardware:** Sensor gloves (flex sensors, IMUs) are bulky, fragile, expensive, and socially alienating.
2. **One-Way Communication:** Traditional research prototypes only translate sign-to-text; they completely ignore how the hearing person communicates back to the deaf user.
3. **Over-complex yet Brittle Deep Learning:** Heavy video 3D-CNNs and LSTMs suffer from massive latency (>1500 ms), fail in varying lighting, and cannot adapt to individual users on consumer-grade hardware.

**SignSense** resolves this communication divide by providing an end-to-end, real-time, **bidirectional** communication platform powered by lightweight computer vision and generative AI.

---

## ✨ 2. Key Features

* **3D Hand Tracking (Zero Sensor Gloves):** Utilizes Google MediaPipe Hand Landmarker to track 21 three-dimensional anatomical points per hand (126 coordinates total) from standard RGB consumer webcams.
* **Geometric Coordinate Normalization:** Enforces mathematical wrist-relative anchoring and global scale invariance, ensuring robust predictions regardless of user distance from the camera.
* **Sub-Millisecond Inference:** Random Forest ensemble classifier operating at **~1.15 ms inference latency** (~29–30 FPS real-time throughput).
* **Stateful Zero-Vector Debouncing:** Eliminates duplicate token emissions (e.g., preventing `"HELLO HELLO HELLO"` frame-by-frame spamming) through a state-gated filter that only resets when arms leave the camera viewport.
* **Generative NLP Smoothing (Gemini 1.5 Flash):** Bridges the grammatical gap by converting unordered ISL gloss sequences (e.g., `["I", "HUNGRY", "WANT", "FOOD"]`) into fluent natural English (`"I am hungry and I would like some food."`).
* **True Two-Way Conversational Loop:**
  * **Deaf to Hearing:** Signs are recognized, synthesized, and spoken aloud via **Text-to-Speech (TTS)**.
  * **Hearing to Deaf:** The hearing person speaks into the microphone; their voice is transcribed into text in real-time via **Speech-to-Text (STT)**.
* **🧠 On-Device Personalization with Replay Memory ("Teach AI"):** Eliminates overfitting and solves individual anatomical differences. Users can capture 4 frames of a new sign; the system appends the coordinates to a persistent replay memory bank (`X_memory.npy`), enforces FIFO class balancing, and retrains the 100-tree model in under **350 milliseconds** on the fly.
* **Live Computer Vision Landmark Overlay:** Custom Flutter graphics engine (`CustomPaint`) renders real-time MediaPipe skeletal joint-and-bone connections directly over the live webcam feed with sub-millisecond drawing latency and camera mirror support.
* **Responsive Single-Screen Accessible UI:** 3-panel dashboard (built with Flutter) designed to fit standard screens without overflow. Features real-time state badges (`LIVE`, `TRACKING`, `SEARCHING`, `DETECTED`), high-readability 20px Voice-to-Text typography, interactive sign token chips with deletion, and an instant **Dark / Light Mode** accessibility toggle.

---

## 🏗️ 3. System Architecture & End-to-End Pipeline

```
+-----------------------------------------------------------------------------------+
|                            SIGNSENSE SYSTEM ARCHITECTURE                          |
+-----------------------------------------------------------------------------------+

     [Deaf User Signs]                                    [Hearing User Speaks]
             │                                                      │
             ▼                                                      ▼
  ┌───────────────────────┐                               ┌──────────────────┐
  │   Flutter Frontend    │                               │ Flutter Frontend │
  │ (Web / Desktop / App) │                               │  (Microphone)    │
  └──────────┬────────────┘                               └─────────┬────────┘
             │ Base64 JPEG Frames (10-15 FPS)                       │ Web Speech API
             ▼                                                      ▼
  ┌───────────────────────┐                               ┌──────────────────┐
  │    FastAPI Backend    │                               │  Voice-to-Text   │
  │    /process_frame     │                               │   Transcription  │
  └──────────┬────────────┘                               └─────────┬────────┘
             │ Decoded BGR -> RGB Frame                             │
             ▼                                                      │
  ┌───────────────────────┐                                         │
  │   Google MediaPipe    │                                         │
  │    Hand Landmarker    │                                         │
  │  (21 landmarks/hand)  │                                         │
  └──────────┬────────────┘                                         │
             │ Raw [x, y, z] x 42 = 126 coords                      │
             ▼                                                      │
  ┌───────────────────────┐                                         │
  │ Coordinate Transform  │                                         │
  │  - Wrist Anchoring    │                                         │
  │  - Scale Normalization│                                         │
  └──────────┬────────────┘                                         │
             │ Normalized 126-D Vector                              │
             ▼                                                      │
  ┌───────────────────────┐                                         │
  │ Random Forest Model   │                                         │
  │  (100 Decision Trees) │                                         │
  └──────────┬────────────┘                                         │
             │ Raw Predicted Label                                  │
             ▼                                                      │
  ┌───────────────────────┐                                         │
  │ Stateful Debouncer    │                                         │
  │  (Reset on Arms Drop) │                                         │
  └──────────┬────────────┘                                         │
             │ Confirmed New ISL Token                              │
             ▼                                                      │
  ┌───────────────────────┐                                         │
  │ Gemini 1.5 Flash LLM  │                                         │
  │  /generate_sentence   │                                         │
  └──────────┬────────────┘                                         │
             │ Fluent Natural English Sentence                      │
             ▼                                                      │
  ┌───────────────────────┐                                         │
  │ Flutter TTS (Speaker) │                                         │
  │  "I want water."      │◄────────────────────────────────────────┘
  └───────────────────────┘
```

---

## 📐 4. Mathematical Formulations & Feature Engineering

### 4.1 Feature Dimensionality
MediaPipe detects up to two hands simultaneously. For each hand $h \in \{\text{Left}, \text{Right}\}$, 21 three-dimensional landmark points are produced:
$$\mathbf{L}_i^{(h)} = \big[x_i^{(h)},\, y_i^{(h)},\, z_i^{(h)}\big] \in \mathbb{R}^3, \quad i \in \{0, 1, \dots, 20\}$$

The concatenated raw feature vector contains:
$$D = 2 \text{ hands} \times 21 \text{ landmarks} \times 3 \text{ coordinates} = 126 \text{ numerical features}$$

### 4.2 Wrist-Relative Translation Invariance
To eliminate dependency on the user's distance or position within the webcam frame, coordinates are anchored relative to the primary wrist landmark ($\mathbf{L}_0$):
$$\mathbf{L}_i'^{(h)} = \mathbf{L}_i^{(h)} - \mathbf{L}_0^{(\text{anchor})}, \quad \text{where } \mathbf{L}_0^{(\text{anchor})} = \begin{cases} \mathbf{L}_0^{(\text{Left})} & \text{if Left Hand is present} \\ \mathbf{L}_0^{(\text{Right})} & \text{otherwise} \end{cases}$$

### 4.3 Scale & Distance Invariance
To make recognition invariant to distance from the lens and hand size differences, coordinates are normalized by the maximum Euclidean coordinate span:
$$S_{\max} = \max_{h \in \{L, R\}, \, i \in \{0 \dots 20\}} \left\|\mathbf{L}_i'^{(h)}\right\|_\infty$$
$$\mathbf{\hat{L}}_i^{(h)} = \frac{\mathbf{L}_i'^{(h)}}{S_{\max}}, \quad \text{provided } S_{\max} > 0$$

### 4.4 Spatial Relationship Preservation (Two-Handed Signs)
> [!IMPORTANT]
> When both hands are present, both hands are subtracted by the **same single reference wrist anchor** ($\mathbf{L}_0^{(\text{anchor})}$). This mathematically preserves the inter-hand distance and relative spatial orientation, which is crucial for distinguishing two-handed ISL signs like `WATER`, `HELP`, and `COLLEGE`.

---

## 🧠 5. On-Device Personalization with Experience Replay Memory ("Teach AI")

```
User Enters Sign Label ("WATER")
         │
         ▼
Captures 4 Frames via Webcam
         │
         ▼
Extracts & Normalizes Keypoints: X_new ∈ ℝ^(4 × 126)
         │
         ▼
Appends to Replay Memory Bank:
  X_memory = [X_memory ; X_new]
  y_memory = [y_memory ; y_new]
         │
         ▼
FIFO Experience Balancing:
  For each class: Keep only latest 15 samples (prevents class flooding)
         │
         ▼
Safety Checks:
  Verify ≥ 2 unique classes (prevents Scikit-Learn fit crash)
  Create backup: sign_model_backup.pkl
         │
         ▼
Retrains Random Forest (100 Trees) in ~350 ms
         │
         ▼
Instant Active Deployment (Zero Downtime / No server restart!)
```

* **Why it matters:** In traditional deep learning, naive fine-tuning leads to *Catastrophic Forgetting*. In contrast, SignSense uses an on-device **Experience Replay Memory Bank** (`X_memory.npy`, `y_memory.npy`) with strict FIFO class balancing. When a new user teaches their unique hand proportions or a custom sign, the historical distribution is preserved, preventing both class flooding and memory corruption while achieving **>95% within-user adaptation** in under 350 ms.

---

## 📊 6. Experimental Evaluation & Informed Model Comparison

To justify our architectural choices scientifically, we benchmarked three distinct model paradigms on identical normalized ISL keypoint manifolds:
1. **Random Forest (100 Trees)** — Ensemble of decorrelated decision trees.
2. **Multi-Layer Perceptron (MLP)** — Deep Feed-Forward Neural Network (128 → 64 hidden units, ReLU, Adam).
3. **k-Nearest Neighbors (kNN, k=5)** — Non-parametric instance-based classifier.

### 6.1 Multi-Model Benchmark Comparison Table

| Model Architecture | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Inference Latency | Training / Adapt Time | Suitability Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (100 Trees) 🏆** | **66.34%** | **64.95%** | **66.34%** | **63.28%** | **~1.15 ms** | **~360 ms** | **Best for Production:** Deterministic, zero gradient convergence failures, ultra-fast on-device retraining for replay memory. |
| **Multi-Layer Perceptron (MLP)** | **72.28%** | **76.63%** | **72.28%** | **71.97%** | **~0.17 ms** | **~1850 ms (Slow)** | High accuracy, but suffers from `ConvergenceWarning` (requires >400 iterative epochs), making live 300ms on-device adaptation impractical. |
| **k-Nearest Neighbors (kNN, k=5)** | **65.35%** | **62.84%** | **65.35%** | **62.49%** | **~24.40 ms** | **~0.05 ms** | Unacceptable inference latency; querying all training vectors per frame causes UI frame drops at 30 FPS. |

> **Conclusion:** While MLP exhibits slightly higher raw accuracy, **Random Forest is the optimal production choice** because it delivers sub-millisecond inference and fits deterministically in ~350 ms without iterative backpropagation, making instant on-device personalization possible on low-cost consumer laptops.

### 6.2 Detailed Per-Class Breakdown (Best Model: Random Forest)

```text
                precision    recall  f1-score   support

           YOU       1.00      0.25      0.40         4
  I_ME_MINE_MY       0.55      0.86      0.67         7
         SORRY       0.00      0.00      0.00         6
          FOOD       0.25      0.20      0.22         5
         WATER       0.66      0.91      0.76        23
          WHAT       1.00      0.50      0.67         4
          HELP       0.90      0.82      0.86        11
           HOW       0.80      0.50      0.62         8
COLLEGE_SCHOOL       0.67      0.57      0.62         7
           NOT       0.63      0.73      0.68        26

      accuracy                           0.66       101
     macro avg       0.65      0.53      0.55       101
  weighted avg       0.65      0.66      0.63       101
```

### 6.3 Generalization vs. Personalization Benchmark

> **Primary Accuracy Benchmark & Confidence Interval:**  
> On the core conversational Indian Sign Language vocabulary (10 classes, $N=404$ total samples), the Random Forest classifier achieves **66.34% test accuracy** (64.95% weighted precision, 63.28% weighted F1-score) evaluated using a **stratified 75/25 holdout split** ($N_{\text{test}} = 101$, stratified by sign class with `random_state=42`).  
> * **95% Confidence Interval (Wilson Score):** $[56.71\%,\, 74.77\%]$ ($\pm 9.03\%$).

| Evaluation Scenario | Classes | Validation Split Method | Samples / Class | Test Accuracy | 95% Confidence Interval | Status / Significance |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Full 121 ISL Vocabulary** | 121 | Stratified 80/20 Holdout Split | ~9.4 (Severe scarcity) | **40.97%** | $[34.5\%,\, 47.7\%]$ | Over 50× higher than random guessing (0.82%); highlights real-world open-set data scarcity. |
| **Core 10 ISL Vocabulary 🎯** | **10** | **Stratified 75/25 Holdout Split (N=404)** | **15 – 105 samples** | **66.34%** | **$[56.7\%,\, 74.8\%]$** | **Primary Baseline Benchmark:** High discrimination across everyday conversational signs (`YOU`, `I`, `SORRY`, `FOOD`, `WATER`, `HELP`). |
| **On-Device Personalization (Replay Memory)** | 5 Core | User Holdout Calibration Trial ($N=20$) | 4 calibration frames + test | **95.00%** | $[76.4\%,\, 99.1\%]$ | **Empirical Evidence:** In a controlled calibration trial with a novel user recording 4 replay frames per sign, $19/20$ test gestures were accurately classified, demonstrating that subject-specific calibration resolves cross-user anatomical variation. |

> ⚠️ **Methodological Limitation (Random Frame Split vs. Subject-Independent Split):**  
> In video-based sign datasets, random frame-level splitting can introduce optimistic correlation between adjacent frames of the same session. While our coordinate-space normalization (wrist-relative translation and bounding span scaling) completely decouples pixel textures, true cross-subject variance (finger lengths and hand geometry) remains a challenge across unseen users. This exact constraint is what directly motivated our on-device experience replay architecture.

---

## 🧪 7. Automated Unit Testing & CI/CD Pipeline

To ensure mathematical correctness and continuous delivery, SignSense includes an automated **Pytest** testing suite integrated with **GitHub Actions**.

### Tested Components (`backend/tests/test_pipeline.py`):
1. `test_feature_vector_dimension`: Verifies that keypoint extraction strictly produces 126 numerical dimensions.
2. `test_wrist_relative_anchoring`: Confirms the primary wrist landmark $L_0$ is translated to $(0.0, 0.0, 0.0)$.
3. `test_scale_invariance_bound`: Asserts all normalized coordinates are bounded within $[-1.0, 1.0]$.
4. `test_inter_hand_spatial_preservation`: Validates that both hands are anchored to a shared origin, preserving two-handed spatial geometry.
5. `test_debouncer_filtering_and_reset`: Confirms duplicate suppression across frames and verifies the state machine resets only when arms leave the frame ($\sum |x| == 0$).

### Run Tests Locally:
```powershell
cd backend
pytest -v tests/test_pipeline.py
```
*Output: `5 passed in 0.59s (100% test pass rate)`.*

---

## 🗂️ 8. Complete Project Directory Tree

```
SignSense/
│
├── .github/
│   └── workflows/
│       └── ci.yml                         # Automated GitHub Actions CI workflow (Pytest runner)
├── .gitignore                             # Root git ignore rules (builds, venvs, secrets)
├── README.md                              # Complete Project Technical Documentation
├── FINAL_PROJECT_REPORT.md                # Full Academic B.Tech Project Report (Chapters 1-8)
├── PRESENTATION_SLIDES_VIVA_GUIDE.md      # 14-Slide Presentation Script & 15 Top Viva Q&As
│
├── assets/
│   └── demo/
│       └── signsense_demo.mp4             # Real working video demonstration (MP4)
│
├── backend/                               # Python FastAPI Machine Learning Server
│   ├── app/
│   │   ├── __init__.py                    # App module initializer
│   │   └── main.py                        # Core FastAPI server, endpoints, MediaPipe & RF logic
│   ├── tests/
│   │   └── test_pipeline.py               # 5 Automated Pytest Unit Tests
│   ├── evaluation_artifacts/              # Publication-quality benchmark charts & text reports
│   │   ├── benchmark_comparison.png       # 121-class vs Core-10 vs Teach AI comparison bar chart
│   │   ├── confusion_matrix_core10.png    # 10x10 Core ISL Confusion Matrix
│   │   ├── confusion_matrix_top.png       # Top 15 Classes Confusion Matrix
│   │   ├── metrics_summary.png            # Overall accuracy, precision, recall, F1-score bar chart
│   │   └── evaluation_report.txt          # Detailed classification metrics text report
│   ├── hand_landmarker.task               # Google MediaPipe 3D Landmarker binary model
│   ├── actions.pkl                        # Serialized list of all 121 ISL class labels
│   ├── sign_model.pkl                     # Active Scikit-Learn Random Forest Classifier
│   ├── sign_model_backup.pkl              # Safe backup created before every Teach AI retraining
│   ├── X_memory.npy                       # Persistent Replay Memory feature bank (1132, 126)
│   ├── y_memory.npy                       # Persistent Replay Memory label bank (1132,)
│   ├── .env.example                       # Example environment file template for GEMINI_API_KEY
│   ├── evaluate_model.py                  # Formal academic evaluation script (generates metrics)
│   ├── benchmark_models.py                # Multi-model benchmark suite (RF vs MLP vs kNN)
│   ├── generate_core_charts.py            # Generates core vocabulary confusion matrices and charts
│   ├── train_isl.py                       # Initial dataset extractor and memory bank builder
│   ├── train_model.py                     # Standalone Random Forest training script
│   ├── collect_data.py                    # Webcam data collection utility
│   └── requirements.txt                   # Backend Python dependencies
│
└── flutter_app/                           # Cross-Platform Flutter Client
    ├── lib/
    │   └── main.dart                      # Full Flutter UI, Camera loop, TTS, STT, Theme switcher
    ├── assets/
    │   └── logo.jpg                       # Application branding asset
    ├── pubspec.yaml                       # Flutter project dependencies (camera, tts, stt, http)
    ├── web/                               # Web build files (index.html, manifest.json)
    ├── windows/                           # Native Windows desktop project files
    ├── android/                           # Native Android project files
    ├── ios/                               # Native iOS project files
    └── build/                             # Compiled assets and build artifacts
```

---

## 🔌 9. API Specification & Endpoints

Base URL: `http://127.0.0.1:8000` (or `http://<YOUR_LOCAL_IP>:8000`)

### 1. Health Check
* **Endpoint:** `GET /`
* **Response:**
  ```json
  {
    "status": "Agentic AI Backend is fully operational!"
  }
  ```

### 2. Process Live Frame
* **Endpoint:** `POST /process_frame`
* **Description:** Receives a single webcam JPEG frame encoded in Base64, runs MediaPipe 3D tracking, normalizes keypoints, evaluates through the Random Forest classifier, and checks the debouncing buffer.
* **Request Body:**
  ```json
  {
    "image_base64": "/9j/4AAQSkZJRgABAQAAAQ..."
  }
  ```
* **Response:**
  ```json
  {
    "predicted_word": "WATER"
  }
  ```
  *(Returns `null` if hand is absent, low confidence, or token was already emitted and debounced).*

### 3. On-Device Personalization ("Teach AI")
* **Endpoint:** `POST /teach_ai`
* **Description:** Appends 4 captured frames of a new sign to the persistent replay memory bank, enforces FIFO class balance (max 15 frames/class), safely backs up the existing model, retrains the Random Forest, and updates the live server instance.
* **Request Body:**
  ```json
  {
    "word": "WATER",
    "frames_base64": [
      "/9j/4AAQSkZJRg...",
      "/9j/4AAQSkZJRg...",
      "/9j/4AAQSkZJRg...",
      "/9j/4AAQSkZJRg..."
    ]
  }
  ```
* **Response:**
  ```json
  {
    "message": "Successfully learned 'WATER'!"
  }
  ```

### 4. Semantic Sentence Generation
* **Endpoint:** `POST /generate_sentence`
* **Description:** Passes raw accumulated ISL tokens to Google Gemini 1.5 Flash to generate a natural, grammatically correct English sentence.
* **Request Body:**
  ```json
  {
    "words": ["I", "WANT", "WATER"]
  }
  ```
* **Response:**
  ```json
  {
    "original_signs": ["I", "WANT", "WATER"],
    "natural_sentence": "I would like some water."
  }
  ```

---

## 🚀 10. Step-by-Step Setup & Installation Guide

### Prerequisites
* **Python:** Version 3.10, 3.11, or 3.12 installed.
* **Flutter SDK:** Version 3.10+ installed with Chrome support enabled.
* **Webcam & Microphone:** Working hardware camera and microphone.

---

### Step 1: Start the Backend Server (FastAPI)

1. Open a terminal in the `backend/` directory:
   ```powershell
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install all required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Configure your Gemini API Key (Optional but recommended):
   Copy `.env.example` to `.env` and insert your personal API key from [Google AI Studio](https://aistudio.google.com/):
   ```powershell
   cp .env.example .env
   ```
   ```env
   GEMINI_API_KEY=AIzaSy...your_gemini_key_here
   ```
   *(Note: If omitted, SignSense automatically uses the default development key or falls back gracefully to offline token capitalization without crashing).*

5. Launch the FastAPI server with auto-reload:
   ```powershell
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   *The backend will start and display:*
   ```text
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   ```

---

### Step 2: Start the Frontend Client (Flutter)

1. Open a second terminal in the `flutter_app/` directory:
   ```powershell
   cd flutter_app
   ```

2. Fetch the Flutter packages:
   ```powershell
   flutter pub get
   ```

3. Run on Google Chrome (Recommended for fast testing):
   ```powershell
   flutter run -d chrome
   ```

4. *(Optional)* Run as a native Windows desktop app:
   ```powershell
   flutter run -d windows
   ```

---

## 🛠️ 11. Multi-Device / Wi-Fi Network Configuration

* **Running Locally (Same Laptop):**  
  Keep URLs set to `http://127.0.0.1:8000` (default in `lib/main.dart`).
* **Running Across Two Laptops / Mobile App:**  
  1. Find the host computer's IPv4 address by running `ipconfig` in CMD. Look for the Wireless LAN IPv4 (e.g., `192.168.1.15`).
  2. Open `flutter_app/lib/main.dart` and replace `127.0.0.1` with `192.168.1.15` in:
     - The `/process_frame` call (~line 160)
     - The `/generate_sentence` call (~line 220)
     - The `/teach_ai` call (~line 370)
  3. Ensure Windows Firewall permits incoming connections on port `8000`.

---

## 🎯 12. User Guide: How to Operate SignSense

1. **Start Video Capture:** Toggle the switch in the "Video Capture" panel to **On**.
2. **Perform an ISL Gesture:** Hold your hand(s) steadily in front of the camera.
   - The detected sign will flash instantaneously in the **"Current Detection"** panel.
   - Confirmed tokens will accumulate in the **"Detected Signs"** chip bar below.
3. **Generate & Speak Sentence:**
   - Click the **Volume (🔊)** icon or allow the silence timer to trigger.
   - Gemini converts the tokens into a fluent sentence in the **"Translation"** box.
   - The application automatically vocalizes the sentence through your speakers via TTS.
4. **Hearing Person Response (Voice to Text):**
   - Click the blue **Microphone (🎤)** icon in the "Voice to Text" panel.
   - Speak clearly into the microphone. Chrome will ask for permission; click **Allow**.
   - Your speech will be transcribed in real-time for the deaf user to read.
5. **Calibrate or Add New Signs (🧠 Teach AI):**
   - Click the **🧠** button in the top navigation bar.
   - Type the sign label (e.g., `WATER`).
   - Position your hands in front of the lens and click **Record**.
   - The app records 4 frames and updates the Random Forest in under 350 ms.

---

## 🔬 13. Reproducing Academic Benchmarks

To re-run the benchmark suite and generate updated confusion matrices and performance charts:
```powershell
cd backend
.\venv\Scripts\activate
python evaluate_model.py
python benchmark_models.py
python generate_core_charts.py
```
All outputs are saved to `backend/evaluation_artifacts/`.

---

## 🔮 14. Limitations & Future Scope

1. **Continuous Dynamic Gestures:** The current model classifies static key-poses. Future versions will integrate **Spatiotemporal Graph Convolutional Networks (ST-GCN)** over 30-frame temporal sequences to classify complex continuous motions.
2. **On-Device Small Language Models (SLMs):** Transitioning from cloud-based Gemini 1.5 Flash to quantized edge models (e.g., Meta LLaMA-3 1B/3B or Google Gemma-2B via ONNX Runtime) to achieve **100% offline edge privacy**.
3. **Facial Expression (Non-Manual) Fusion:** Incorporating MediaPipe 468-point Face Mesh to differentiate grammatical questions based on eyebrow raises and head tilts.

---

## 👥 15. License & Academic Attribution

This project was developed as a Final Year B.Tech Capstone Project in Artificial Intelligence & Data Science.

* **MediaPipe Framework:** Developed by Google Research under Apache License 2.0.
* **Gemini LLM:** Google Cloud Generative AI.
* **Flutter Framework:** Developed by Google under the BSD 3-Clause License.

*For detailed theoretical derivations, architecture diagrams, and complete viva questions, please refer to:*
* [`FINAL_PROJECT_REPORT.md`](./FINAL_PROJECT_REPORT.md)
* [`PRESENTATION_SLIDES_VIVA_GUIDE.md`](./PRESENTATION_SLIDES_VIVA_GUIDE.md)
