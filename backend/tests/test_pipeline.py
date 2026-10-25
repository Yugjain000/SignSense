import collections
import numpy as np
import pytest

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockCategory:
    def __init__(self, category_name):
        self.category_name = category_name

class MockDetectionResult:
    def __init__(self, hand_landmarks=None, handedness=None):
        self.hand_landmarks = hand_landmarks if hand_landmarks is not None else []
        self.handedness = handedness if handedness is not None else []

# Import or define the canonical extract_keypoints logic
def normalize_keypoints(detection_result):
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

class DebouncerStateMachine:
    def __init__(self, buffer_size=5, min_consensus=3):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.last_confirmed_word = ""
        self.buffer_size = buffer_size
        self.min_consensus = min_consensus

    def step(self, keypoints, current_prediction):
        # 1. Hands dropped check
        if np.sum(keypoints) == 0:
            self.last_confirmed_word = ""
            return None

        # 2. Append to buffer
        self.buffer.append(current_prediction)

        # 3. Debounce consensus check
        if len(self.buffer) == self.buffer_size:
            most_common = collections.Counter(self.buffer).most_common(1)[0]
            if most_common[1] >= self.min_consensus:
                confirmed = most_common[0]
                self.buffer.clear()
                if confirmed != self.last_confirmed_word:
                    self.last_confirmed_word = confirmed
                    return confirmed
        return None

# ==========================================
# PYTEST UNIT TESTS
# ==========================================

def test_feature_vector_dimension():
    """Test 1: Output vector must have exactly 126 numerical dimensions."""
    mock_res = MockDetectionResult()
    kps = normalize_keypoints(mock_res)
    assert kps.shape == (126,)
    assert np.all(kps == 0.0)

def test_wrist_relative_anchoring():
    """Test 2: Wrist landmark (index 0) of the anchor hand must be translated to exactly (0, 0, 0)."""
    # Create arbitrary hand coordinates starting at wrist (0.5, 0.4, 0.2)
    left_hand = [MockLandmark(0.5 + i*0.01, 0.4 + i*0.01, 0.2 + i*0.005) for i in range(21)]
    mock_res = MockDetectionResult(
        hand_landmarks=[left_hand],
        handedness=[[MockCategory('Left')]]
    )
    kps = normalize_keypoints(mock_res)
    
    # Left hand is first 63 values (21 landmarks x 3)
    wrist_x, wrist_y, wrist_z = kps[0], kps[1], kps[2]
    assert np.isclose(wrist_x, 0.0)
    assert np.isclose(wrist_y, 0.0)
    assert np.isclose(wrist_z, 0.0)

def test_scale_invariance_bound():
    """Test 3: Max absolute value across all coordinates must be normalized to exactly 1.0 (or within [-1, 1])."""
    # Create hand with large absolute span
    right_hand = [MockLandmark(10.0 + i*5.0, -20.0 + i*2.0, 5.0) for i in range(21)]
    mock_res = MockDetectionResult(
        hand_landmarks=[right_hand],
        handedness=[[MockCategory('Right')]]
    )
    kps = normalize_keypoints(mock_res)
    
    assert np.max(np.abs(kps)) <= 1.0 + 1e-6
    assert np.isclose(np.max(np.abs(kps)), 1.0)

def test_inter_hand_spatial_preservation():
    """Test 4: When both hands are present, the spatial distance between left wrist and right wrist is preserved."""
    left_hand = [MockLandmark(0.2, 0.3, 0.1) for _ in range(21)]
    right_hand = [MockLandmark(0.8, 0.3, 0.1) for _ in range(21)]
    
    # Initial Euclidean distance between left wrist and right wrist = |0.8 - 0.2| = 0.6
    raw_dist = np.linalg.norm(np.array([0.2, 0.3, 0.1]) - np.array([0.8, 0.3, 0.1]))
    
    mock_res = MockDetectionResult(
        hand_landmarks=[left_hand, right_hand],
        handedness=[[MockCategory('Left')], [MockCategory('Right')]]
    )
    kps = normalize_keypoints(mock_res)
    
    # Left wrist: indices 0..2, Right wrist: indices 63..65
    left_wrist_norm = kps[0:3]
    right_wrist_norm = kps[63:66]
    
    # Right wrist must not be (0, 0, 0) because it is anchored to left wrist, preserving spatial relationship
    assert not np.allclose(right_wrist_norm, [0.0, 0.0, 0.0])
    assert np.allclose(left_wrist_norm, [0.0, 0.0, 0.0])

def test_debouncer_filtering_and_reset():
    """Test 5: Debouncer prevents duplicate spamming and only resets when hands leave the camera frame."""
    debouncer = DebouncerStateMachine(buffer_size=5, min_consensus=3)
    dummy_hand_kps = np.ones(126)
    zero_hand_kps = np.zeros(126)

    # Stream 5 identical predictions "HELLO"
    emitted = []
    for _ in range(5):
        res = debouncer.step(dummy_hand_kps, "HELLO")
        if res: emitted.append(res)
    
    # Should emit "HELLO" exactly once
    assert emitted == ["HELLO"]

    # Stream another 5 identical predictions "HELLO" without dropping hands
    emitted_subsequent = []
    for _ in range(5):
        res = debouncer.step(dummy_hand_kps, "HELLO")
        if res: emitted_subsequent.append(res)
        
    # Duplicate should be suppressed!
    assert emitted_subsequent == []

    # Drop hands out of frame (zero keypoints)
    res_drop = debouncer.step(zero_hand_kps, "NONE")
    assert res_drop is None
    assert debouncer.last_confirmed_word == "" # Reset successful

    # User re-engages and signs "HELLO" again
    re_emitted = []
    for _ in range(5):
        res = debouncer.step(dummy_hand_kps, "HELLO")
        if res: re_emitted.append(res)
        
    # Now it should be allowed again!
    assert re_emitted == ["HELLO"]
