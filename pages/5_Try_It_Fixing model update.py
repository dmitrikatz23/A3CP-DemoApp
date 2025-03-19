
# #model selector
import logging
import mediapipe as mp
from pathlib import Path
from typing import List, NamedTuple
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download, HfApi
import os
import re
import pandas as pd
import os
from collections import deque
import threading
import sys
import time
from huggingface_hub import Repository, HfApi

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# -----------------------------------
# Logging Setup
# -----------------------------------
DEBUG_MODE = False  # Set to True only for debugging

def debug_log(message):
    if DEBUG_MODE:
        logging.info(message)

logging.basicConfig(
    level=logging.WARNING if not DEBUG_MODE else logging.DEBUG,  # Only show debug logs if DEBUG_MODE is True
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  
        logging.FileHandler("app_debug.log", mode='w')  
    ]
)

logger = logging.getLogger(__name__)
logger.info("🚀 Logging is initialized!")

# -----------------------------------
# Threading and Session State Management
# -----------------------------------

# Ensure landmark_queue persists across Streamlit reruns
if "landmark_queue" not in st.session_state:
    st.session_state.landmark_queue = deque(maxlen=1000)

# Use the stored queue in session state
landmark_queue = st.session_state.landmark_queue

# Thread-safe lock for WebRTC thread access stored in session state
if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()
        
lock = st.session_state.lock

# Initialize landmark_queue_snapshot in session state to prevent KeyError
if 'landmark_queue_snapshot' not in st.session_state:
    st.session_state['landmark_queue_snapshot'] = []

# Initialize action_word in session state
if 'action_word' not in st.session_state:
    st.session_state['action_word'] = "Unknown_Action"

# Initialize a flag to track streamer state
if 'streamer_running' not in st.session_state:
    st.session_state['streamer_running'] = False

def store_landmarks(row_data):
    with lock:  # Ensures only one thread writes at a time
        landmark_queue.append(row_data)
    
    debug_log(f"Stored {len(landmark_queue)} frames in queue")  # Debugging
    debug_log(f"First 5 values: {row_data[:5]}")  # Debug first few values


def get_landmark_queue():
    """Thread-safe function to retrieve a copy of the landmark queue."""
    with lock:
        queue_snapshot = list(landmark_queue)  # Copy queue safely

    # Store snapshot for session persistence
    st.session_state.landmark_queue_snapshot = queue_snapshot
    debug_log(f"🟡 Snapshot taken with {len(queue_snapshot)} frames.")

    return queue_snapshot  # Return the copied queue


def clear_landmark_queue():
    """Thread-safe function to clear the landmark queue."""
    with lock:
        debug_log(f"🟡 Clearing queue... Current size: {len(landmark_queue)}")
        landmark_queue.clear()
    debug_log("🟡 Landmark queue cleared.")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

# -----------------------------
# Model Update Button v. 12:53
# -----------------------------

def update_model():
    """Fetch and load the latest model and encoder."""
    latest_model, latest_encoder = get_latest_model()
    
    if latest_model and latest_encoder:
        st.session_state["tryit_model_confirmed"] = True
        model_path = hf_hub_download(model_repo_name, latest_model, local_dir=LOCAL_MODEL_DIR, repo_type="model", token=hf_token)
        encoder_path = hf_hub_download(model_repo_name, latest_encoder, local_dir=LOCAL_MODEL_DIR, repo_type="model", token=hf_token)
        
        st.session_state["tryit_model"] = tf.keras.models.load_model(model_path)
        st.session_state["tryit_encoder"] = joblib.load(encoder_path)
        st.success(f"Latest model and encoder loaded: {latest_model}, {latest_encoder}")
    else:
        st.warning("No valid model/encoder pairs found in the repository.")





# -----------------------------------
# MediaPipe Initialization & Landmark Constants
# -----------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Number of landmarks in various MediaPipe models
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468


# -----------------------------------
# Angle Names for Hands
# -----------------------------------
angle_names_base = [
    'thumb_mcp', 'thumb_ip', 
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip', 
    'ring_mcp', 'ring_pip', 'ring_dip', 
    'little_mcp', 'little_pip', 'little_dip'
]
left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]


# -----------------------------------
# Landmark Header Definitions
# -----------------------------------
pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]


# -----------------------------------
# CSV Header Loader
# -----------------------------------
@st.cache_data
def load_csv_header():
    """
    Generate and return the CSV header.
    """
    return (
        ['class', 'sequence_id']
        + pose_landmarks
        + left_hand_landmarks
        + left_hand_angle_names
        + right_hand_landmarks
        + right_hand_angle_names
        + face_landmarks
    )

header = load_csv_header()

# -----------------------------------
# MediaPipe Model Loader
# -----------------------------------
@st.cache_resource
def load_mediapipe_model():
    """
    Load and cache the MediaPipe Holistic model for optimized video processing.
    """
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_model()

# -----------------------------------
# Helper Functions
# -----------------------------------
def calculate_angle(a, b, c):
    """
    Calculate the angle formed at point b by (a -> b -> c).
    Returns angle in degrees (0-180).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def hand_angles(hand_landmarks):
    """
    Calculate angles for each finger joint using hand landmarks.
    Returns a list of angles for all joints in the expected order.
    """
    if (not hand_landmarks) or all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)

    # Convert list of landmarks into a dict for easy indexing
    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i):
        return [h[i][0], h[i][1]]

    # Thumb angles
    thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
    thumb_ip  = calculate_angle(pt(2), pt(3), pt(4))

    # Index finger angles
    index_mcp = calculate_angle(pt(0), pt(5), pt(6))
    index_pip = calculate_angle(pt(5), pt(6), pt(7))
    index_dip = calculate_angle(pt(6), pt(7), pt(8))

    # Middle finger angles
    middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
    middle_pip = calculate_angle(pt(9), pt(10), pt(11))
    middle_dip = calculate_angle(pt(10), pt(11), pt(12))

    # Ring finger angles
    ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
    ring_pip = calculate_angle(pt(13), pt(14), pt(15))
    ring_dip = calculate_angle(pt(14), pt(15), pt(16))

    # Little finger angles
    little_mcp = calculate_angle(pt(0), pt(17), pt(18))
    little_pip = calculate_angle(pt(17), pt(18), pt(19))
    little_dip = calculate_angle(pt(18), pt(19), pt(20))

    # Return all angles in a flat list
    return [
        thumb_mcp, thumb_ip,
        index_mcp, index_pip, index_dip,
        middle_mcp, middle_pip, middle_dip,
        ring_mcp, ring_pip, ring_dip,
        little_mcp, little_pip, little_dip
    ]

def process_frame(frame):
    """
    Process a single BGR frame with MediaPipe Holistic.
    Returns:
        annotated_image: The original frame annotated with landmarks.
        pose_data: 2D + visibility for each pose landmark.
        left_hand_data: 2D + visibility for each landmark in the left hand.
        left_hand_angles_data: The angles computed for the left hand joints.
        right_hand_data: 2D + visibility for each landmark in the right hand.
        right_hand_angles_data: The angles computed for the right hand joints.
        face_data: 2D + visibility for each face landmark.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks if available
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Extract data
    def extract_data(landmarks, count):
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count

    pose_data = extract_data(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = extract_data(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = extract_data(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = extract_data(results.face_landmarks, num_face_landmarks)

    # Compute joint angles for hands
    left_hand_angles_data = hand_angles(left_hand_data)
    right_hand_angles_data = hand_angles(right_hand_data)

    return (
        annotated_image,
        pose_data,
        left_hand_data,
        left_hand_angles_data,
        right_hand_data,
        right_hand_angles_data,
        face_data
    )

def flatten_landmarks(
    pose_data,
    left_hand_data,
    left_hand_angles_data,
    right_hand_data,
    right_hand_angles_data,
    face_data
):
    """
    Flatten all landmark data and angles into a single 1D list.
    """
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    left_hand_angles_flat = left_hand_angles_data
    right_hand_angles_flat = right_hand_angles_data
    face_flat = [val for landmark in face_data for val in landmark]

    return (
        pose_flat +
        left_hand_flat +
        left_hand_angles_flat +
        right_hand_flat +
        right_hand_angles_flat +
        face_flat
    )

def calculate_velocity(landmarks):
    """
    Calculate velocity from landmark coordinates (frame-to-frame displacement).
    Expected input: NxM array (N frames, M features).
    """
    velocities = []
    for i in range(1, len(landmarks)):
        velocity = np.linalg.norm(landmarks[i] - landmarks[i-1])
        velocities.append(velocity)
    return np.array(velocities)

def calculate_acceleration(velocities):
    """
    Calculate acceleration from velocity (frame-to-frame change in velocity).
    """
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = np.abs(velocities[i] - velocities[i-1])
        accelerations.append(acceleration)
    return np.array(accelerations)

def identify_keyframes(
    landmarks,
    velocity_threshold=0.1,
    acceleration_threshold=0.1
):
    """
    Identify keyframes based on velocity and acceleration thresholds.
    `landmarks` is a NxM array representing frames (N) by flattened features (M).
    """
    velocities = calculate_velocity(landmarks)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i + 1)  # +1 offset because acceleration index starts at 1
    return keyframes

# -----------------------------------
# WebRTC Video Callback
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    input_bgr = frame.to_ndarray(format="bgr24")
    debug_log("📷 video_frame_callback triggered")  # Debugging

    # Process frame with MediaPipe
    (
        annotated_image,
        pose_data,
        left_hand_data,
        left_hand_angles,
        right_hand_data,
        right_hand_angles,
        face_data
    ) = process_frame(input_bgr)

    if pose_data or left_hand_data or right_hand_data or face_data:
        debug_log("🟢 Landmarks detected, processing...")
    else:
        debug_log("⚠️ No landmarks detected, skipping storage.")

    # Flatten and store landmarks
    row_data = flatten_landmarks(
        pose_data,
        left_hand_data,
        left_hand_angles,
        right_hand_data,
        right_hand_angles,
        face_data
    )

    if row_data and any(row_data):  # Ensure data is not empty
        debug_log("Storing landmarks in queue...")
        store_landmarks(row_data)
    else:
        debug_log("⚠️ No valid landmarks detected. Skipping storage.")

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")



# -----------------------------
# Hugging Face Setup
# -----------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is set.")
    st.stop()

hf_api = HfApi()
model_repo_name = "dk23/A3CP_models"
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)





# -----------------------------
# Sidebar: Model Selection
# -----------------------------
@st.cache_data

def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs from Hugging Face."""
    repo_files = hf_api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]

    pairs = {}
    for mf in model_files:
        ts = mf[len("LSTM_model_"):-3]  # Extract timestamp
        pairs.setdefault(ts, {})["model"] = mf
    for ef in encoder_files:
        ts = ef[len("label_encoder_"):-4]  # Extract timestamp
        pairs.setdefault(ts, {})["encoder"] = ef

    valid_pairs = [(ts, items["model"], items["encoder"]) for ts, items in pairs.items() if "model" in items and "encoder" in items]
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

with st.sidebar:
    if st.button("Refresh Models from Repository"): #new button
        update_model() #new
        
    st.subheader("Select a Model/Encoder Pair")
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found.")
    else:
        pair_options = {f"{ts} | Model: {mf} | Encoder: {ef}": (mf, ef) for ts, mf, ef in model_encoder_pairs}
        selected_label = st.selectbox("Choose a matched pair:", list(pair_options.keys()))

        if selected_label:
            chosen_model, chosen_encoder = pair_options[selected_label]
            st.write("**Selected Model:**", chosen_model)
            st.write("**Selected Encoder:**", chosen_encoder)

        if st.button("Confirm Model") and selected_label:
            st.session_state["tryit_selected_pair"] = pair_options[selected_label]
            st.session_state["tryit_model_confirmed"] = True

            model_path = os.path.join(LOCAL_MODEL_DIR, chosen_model)
            encoder_path = os.path.join(LOCAL_MODEL_DIR, chosen_encoder)

            if not os.path.exists(model_path):
                hf_hub_download(model_repo_name, chosen_model, local_dir=LOCAL_MODEL_DIR, repo_type="model", token=hf_token)
            if not os.path.exists(encoder_path):
                hf_hub_download(model_repo_name, chosen_encoder, local_dir=LOCAL_MODEL_DIR, repo_type="model", token=hf_token)

            # Load Model & Encoder
            st.session_state["tryit_model"] = tf.keras.models.load_model(model_path)
            st.session_state["tryit_encoder"] = joblib.load(encoder_path)
            st.success("Model and encoder loaded successfully!")

# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("WebRTC Stream")
    if st.session_state.get("tryit_model_confirmed"):
        webrtc_streamer(
            key="tryit-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

with right_col:
    st.header("Predicted Gesture")
    
    prediction_placeholder = st.empty()  # Placeholder for dynamic updates
    
    while True:
        if len(landmark_queue) >= 30 and st.session_state.get("tryit_model"):
            model = st.session_state["tryit_model"]
            encoder = st.session_state["tryit_encoder"]
            
            # Prepare input for the model
            X_input = np.array(list(landmark_queue)[-30:])  # Last 30 frames
            X_input = np.expand_dims(X_input, axis=0)  # Shape: (1, sequence_length, num_features)
            
            # Predict gesture
            y_pred = model.predict(X_input)
            gesture_index = np.argmax(y_pred, axis=1)[0]
            gesture_name = encoder.inverse_transform([gesture_index])[0] if np.max(y_pred) > 0.5 else "No gesture detected"
            
            
            # Store prediction in session state
            st.session_state["tryit_predicted_text"] = gesture_name
            
            # Update displayed prediction dynamically
            prediction_placeholder.write(f"**Prediction:** {gesture_name}")
            
        time.sleep(0.5)  # Update interval