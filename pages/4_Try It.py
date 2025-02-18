# -----------------------------------
# Imports and Logging Setup
# -----------------------------------
import os
import sys
import logging
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
import re
import csv
import time
import numpy as np
import pandas as pd
import cv2
import av
import streamlit as st
import joblib
from huggingface_hub import HfApi, hf_hub_download, Repository
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model
import mediapipe as mp

# Optional: Append additional module paths if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))
# from sample_utils.download import download_file
# from sample_utils.turn import get_ice_servers

# For ICE servers, define a simple function (or use your own get_ice_servers)
def get_ice_servers():
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

# Logging configuration
DEBUG_MODE = False  # Set to True for debugging
def debug_log(message):
    if DEBUG_MODE:
        logging.debug(message)
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app_debug.log", mode='w')]
)
logger = logging.getLogger(__name__)
logger.info("🚀 Logging initialized!")

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(layout="wide", page_title="Gesture Recognition")
st.title("Gesture Recognition System")

# -----------------------------------
# Hugging Face Token and API Setup
# -----------------------------------
# Use the working token variable name from your viz page
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()
hf_api = HfApi(token=hf_token)

# Define repository details (update these as needed)
MODEL_REPO_NAME = "dk23/A3CP_models"  # Repository containing models and encoders
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# -----------------------------------
# Session State Initialization
# -----------------------------------
if "landmark_queue" not in st.session_state:
    st.session_state.landmark_queue = deque(maxlen=1000)
if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()
if "model" not in st.session_state:
    st.session_state.model = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False
if "recognized_action" not in st.session_state:
    st.session_state.recognized_action = "Waiting for prediction..."

landmark_queue = st.session_state.landmark_queue
lock = st.session_state.lock

# -----------------------------------
# MediaPipe Initialization & Constants
# -----------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# Landmark counts and angle names
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468
angle_names_base = [
    'thumb_mcp', 'thumb_ip', 
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip', 
    'ring_mcp', 'ring_pip', 'ring_dip', 
    'little_mcp', 'little_pip', 'little_dip'
]
left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]

# (Optional) CSV header for saving keyframes (if needed later)
@st.cache_data
def load_csv_header():
    pose_headers = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
    left_hand_headers = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
    right_hand_headers = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
    face_headers = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]
    return (['class', 'sequence_id'] + pose_headers + left_hand_headers + left_hand_angle_names +
            right_hand_headers + right_hand_angle_names + face_headers)
header = load_csv_header()

# -----------------------------------
# Helper Functions
# -----------------------------------
def store_landmarks(row_data):
    """Store landmark data in a thread-safe manner."""
    with lock:
        landmark_queue.append(row_data)
    debug_log(f"Stored {len(landmark_queue)} frames in queue")

def get_landmark_queue():
    """Retrieve a snapshot of the landmark queue."""
    with lock:
        queue_snapshot = list(landmark_queue)
    debug_log(f"Snapshot taken with {len(queue_snapshot)} frames.")
    return queue_snapshot

def clear_landmark_queue():
    """Clear the landmark queue in a thread-safe manner."""
    with lock:
        debug_log(f"Clearing queue... Current size: {len(landmark_queue)}")
        landmark_queue.clear()
    debug_log("Landmark queue cleared.")

def calculate_angle(a, b, c):
    """Calculate the angle formed at point b (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def hand_angles(hand_landmarks):
    """Compute joint angles for the hand."""
    if (not hand_landmarks) or all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)
    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i):
        return [h[i][0], h[i][1]]
    return [
        calculate_angle(pt(1), pt(2), pt(3)),  # thumb_mcp
        calculate_angle(pt(2), pt(3), pt(4)),  # thumb_ip
        calculate_angle(pt(0), pt(5), pt(6)),  # index_mcp
        calculate_angle(pt(5), pt(6), pt(7)),  # index_pip
        calculate_angle(pt(6), pt(7), pt(8)),  # index_dip
        calculate_angle(pt(0), pt(9), pt(10)), # middle_mcp
        calculate_angle(pt(9), pt(10), pt(11)),# middle_pip
        calculate_angle(pt(10), pt(11), pt(12)),# middle_dip
        calculate_angle(pt(0), pt(13), pt(14)),# ring_mcp
        calculate_angle(pt(13), pt(14), pt(15)),# ring_pip
        calculate_angle(pt(14), pt(15), pt(16)),# ring_dip
        calculate_angle(pt(0), pt(17), pt(18)),# little_mcp
        calculate_angle(pt(17), pt(18), pt(19)),# little_pip
        calculate_angle(pt(18), pt(19), pt(20)) # little_dip
    ]

def process_frame(frame):
    """
    Process a single frame using MediaPipe Holistic.
    Returns:
        annotated_image: Frame with landmarks drawn.
        pose_data, left_hand_data, right_hand_data, face_data: Landmark lists.
        left_hand_angles, right_hand_angles: Computed hand angles.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    def extract_data(landmarks, count):
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count
    pose_data = extract_data(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = extract_data(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = extract_data(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = extract_data(results.face_landmarks, num_face_landmarks)
    left_hand_angles = hand_angles(left_hand_data)
    right_hand_angles = hand_angles(right_hand_data)
    return annotated_image, pose_data, left_hand_data, left_hand_angles, right_hand_data, right_hand_angles, face_data

def flatten_landmarks(pose_data, left_hand_data, left_hand_angles, right_hand_data, right_hand_angles, face_data):
    """Flatten all landmark and angle data into a single list."""
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    face_flat = [val for landmark in face_data for val in landmark]
    return pose_flat + left_hand_flat + left_hand_angles + right_hand_flat + right_hand_angles + face_flat

# -----------------------------------
# Model & Encoder Selection UI (Sidebar)
# -----------------------------------
st.sidebar.header("Model & Encoder Selection")
@st.cache_data
def get_model_files():
    repo_files = hf_api.list_repo_files(MODEL_REPO_NAME, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    return model_files, encoder_files

model_files, encoder_files = get_model_files()
selected_model_file = st.sidebar.selectbox("Select Model File", model_files)
selected_encoder_file = st.sidebar.selectbox("Select Encoder File", encoder_files)

if st.sidebar.button("Load Model & Encoder"):
    with st.spinner("Loading model and encoder..."):
        try:
            model_path = hf_hub_download(MODEL_REPO_NAME, selected_model_file, repo_type="model", local_dir=LOCAL_MODEL_DIR, token=hf_token)
            model = load_model(model_path)
            encoder_path = hf_hub_download(MODEL_REPO_NAME, selected_encoder_file, repo_type="model", local_dir=LOCAL_MODEL_DIR, token=hf_token)
            label_encoder = joblib.load(encoder_path)
            st.session_state.model = model
            st.session_state.label_encoder = label_encoder
            st.session_state.model_loaded = True
            st.sidebar.success("Model and encoder loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model/encoder: {e}")

# -----------------------------------
# Video Streaming and Prediction UI
# -----------------------------------
if st.session_state.get("model_loaded", False):
    # Button to start the video stream
    if st.button("Start Video Stream"):
        st.session_state.stream_active = True

    # Define video frame callback
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        input_bgr = frame.to_ndarray(format="bgr24")
        debug_log("video_frame_callback triggered")
        (annotated_image,
         pose_data,
         left_hand_data,
         left_hand_angles,
         right_hand_data,
         right_hand_angles,
         face_data) = process_frame(input_bgr)
        row_data = flatten_landmarks(pose_data, left_hand_data, left_hand_angles, right_hand_data, right_hand_angles, face_data)
        if row_data and any(row_data):
            store_landmarks(row_data)
        # Run prediction if possible
        try:
            input_data = np.array(row_data).reshape(1, -1)
            prediction = st.session_state.model.predict(input_data)
            predicted_index = np.argmax(prediction)
            predicted_class = st.session_state.label_encoder.inverse_transform([predicted_index])[0]
            st.session_state.recognized_action = predicted_class
        except Exception as e:
            st.session_state.recognized_action = f"Error: {e}"
        cv2.putText(annotated_image, f"Action: {st.session_state.recognized_action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    # Start the video stream if activated
    if st.session_state.get("stream_active", False):
        webrtc_streamer(
            key="gesture_streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
    
    # Display the recognized action below the video
    st.subheader("Recognized Action:")
    st.write(st.session_state.get("recognized_action", "Waiting for prediction..."))
else:
    st.info("Please load a model and encoder from the sidebar to enable video streaming.")
