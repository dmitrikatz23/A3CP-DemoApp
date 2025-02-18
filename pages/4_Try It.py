# Import necessary libraries
import os
import logging
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
import re
import csv
import time
import pandas as pd
import numpy as np
import streamlit as st
import cv2
import av
import mediapipe as mp
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from huggingface_hub import HfApi, hf_hub_download, Repository
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# -----------------------------------
# Logging Setup
# -----------------------------------
DEBUG_MODE = False  # Set to True only for debugging

def debug_log(message):
    if DEBUG_MODE:
        logging.info(message)

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.WARNING,
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
if "landmark_queue" not in st.session_state:
    st.session_state.landmark_queue = deque(maxlen=1000)

landmark_queue = st.session_state.landmark_queue

if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()

lock = st.session_state.lock

if 'landmark_queue_snapshot' not in st.session_state:
    st.session_state['landmark_queue_snapshot'] = []

if 'action_word' not in st.session_state:
    st.session_state['action_word'] = "Unknown_Action"

if 'streamer_running' not in st.session_state:
    st.session_state['streamer_running'] = False

# -----------------------------------
# Hugging Face Integration
# -----------------------------------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face token not found. Please set the 'HF_TOKEN' environment variable.")
    st.stop()

api = HfApi()

# Model repository details
model_repo_name = "dk23/A3CP_models"
local_model_path = "local_models"
os.makedirs(local_model_path, exist_ok=True)

# Fetch available models and encoders from the repository
@st.cache_data
def get_model_files():
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    return model_files, encoder_files

model_files, encoder_files = get_model_files()

# -----------------------------------
# Streamlit UI for Model and Encoder Selection
# -----------------------------------
st.title("Gesture Recognition with Model Selection")

# Model selection
selected_model_file = st.selectbox("Select a pre-trained model:", model_files)
selected_encoder_file = st.selectbox("Select a label encoder:", encoder_files)

# Download and load the selected model and encoder
if st.button("Load Model and Encoder"):
    with st.spinner("Downloading and loading model and encoder..."):
        try:
            # Download model
            model_path = hf_hub_download(
                repo_id=model_repo_name,
                filename=selected_model_file,
                repo_type="model",
                local_dir=local_model_path,
                token=hf_token
            )
            model = load_model(model_path)
            st.success(f"Model '{selected_model_file}' loaded successfully.")

            # Download encoder
            encoder_path = hf_hub_download(
                repo_id=model_repo_name,
                filename=selected_encoder_file,
                repo_type="model",
                local_dir=local_model_path,
                token=hf_token
            )
            label_encoder = joblib.load(encoder_path)
            st.success(f"Label encoder '{selected_encoder_file}' loaded successfully.")

            # Store in session state
            st.session_state['model'] = model
            st.session_state['label_encoder'] = label_encoder

        except Exception as e:
            st.error(f"Error loading model or encoder: {e}")
            st.stop()

# -----------------------------------
# MediaPipe Initialization & Constants
# -----------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

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

pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks + 1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand + 1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand + 1) for axis in ['x', 'y', 'v']]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks + 1) for axis in ['x', 'y', 'v']]

# -----------------------------------
# CSV Header Loader
# -----------------------------------
@st.cache_data
def load_csv_header():
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
    Calculate the angle between three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_hand_angles(hand_landmarks):
    """
    Extract angles for hand landmarks.
    """
    if not hand_landmarks:
        return [0] * 15  # Return a list of zeros if no landmarks

    # Define points for angle calculation
    joint_list = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),  # Thumb
        (0, 5, 6), (5, 6, 7), (6, 7, 8),  # Index
        (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Middle
        (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Ring
        (0, 17, 18), (17, 18, 19), (18, 19, 20)  # Pinky
    ]

    angles = []
    for joint in joint_list:
        a = [hand_landmarks[joint[0]].x, hand_landmarks[joint[0]].y]
        b = [hand_landmarks[joint[1]].x, hand_landmarks[joint[1]].y]
        c = [hand_landmarks[joint[2]].x, hand_landmarks[joint[2]].y]
        angle = calculate_angle(a, b, c)
        angles.append(angle)

    return angles

def process_frame(frame):
    """
    Process a single frame to extract landmarks and angles.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = frame.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    # Extract landmarks and angles
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
    left_hand_landmarks = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
    right_hand_landmarks = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
    face_landmarks = results.face_landmarks.landmark if results.face_landmarks else []

    left_hand_angles = extract_hand_angles(left_hand_landmarks)
    right_hand_angles = extract_hand_angles(right_hand_landmarks)

    return annotated_image, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks, left_hand_angles, right_hand_angles

def flatten_landmarks(pose, left_hand, right_hand, face, left_hand_angles, right_hand_angles):
    """
    Flatten all landmarks and angles into a single list.
    """
    def flatten(landmarks):
        return [coord for landmark in landmarks for coord in (landmark.x, landmark.y, landmark.z)]

    pose_flat = flatten(pose)
    left_hand_flat = flatten(left_hand)
    right_hand_flat = flatten(right_hand)
    face_flat = flatten(face)

    return pose_flat + left_hand_flat + right_hand_flat + face_flat + left_hand_angles + right_hand_angles

def store_landmarks(landmarks):
    """
    Store landmarks in a thread-safe manner.
    """
    with lock:
        landmark_queue.append(landmarks)



 
