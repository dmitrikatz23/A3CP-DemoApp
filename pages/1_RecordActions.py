# -----------------------------------
# Imports
# -----------------------------------
# Essential Python libraries
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

# Third-party libraries
import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import re

# For file operations and handling
import csv
import time
import pandas as pd
import os
from datetime import datetime

# Adjust system path for importing utilities
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Custom utility imports
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# -----------------------------------
# Logging Setup
# -----------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(layout="wide")

# -----------------------------------
# MediaPipe Initialization
# -----------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# -----------------------------------
# WebRTC Video Callback Function
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    WebRTC callback that uses MediaPipe Holistic to process frames in real-time.
    """
    input_bgr = frame.to_ndarray(format="bgr24")
    (annotated_image,
     _pose_data,
     _left_hand_data,
     _left_hand_angles,
     _right_hand_data,
     _right_hand_angles,
     _face_data) = process_frame(input_bgr)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -----------------------------------
# Define CSV Header Structure
# -----------------------------------
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468

# Base angle names for hands
angle_names_base = [
    'thumb_mcp', 'thumb_ip', 
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip', 
    'ring_mcp', 'ring_pip', 'ring_dip', 
    'little_mcp', 'little_pip', 'little_dip'
]

# Complete list of landmark names
left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]
pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]

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
def calculate_velocity(landmarks):
    """
    Calculate velocity from landmark coordinates.
    """
    velocities = []
    for i in range(1, len(landmarks)):
        velocity = np.linalg.norm(landmarks[i] - landmarks[i-1])
        velocities.append(velocity)
    return np.array(velocities)

def calculate_acceleration(velocities):
    """
    Calculate acceleration from velocity values.
    """
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = np.abs(velocities[i] - velocities[i-1])
        accelerations.append(acceleration)
    return np.array(accelerations)

def identify_keyframes(landmarks, velocity_threshold=0.1, acceleration_threshold=0.1):
    """
    Identify keyframes based on velocity and acceleration thresholds.
    """
    velocities = calculate_velocity(landmarks)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i + 1)  # +1 because acceleration index starts from 1
    return keyframes

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(
        c[1] - b[1], c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_frame(frame):
    """
    Process a single BGR frame with MediaPipe Holistic.
    Returns:
        annotated_image, pose_data, left_hand_data, left_hand_angles,
        right_hand_data, right_hand_angles, face_data
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

    # Data extraction
    def extract_data(landmarks, count):
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count

    pose_data = extract_data(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = extract_data(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = extract_data(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = extract_data(results.face_landmarks, num_face_landmarks)

    return annotated_image, pose_data, left_hand_data, [], right_hand_data, [], face_data

# -----------------------------------
# CSV Initialization
# -----------------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state["csv_file"] = os.path.join(csv_folder, f"all_actions_recorded_{session_start_str}.csv")

csv_file = st.session_state["csv_file"]

@st.cache_data
def initialize_csv(file_name, header):
    """
    Initialize the CSV file with the header.
    """
    with open(file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
    return True

if "csv_initialized" not in st.session_state:
    st.session_state["csv_initialized"] = initialize_csv(csv_file, header)

# -----------------------------------
# Streamlit UI and Logic
# -----------------------------------
# Streamlit UI for recording actions
st.title("Record an Action")

# Initialize session state variables
if 'actions' not in st.session_state:
    st.session_state['actions'] = {}
if 'record_started' not in st.session_state:
    st.session_state['record_started'] = False
if 'sequence_id' not in st.session_state:
    st.session_state['sequence_id'] = 0
if 'action_confirmed' not in st.session_state:
    st.session_state['action_confirmed'] = False  # Track whether action is confirmed
if 'active_streamer_key' not in st.session_state:
    st.session_state['active_streamer_key'] = None  # Track the active streamer

left_col, right_col = st.columns([1, 2])
FRAME_WINDOW = right_col.image([])
status_bar = right_col.empty()

file_exists = os.path.isfile(csv_file)

with left_col:
    st.header("Controls")
    action_word = st.text_input("Enter the intended meaning for the action e.g. I'm hungry")

    # "Confirm Action" logic
    if st.button("Confirm Action") and action_word:
        # Clean the action_word to make a valid key
        sanitized_action_word = re.sub(r'[^a-zA-Z0-9_]', '_', action_word.strip())

        # If there is already an active streamer, clear it
        if st.session_state.get('active_streamer_key') is not None:
            # Reset the previous streamer state
            st.session_state['action_confirmed'] = False
            del st.session_state[st.session_state['active_streamer_key']]  # Clear state for the old streamer

        # Set up the new action and streamer key
        st.session_state['actions'][action_word] = None
        st.session_state['action_confirmed'] = True
        st.session_state['active_streamer_key'] = f"record-actions-{sanitized_action_word}"
        st.success(f"Action '{action_word}' confirmed!")

    # Conditionally show the WebRTC streamer for the confirmed action
    if st.session_state.get('action_confirmed', False):
        # Retrieve the active key for the streamer
        streamer_key = st.session_state['active_streamer_key']

        # Display WebRTC streamer for the current action
        st.info(f"Streaming activated! Perform the action: {action_word}")
        webrtc_streamer(
            key=streamer_key,  # Ensure the key is clean and unique
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

st.header("Recorded Actions")
if st.session_state['actions']:
    all_rows = []
    for action, all_frames in st.session_state['actions'].items():
        if all_frames is not None and len(all_frames) > 1:
            flat_landmarks_per_frame = []
            for f in all_frames:
                (pose_data, 
                 left_hand_data, 
                 left_hand_angles_data, 
                 right_hand_data, 
                 right_hand_angles_data, 
                 face_data) = f

                flat_landmarks_per_frame.append(
                    flatten_landmarks(
                        pose_data, 
                        left_hand_data, 
                        left_hand_angles_data,
                        right_hand_data, 
                        right_hand_angles_data, 
                        face_data
                    )
                )

            flat_landmarks_per_frame = np.array(flat_landmarks_per_frame)
            keyframes = identify_keyframes(
                flat_landmarks_per_frame, 
                velocity_threshold=0.1, 
                acceleration_threshold=0.1
            )
            for kf in keyframes:
                if kf < len(all_frames):
                    st.session_state['sequence_id'] += 1
                    (pose_data, 
                     left_hand_data, 
                     left_hand_angles_data, 
                     right_hand_data, 
                     right_hand_angles_data, 
                     face_data) = all_frames[kf]
                    
                    row_data = flatten_landmarks(
                        pose_data, 
                        left_hand_data, 
                        left_hand_angles_data, 
                        right_hand_data, 
                        right_hand_angles_data, 
                        face_data
                    )
                    row = [action, st.session_state['sequence_id']] + row_data
                    all_rows.append(row)
                    
    if all_rows:
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(all_rows)
        st.success(f"All recorded actions appended to '{csv_file}'")

        df = pd.read_csv(csv_file)
        unique_actions = df['class'].unique()
        num_actions = len(unique_actions)
        num_cols = 8
        num_rows = (num_actions + num_cols - 1) // num_cols

        for r in range(num_rows):
            row_actions = unique_actions[r*num_cols:(r+1)*num_cols]
            cols = st.columns(num_cols)
            for col, a in zip(cols, row_actions):
                if a:
                    col.markdown(
                        f"<h4 style='margin:10px; text-align:center; font-family:sans-serif;'>{a}</h4>",
                        unsafe_allow_html=True
                    )

else:
    st.info("No actions recorded yet.")
