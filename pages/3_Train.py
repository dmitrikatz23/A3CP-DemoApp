# Page 4: Try it out
#import streamlit as st
#st.title('this page will let the user try the app and see if it can identify the mapped meaning of their actions ')

# -----------------------------------
# Imports
# -----------------------------------
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase
import re

import csv
import time
import pandas as pd
import os
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# ---------------------------
# Additional Hugging Face Imports
# ---------------------------
from huggingface_hub import Repository

# -----------------------------------
# Logging Setup
# -----------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(layout="wide")

# -----------------------------------
# MediaPipe Initialization & Landmark Constants
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
    """Calculate angle formed at b by (a->b->c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def hand_angles(hand_landmarks):
    """Calculate angles for each finger joint using hand landmarks."""
    if (not hand_landmarks) or all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)

    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i): return [h[i][0], h[i][1]]

    # Thumb
    thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
    thumb_ip = calculate_angle(pt(2), pt(3), pt(4))

    # Index
    index_mcp = calculate_angle(pt(0), pt(5), pt(6))
    index_pip = calculate_angle(pt(5), pt(6), pt(7))
    index_dip = calculate_angle(pt(6), pt(7), pt(8))

    # Middle
    middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
    middle_pip = calculate_angle(pt(9), pt(10), pt(11))
    middle_dip = calculate_angle(pt(10), pt(11), pt(12))

    # Ring
    ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
    ring_pip = calculate_angle(pt(13), pt(14), pt(15))
    ring_dip = calculate_angle(pt(14), pt(15), pt(16))

    # Little
    little_mcp = calculate_angle(pt(0), pt(17), pt(18))
    little_pip = calculate_angle(pt(17), pt(18), pt(19))
    little_dip = calculate_angle(pt(18), pt(19), pt(20))

    return [
        thumb_mcp, thumb_ip,
        index_mcp, index_pip, index_dip,
        middle_mcp, middle_pip, middle_dip,
        ring_mcp, ring_pip, ring_dip,
        little_mcp, little_pip, little_dip
    ]

def process_frame(frame_bgr):
    """
    Process a single BGR frame with MediaPipe Holistic.
    Returns annotated_image + all landmarks data.
    """
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0,0,0]]*count

    pose_data = extract_data(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = extract_data(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = extract_data(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = extract_data(results.face_landmarks, num_face_landmarks)

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
    """Flatten all landmark data and angles into a single 1D list."""
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    left_hand_angles_flat = left_hand_angles_data
    right_hand_angles_flat = right_hand_angles_data
    face_flat = [val for landmark in face_data for val in landmark]

    return (
        pose_flat
        + left_hand_flat
        + left_hand_angles_flat
        + right_hand_flat
        + right_hand_angles_flat
        + face_flat
    )

def calculate_velocity(landmarks):
    """Calculate velocity from NxM landmark array."""
    velocities = []
    for i in range(1, len(landmarks)):
        velocities.append(np.linalg.norm(landmarks[i] - landmarks[i-1]))
    return np.array(velocities)

def calculate_acceleration(velocities):
    """Calculate acceleration from velocity array."""
    accelerations = []
    for i in range(1, len(velocities)):
        accelerations.append(np.abs(velocities[i] - velocities[i-1]))
    return np.array(accelerations)

def identify_keyframes(landmarks, velocity_threshold=0.1, acceleration_threshold=0.1):
    """Identify keyframes based on velocity/acceleration thresholds."""
    velocities = calculate_velocity(landmarks)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i+1)  # +1 offset
    return keyframes

# -----------------------------------
# Custom Video Processor
# -----------------------------------
class HolisticFrameProcessor(VideoProcessorBase):
    """
    Custom VideoProcessor that collects frames + landmarks for later processing.
    """
    def __init__(self):
        self.collected_frames = []
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        (
            annotated_image,
            pose_data,
            left_hand_data,
            left_hand_angles_data,
            right_hand_data,
            right_hand_angles_data,
            face_data
        ) = process_frame(frame_bgr)

        # Store the landmark data (excluding annotated image) in memory
        self.collected_frames.append(
            (
                pose_data,
                left_hand_data,
                left_hand_angles_data,
                right_hand_data,
                right_hand_angles_data,
                face_data
            )
        )
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -----------------------------------
# CSV Setup
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
    Initialize the CSV file with the header row if it doesn't exist.
    """
    with open(file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
    return True

if "csv_initialized" not in st.session_state:
    st.session_state["csv_initialized"] = initialize_csv(csv_file, header)

# ---------------------------
# Hugging Face Setup
# ---------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure it's set as 'Recorded_Datasets' in your Space secrets.")
    st.stop()

repo_name = "dk23/A3CP_actions"  # Your HF Dataset Repository
local_repo_path = "local_repo"
git_user = "A3CP_bot"  # Generic username for commits
git_email = "no-reply@huggingface.co"  # Generic email for commits

repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")
repo.git_config_username_and_email(git_user, git_email)

def save_csv_to_huggingface():
    """Saves the local CSV file to the Hugging Face dataset repository."""
    if not os.path.exists(csv_file):
        st.warning("No CSV file found to save.")
        return

    os.makedirs(local_repo_path, exist_ok=True)
    repo_file_path = os.path.join(local_repo_path, os.path.basename(csv_file))

    df = pd.read_csv(csv_file)
    df.to_csv(repo_file_path, index=False)

    repo.git_add(os.path.basename(csv_file))
    repo.git_commit("Update actions CSV")
    repo.git_push()

# -----------------------------------
# Streamlit UI and Logic
# -----------------------------------
st.title("Record an Action")

if 'actions' not in st.session_state:
    st.session_state['actions'] = {}
if 'record_started' not in st.session_state:
    st.session_state['record_started'] = False
if 'sequence_id' not in st.session_state:
    st.session_state['sequence_id'] = 0
if 'action_confirmed' not in st.session_state:
    st.session_state['action_confirmed'] = False
if 'active_streamer_key' not in st.session_state:
    st.session_state['active_streamer_key'] = None

left_col, right_col = st.columns([1, 2])
FRAME_WINDOW = right_col.empty()
status_bar = right_col.empty()

file_exists = os.path.isfile(csv_file)

# -----------------------------------
# Left Column: Controls
# -----------------------------------
with left_col:
    st.header("Controls")
    action_word = st.text_input("Enter the intended meaning for the action e.g. I'm hungry")

    # Confirm Action button
    if st.button("Confirm Action") and action_word:
        # Sanitize action word for internal use
        sanitized_action_word = re.sub(r'[^a-zA-Z0-9_]', '_', action_word.strip())

        # If an active streamer already exists, clear its state
        if st.session_state.get('active_streamer_key'):
            st.session_state['action_confirmed'] = False
            old_key = st.session_state['active_streamer_key']
            if old_key in st.session_state:
                del st.session_state[old_key]

        st.session_state['actions'][action_word] = None
        st.session_state['action_confirmed'] = True
        st.session_state['active_streamer_key'] = f"record-actions-{sanitized_action_word}"
        st.success(f"Action '{action_word}' confirmed!")

    # If an action has been confirmed, show the WebRTC streamer
    if st.session_state.get('action_confirmed', False):
        streamer_key = st.session_state['active_streamer_key']
        st.info(f"Streaming activated! Perform the action: {action_word}")
        
        # Start the streamer with our custom video processor
        webrtc_ctx = webrtc_streamer(
            key=streamer_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=HolisticFrameProcessor,
            async_processing=True,
        )

        # Button to stop recording and store frames in session state
        if st.button("Stop Recording"):
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                frames_collected = webrtc_ctx.video_processor.collected_frames
                webrtc_ctx.video_processor.collected_frames = []  # reset

                # Store frames in session state for this action
                st.session_state['actions'][action_word] = frames_collected
                st.success(f"Recorded frames saved for action '{action_word}'!")
            else:
                st.warning("No frames collected or streamer not active.")

    # Button to push CSV to Hugging Face
    if st.button("Save CSV to Hugging Face"):
        try:
            save_csv_to_huggingface()
            st.success(f"CSV successfully saved to {repo_name}")
        except Exception as e:
            st.error(f"Error saving to repository: {e}")

# -----------------------------------
# Right/Main Area: Recorded Actions
# -----------------------------------
st.header("Recorded Actions")

if st.session_state['actions']:
    all_rows = []

    # Iterate over each action in the session
    for action, all_frames in st.session_state['actions'].items():
        if all_frames is not None and len(all_frames) > 1:
            flat_landmarks_per_frame = []
            # Flatten each frame's landmarks for later keyframe detection
            for f in all_frames:
                (
                    pose_data,
                    left_hand_data,
                    left_hand_angles_data,
                    right_hand_data,
                    right_hand_angles_data,
                    face_data
                ) = f

                flattened = flatten_landmarks(
                    pose_data,
                    left_hand_data,
                    left_hand_angles_data,
                    right_hand_data,
                    right_hand_angles_data,
                    face_data
                )
                flat_landmarks_per_frame.append(flattened)

            flat_landmarks_per_frame = np.array(flat_landmarks_per_frame)

            # Identify keyframes
            keyframes = identify_keyframes(
                flat_landmarks_per_frame,
                velocity_threshold=0.1,
                acceleration_threshold=0.1
            )

            # For each keyframe, append a new row to CSV
            for kf in keyframes:
                if kf < len(all_frames):
                    st.session_state['sequence_id'] += 1
                    (
                        pose_data,
                        left_hand_data,
                        left_hand_angles_data,
                        right_hand_data,
                        right_hand_angles_data,
                        face_data
                    ) = all_frames[kf]

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

    # Write new rows to CSV if any
    if all_rows:
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(all_rows)

        st.success(f"All recorded actions appended to '{csv_file}'")

        # Display a summary of recorded classes
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
        st.warning("No keyframes were identified. Try performing a more dynamic gesture.")
else:
    st.info("No actions recorded yet.")
