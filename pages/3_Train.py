import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
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
# Hugging Face Integration
# ---------------------------
from huggingface_hub import Repository

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

    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i): return [h[i][0], h[i][1]]

    # Thumb angles
    thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
    thumb_ip  = calculate_angle(pt(2), pt(3), pt(4))

    # Index finger
    index_mcp = calculate_angle(pt(0), pt(5), pt(6))
    index_pip = calculate_angle(pt(5), pt(6), pt(7))
    index_dip = calculate_angle(pt(6), pt(7), pt(8))

    # Middle finger
    middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
    middle_pip = calculate_angle(pt(9), pt(10), pt(11))
    middle_dip = calculate_angle(pt(10), pt(11), pt(12))

    # Ring finger
    ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
    ring_pip = calculate_angle(pt(13), pt(14), pt(15))
    ring_dip = calculate_angle(pt(14), pt(15), pt(16))

    # Little finger
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

def process_frame(frame):
    """
    Process a single BGR frame with MediaPipe Holistic.
    Returns:
        annotated_image: The original frame annotated with landmarks.
        pose_data, left_hand_data, left_hand_angles_data,
        right_hand_data, right_hand_angles_data, face_data
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

    def extract_data(landmarks, count):
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count

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
    `landmarks` is a NxM array representing frames.
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
    `landmarks` is a NxM array representing frames by flattened features.
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
    """
    WebRTC callback that uses MediaPipe Holistic to process frames in real-time.
    Returns an annotated frame.
    
    IMPORTANT: We store each frame's landmarks if an action is confirmed.
    """
    input_bgr = frame.to_ndarray(format="bgr24")
    (
        annotated_image,
        pose_data,
        left_hand_data,
        left_hand_angles_data,
        right_hand_data,
        right_hand_angles_data,
        face_data
    ) = process_frame(input_bgr)

    # Collect frames for the currently confirmed action
    if st.session_state.get('action_confirmed') and st.session_state.get('current_action'):
        action_word = st.session_state['current_action']
        frames_collector = st.session_state['actions'].get(action_word, [])
        frames_collector.append(
            (
                pose_data,
                left_hand_data,
                left_hand_angles_data,
                right_hand_data,
                right_hand_angles_data,
                face_data
            )
        )
        st.session_state['actions'][action_word] = frames_collector
        # DEBUG: Log collected frames
        st.write(f"Collected {len(frames_collector)} frames for action: '{action_word}'")  # DEBUG


    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -----------------------------------
# CSV Setup
# -----------------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# If a CSV file hasn't been set in session state, create one with a timestamped name
if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state["csv_file"] = os.path.join(csv_folder, f"all_actions_recorded_{session_start_str}.csv")

csv_file = st.session_state["csv_file"]

@st.cache_data
def initialize_csv(file_name, header):
    """
    Initialize the CSV file with the header row if it doesn't exist.
    """
    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
    return True

if "csv_initialized" not in st.session_state:
    st.session_state["csv_initialized"] = initialize_csv(csv_file, header)

# -----------------------------------
# Hugging Face Setup
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")  # Must be set in your HF Spaces secrets

repo = None
if hf_token:
    repo_name = "dk23/A3CP_actions"  # Your dataset repository
    local_repo_path = "local_repo"
    git_user = "A3CP_bot"
    git_email = "no-reply@huggingface.co"

    try:
        repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")
        repo.git_config_username_and_email(git_user, git_email)
    except Exception as e:
        st.error(f"Error setting up Hugging Face repository: {e}")
else:
    st.warning("Hugging Face token not found. CSV will only be saved locally.")

def save_csv_to_huggingface():
    """
    Pushes the local CSV to Hugging Face if repo is configured.
    """
    if not repo:
        st.info("No Hugging Face repository configured; skipping push.")
        return

    if not os.path.exists(csv_file):
        st.warning("No CSV file found to save.")
        return

    try:
        # Copy CSV into local repo folder
        df = pd.read_csv(csv_file)
        os.makedirs(repo.local_dir, exist_ok=True)
        csv_repo_path = os.path.join(repo.local_dir, os.path.basename(csv_file))
        df.to_csv(csv_repo_path, index=False)

        # Commit and push
        repo.git_add(os.path.basename(csv_file))
        repo.git_commit("Update actions CSV")
        repo.git_push()
        st.success(f"CSV successfully pushed to Hugging Face: {repo_name}")
    except Exception as ex:
        st.error(f"Error saving to repository: {ex}")

# -----------------------------------
# NEW: Process Keyframes Function
# -----------------------------------
def process_and_save_rows():
    """
    Processes recorded actions, identifies keyframes, and writes rows to the CSV.
    """
    all_rows = []

    if st.session_state['actions']:
        # Iterate over each action
        for action, all_frames in st.session_state['actions'].items():
            if all_frames and len(all_frames) > 1:
                flat_landmarks_per_frame = []

                for f in all_frames:
                    (
                        pose_data,
                        left_hand_data,
                        left_hand_angles_data,
                        right_hand_data,
                        right_hand_angles_data,
                        face_data
                    ) = f
# Testing to inspect data before calling flatten
                    st.text("Checking variables inside process_and_save_rows:")
                    if pose_data is None:
                        st.warning("Pose data is None.")
                    elif not pose_data:
                        st.warning("Pose data is empty.")
                    else:
                        st.write("Pose data:", pose_data)

                    if left_hand_data is None:
                        st.warning("Left hand data is None.")
                    elif not left_hand_data:
                        st.warning("Left hand data is empty.")
                    else:
                        st.write("Left hand data:", left_hand_data)

                    if right_hand_data is None:
                        st.warning("Right hand data is None.")
                    elif not right_hand_data:
                        st.warning("Right hand data is empty.")
                    else:
                        st.write("Right hand data:", right_hand_data)

                    if face_data is None:
                        st.warning("Face data is None.")
                    elif not face_data:
                        st.warning("Face data is empty.")
                    else:
                        st.write("Face data:", face_data)
# End testing
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
                st.write(f"Flattening and identifying keyframes for action '{action}'...")  # DEBUG
                keyframes = identify_keyframes(
                    flat_landmarks_per_frame,
                    velocity_threshold=0.01,  # Lowered threshold for debugging
                    acceleration_threshold=0.01,
                )
                st.write(f"Detected {len(keyframes)} keyframes for action '{action}'")  # DEBUG
                
                # Append rows for each keyframe
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

        # Write rows to the CSV new version
        if all_rows:
            try:
                with open(csv_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(all_rows)
                st.success(f"All recorded actions appended to '{csv_file}'")

                # DEBUG: Check file contents
                st.write("Verifying written rows...")
                with open(csv_file, 'r') as f:
                    st.text(f.read())

            except Exception as e:
                st.error(f"Error writing to CSV: {e}")
        else:
            st.warning("No rows to write to CSV")

# -----------------------------------
# Streamlit UI and Logic
# -----------------------------------
st.title("Record an Action")

# Initialize session state variables for handling actions, sequences, etc.
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
FRAME_WINDOW = right_col.image([])
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
        if st.session_state.get('active_streamer_key') is not None:
            st.session_state['action_confirmed'] = False
            old_key = st.session_state['active_streamer_key']
            if old_key in st.session_state:
                del st.session_state[old_key]

        # Prepare for a new action
        st.session_state['current_action'] = action_word
        st.session_state['actions'][action_word] = []
        st.session_state['action_confirmed'] = True
        st.session_state['active_streamer_key'] = f"record-actions-{sanitized_action_word}"
        st.success(f"Action '{action_word}' confirmed!")

    # If an action has been confirmed, show the WebRTC streamer
    if st.session_state.get('action_confirmed', False):
        streamer_key = st.session_state['active_streamer_key']
        st.info(f"Streaming activated! Perform the action: {action_word}")

        # Launch Streamlit WebRTC streamer
        webrtc_streamer(
            key=streamer_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

    # Button to process rows and push CSV to Hugging Face
    if st.button("Save CSV to Hugging Face"):
        # 1. Process and save keyframe rows
        process_and_save_rows()
        # 2. Push the updated CSV
        save_csv_to_huggingface()

# -----------------------------------
# Right/Main Area: Display Recorded CSV (if any)
# -----------------------------------
st.header("Recorded Actions Summary (Current CSV)")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if not df.empty:
        unique_actions = df['class'].unique()
        num_actions = len(unique_actions)
        num_cols = 8
        num_rows = (num_actions + num_cols - 1) // num_cols

        # Display each unique action in a grid
        for r in range(num_rows):
            row_actions = unique_actions[r * num_cols:(r + 1) * num_cols]
            cols = st.columns(num_cols)
            for col, a in zip(cols, row_actions):
                if a:
                    col.markdown(
                        f"<h4 style='margin:10px; text-align:center; font-family:sans-serif;'>{a}</h4>",
                        unsafe_allow_html=True
                    )

        # Display the entire CSV for reference
        st.subheader("Full CSV Data")
        df.reset_index(drop=True, inplace=True)  # Reset index to ensure proper display
        st.dataframe(df)
    else:
        st.info("CSV is initialized but has no data rows yet.")
else:
    st.info("No CSV file found yet.")