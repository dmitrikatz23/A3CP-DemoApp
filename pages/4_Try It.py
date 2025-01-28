# app.py - A single-button approach to capture frames and save them to CSV

import logging
import os
import csv
import re
import time
import sys
import pandas as pd
import numpy as np
import cv2
import av
import mediapipe as mp

from queue import Queue
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple
from huggingface_hub import Repository

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("twilio").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Global Queue for Frames
# -----------------------------------------------------------------------------
frame_queue = Queue()

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Record an Action (One-Click: Capture + Save to CSV)")

# -----------------------------------------------------------------------------
# Hugging Face Integration (Optional)
# -----------------------------------------------------------------------------
hf_token = os.getenv("Recorded_Datasets")  # Must be set in your HF Spaces secrets
repo = None
if hf_token:
    repo_name = "dk23/A3CP_actions"  # Your dataset repository
    local_repo_path = "local_repo"
    git_user = "A3CP_bot"
    git_email = "no-reply@huggingface.co"
    try:
        repo = Repository(
            local_dir=local_repo_path,
            clone_from=repo_name,
            use_auth_token=hf_token,
            repo_type="dataset"
        )
        repo.git_config_username_and_email(git_user, git_email)
    except Exception as e:
        st.error(f"Error setting up Hugging Face repository: {e}")
else:
    st.warning("Hugging Face token not found. CSV will only be saved locally.")

def save_csv_to_huggingface(csv_file):
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
        df = pd.read_csv(csv_file)
        os.makedirs(repo.local_dir, exist_ok=True)
        csv_repo_path = os.path.join(repo.local_dir, os.path.basename(csv_file))
        df.to_csv(csv_repo_path, index=False)

        # Commit and push
        repo.git_add(csv_file)
        repo.git_commit("Update A3CP actions CSV")
        repo.git_push()
        st.success(f"CSV successfully pushed to Hugging Face repository: {repo_name}")
        logging.info(f"[Hugging Face] Successfully pushed CSV to repository '{repo_name}'.")
    except Exception as ex:
        st.error(f"Error saving to repository: {ex}")
        logging.error(f"[Hugging Face] Error during push: {ex}")

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "sequence_id" not in st.session_state:
    st.session_state["sequence_id"] = 0

# This holds our current action label, if the user wants to classify frames
if "current_action" not in st.session_state:
    st.session_state["current_action"] = "Unlabeled_Action"

# -----------------------------------------------------------------------------
# CSV File Setup
# -----------------------------------------------------------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state["csv_file"] = os.path.join(csv_folder, f"all_actions_recorded_{session_start_str}.csv")

csv_file = st.session_state["csv_file"]

# Define your CSV header
mp_holistic = mp.solutions.holistic
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468

pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
angle_names_base = [
    'thumb_mcp', 'thumb_ip',
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip',
    'ring_mcp', 'ring_pip', 'ring_dip',
    'little_mcp', 'little_pip', 'little_dip'
]
left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]

HEADER = (
    ['class', 'sequence_id']
    + pose_landmarks
    + left_hand_landmarks
    + left_hand_angle_names
    + right_hand_landmarks
    + right_hand_angle_names
    + face_landmarks
)

def initialize_csv(file_name, header):
    """
    Initialize the CSV file with the header row if it doesn't exist.
    """
    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)

# Ensure CSV is initialized
initialize_csv(csv_file, HEADER)

# -----------------------------------------------------------------------------
# Load the MediaPipe Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_mediapipe_model():
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def validate_frame_data(frame_data):
    required_keys = [
        "pose_data",
        "left_hand_data",
        "left_hand_angles_data",
        "right_hand_data",
        "right_hand_angles_data",
        "face_data"
    ]
    if not isinstance(frame_data, dict):
        return False, "frame_data is not a dictionary"
    for key in required_keys:
        if key not in frame_data or not frame_data[key]:
            return False, key
    return True, None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def hand_angles(hand_landmarks):
    if (not hand_landmarks) or all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)
    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i): return [h[i][0], h[i][1]]

    # Thumb angles
    thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
    thumb_ip = calculate_angle(pt(2), pt(3), pt(4))
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

def process_frame(frame_bgr):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Optional: draw detected landmarks on annotated_image
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
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    face_flat = [val for landmark in face_data for val in landmark]

    return (
        pose_flat +
        left_hand_flat +
        left_hand_angles_data +
        right_hand_flat +
        right_hand_angles_data +
        face_flat
    )

# -----------------------------------------------------------------------------
# WebRTC Callback
# -----------------------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
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

    frame_data = {
        "pose_data": pose_data,
        "left_hand_data": left_hand_data,
        "left_hand_angles_data": left_hand_angles_data,
        "right_hand_data": right_hand_data,
        "right_hand_angles_data": right_hand_angles_data,
        "face_data": face_data,
    }

    logging.debug(
        f"[Callback] Enqueuing frame. Pose sample={pose_data[0] if pose_data else 'None'}"
    )

    frame_queue.put(frame_data)
    logging.debug(f"[Callback] Queue size: {frame_queue.qsize()}")

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1, 2])

# Left Column: Action Input
with left_col:
    st.header("Controls")
    action_word_input = st.text_input("Enter the meaning of this action, e.g. 'I'm hungry'")
    if st.button("Confirm Action") and action_word_input:
        # Clean the action name for CSV or internal usage
        sanitized_action = re.sub(r'[^a-zA-Z0-9_]', '_', action_word_input.strip())
        st.session_state["current_action"] = sanitized_action
        st.success(f"Action confirmed: {sanitized_action}")

# Right Column: Video
FRAME_WINDOW = right_col.image([])
status_bar = right_col.empty()

# Launch WebRTC streamer if desired
st.info("Activate Webcam to Start Capturing Frames")
webrtc_streamer(
    key="record-actions",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

# -----------------------------------------------------------------------------
# Single Button: Capture from Queue + Save to CSV
# -----------------------------------------------------------------------------
if st.button("Save to CSV"):
    logging.debug("[Save to CSV] Button clicked. Starting to process frames.")
    all_rows = []

    # 1) Pull all frames from the queue
    frame_count = frame_queue.qsize()
    logging.debug(f"[Save to CSV] Dequeuing {frame_count} frames from the queue.")

    for _ in range(frame_count):
        try:
            frame_data = frame_queue.get()
            logging.debug(f"[Save to CSV] Dequeued frame_data with keys: {list(frame_data.keys())}")

            # Validate
            is_valid, missing_key = validate_frame_data(frame_data)
            if not is_valid:
                logging.warning(f"[Save to CSV] Invalid frame_data. Missing key: {missing_key}")
                continue

            # Flatten
            row_data = flatten_landmarks(
                frame_data["pose_data"],
                frame_data["left_hand_data"],
                frame_data["left_hand_angles_data"],
                frame_data["right_hand_data"],
                frame_data["right_hand_angles_data"],
                frame_data["face_data"]
            )

            # Label with current_action
            action_label = st.session_state.get("current_action", "UnlabeledAction")
            st.session_state["sequence_id"] += 1
            row = [action_label, st.session_state["sequence_id"]] + row_data
            all_rows.append(row)

        except Exception as e:
            logging.error(f"[Save to CSV] Error processing frame_data: {e}")
            continue

    # 2) Write to CSV
    if all_rows:
        try:
            with open(csv_file, mode="a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(all_rows)
            st.success(f"Saved {len(all_rows)} rows to CSV: {csv_file}")
            logging.info(f"[Save to CSV] Successfully wrote {len(all_rows)} rows to CSV.")

            # Optional: push to Hugging Face
            try:
                df = pd.read_csv(csv_file)
                if df.empty:
                    logging.warning("[Hugging Face] CSV is empty; skipping push.")
                else:
                    save_csv_to_huggingface(csv_file)
            except Exception as e:
                st.error(f"Error reading CSV for repository push: {e}")
                logging.error(f"[Hugging Face] Push error: {e}")

        except Exception as e:
            st.error(f"Error writing to CSV: {e}")
            logging.error(f"[Save to CSV] CSV write failure: {e}")
    else:
        st.warning("No rows to write to CSV.")
        logging.warning("[Save to CSV] No rows were written to CSV because 'all_rows' is empty.")

# -----------------------------------------------------------------------------
# Display the Current CSV
# -----------------------------------------------------------------------------
st.header("Recorded Actions Summary (Current CSV)")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if not df.empty:
        unique_actions = df['class'].unique()
        st.write(f"Unique actions: {unique_actions}")

        st.subheader("Full CSV Data")
        df.reset_index(drop=True, inplace=True)
        st.dataframe(df)
    else:
        st.info("CSV is initialized but has no data rows yet.")
else:
    st.info("No CSV file found yet.")
