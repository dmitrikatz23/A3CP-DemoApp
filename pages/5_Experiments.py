# Updated Code for Proper Queue Usage

import logging
from queue import Queue
from pathlib import Path
from typing import List, NamedTuple
import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit import session_state
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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state queue
if "data_queue" not in st.session_state:
    st.session_state.data_queue = Queue()


# Debug helper function
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
            return False, key  # Missing or empty key

    return True, None

logging.getLogger("twilio").setLevel(logging.WARNING)

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

    return (
        annotated_image,
        pose_data,
        left_hand_data,
        right_hand_data,
        face_data
    )

# -----------------------------------
# WebRTC Video Callback
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    input_bgr = frame.to_ndarray(format="bgr24")

    (annotated_image,
     pose_data,
     left_hand_data,
     right_hand_data,
     face_data) = process_frame(input_bgr)

    frame_data = {
        "pose_data": pose_data,
        "left_hand_data": left_hand_data,
        "right_hand_data": right_hand_data,
        "face_data": face_data,
    }

    if not st.session_state.data_queue.full():
        st.session_state.data_queue.put(frame_data)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("Real-time Landmark Recording")

# Controls
if st.button("Start Stream"):
    webrtc_streamer(
        key="example-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
    )

if st.button("Save to CSV"):
    if not st.session_state.data_queue.empty():
        rows = []
        while not st.session_state.data_queue.empty():
            frame_data = st.session_state.data_queue.get()
            flat_data = (
                frame_data["pose_data"] +
                frame_data["left_hand_data"] +
                frame_data["right_hand_data"] +
                frame_data["face_data"]
            )
            rows.append(flat_data)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        st.success(f"Saved {len(rows)} rows to {csv_file}")
    else:
        st.warning("No data to save.")
