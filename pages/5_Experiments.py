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
from queue import Queue

sys.path.append(str(Path(__file__).resolve().parent.parent))
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
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Initialize default values for all landmarks
    pose_data = [[0.0, 0.0, 0.0] for _ in range(num_pose_landmarks)]
    left_hand_data = [[0.0, 0.0, 0.0] for _ in range(num_hand_landmarks_per_hand)]
    right_hand_data = [[0.0, 0.0, 0.0] for _ in range(num_hand_landmarks_per_hand)]
    face_data = [[0.0, 0.0, 0.0] for _ in range(num_face_landmarks)]

    # Only populate data if landmarks are detected
    if results.pose_landmarks:
        pose_data = [
            [lm.x, lm.y, lm.visibility] 
            for lm in results.pose_landmarks.landmark
        ]
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        left_hand_data = [
            [lm.x, lm.y, lm.visibility] 
            for lm in results.left_hand_landmarks.landmark
        ]
        mp_drawing.draw_landmarks(
            annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        right_hand_data = [
            [lm.x, lm.y, lm.visibility] 
            for lm in results.right_hand_landmarks.landmark
        ]
        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    if results.face_landmarks:
        face_data = [
            [lm.x, lm.y, lm.visibility] 
            for lm in results.face_landmarks.landmark
        ]
        mp_drawing.draw_landmarks(
            annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
        )

    # Compute joint angles
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

# -----------------------------------
# Queue Initialization
# -----------------------------------
if "data_queue" not in st.session_state:
    st.session_state.data_queue = Queue()

# Function to add data to the queue
def add_to_queue(data):
    """
    Adds data to the session state queue.
    """
    st.session_state.data_queue.put(data)
    st.write(f"Added to queue: {data}")
    st.write(f"Queue size: {st.session_state.data_queue.qsize()}")

# Function to write data from the queue to the CSV
def write_queue_to_csv(file_path):
    """
    Writes all data in the session state queue to the CSV file.
    """
    if st.session_state.data_queue.empty():
        st.warning("No data in queue to write.")
        return

    rows = []
    while not st.session_state.data_queue.empty():
        rows.append(st.session_state.data_queue.get())

    try:
        with open(file_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)
        st.success(f"Written {len(rows)} rows from queue to {file_path}")
    except Exception as e:
        st.error(f"Error writing to CSV: {e}")

# -----------------------------------
# WebRTC Video Callback
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    WebRTC callback that uses MediaPipe Holistic to process frames in real-time.
    Returns an annotated frame.
    """
    input_bgr = frame.to_ndarray(format="bgr24")
    (
        annotated_image,
        pose_data,
        left_hand_data,
        left_hand_angles,
        right_hand_data,
        right_hand_angles,
        face_data
    ) = process_frame(input_bgr)

    # Flatten landmarks and add to queue
    flattened_data = flatten_landmarks(
        pose_data,
        left_hand_data,
        left_hand_angles,
        right_hand_data,
        right_hand_angles,
        face_data
    )
    add_to_queue(flattened_data)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

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

# -----------------------------------
# CSV Setup
# -----------------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# If a CSV file hasn't been set in session state, create one with a timestamped name
if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state["csv_file"] = os.path.join(csv_folder, f"my_actions_{session_start_str}.csv")

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
        st.session_state['actions'][action_word] = None
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

# -----------------------------------
# Right/Main Area: Recorded Actions
# -----------------------------------
st.header("Recorded Actions")

# Write queue to CSV when the button is clicked
if st.button("Write Queue to CSV"):
    write_queue_to_csv(csv_file)

# Display the CSV contents
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.dataframe(df)
else:
    st.info("No CSV file exists yet.")