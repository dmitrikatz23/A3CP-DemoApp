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

# Additional imports
import csv
import time
import pandas as pd
import os
from datetime import datetime

# For import from upper folder
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# Logging setup
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ---------------------------
# Landmark / Angle Structure
# ---------------------------
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

pose_landmarks = [
    f'pose_{axis}{i}' for i in range(1, num_pose_landmarks + 1) for axis in ['x', 'y', 'v']
]
left_hand_landmarks = [
    f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand + 1) for axis in ['x', 'y', 'v']
]
right_hand_landmarks = [
    f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand + 1) for axis in ['x', 'y', 'v']
]
face_landmarks = [
    f'face_{axis}{i}' for i in range(1, num_face_landmarks + 1) for axis in ['x', 'y', 'v']
]

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

@st.cache_resource
def load_mediapipe_model():
    """
    Load the MediaPipe Holistic model for optimized video processing.
    """
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_model()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_frame(bgr_image):
    """
    Process a BGR image using MediaPipe Holistic, returning:
    (annotated_image, pose_data, left_hand_data, left_hand_angles, right_hand_data, right_hand_angles, face_data)
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(rgb_image)
    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Draw all landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Pose data
    if results.pose_landmarks:
        pose_data = [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
    else:
        pose_data = [[0, 0, 0]] * num_pose_landmarks

    # Hand data
    def extract_hand_data(hand_landmarks):
        if hand_landmarks:
            return [[lm.x, lm.y, lm.visibility] for lm in hand_landmarks.landmark]
        else:
            return [[0, 0, 0]] * num_hand_landmarks_per_hand

    left_hand_data = extract_hand_data(results.left_hand_landmarks)
    right_hand_data = extract_hand_data(results.right_hand_landmarks)

    # Face data
    if results.face_landmarks:
        face_data = [[lm.x, lm.y, lm.visibility] for lm in results.face_landmarks.landmark]
    else:
        face_data = [[0, 0, 0]] * num_face_landmarks

    # Calculate angles
    def hand_angles(hand_landmarks):
        if all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
            return [0] * len(angle_names_base)

        h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
        def pt(i): return [h[i][0], h[i][1]]

        # Example calculations
        thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
        thumb_ip  = calculate_angle(pt(2), pt(3), pt(4))

        index_mcp = calculate_angle(pt(0), pt(5), pt(6))
        index_pip = calculate_angle(pt(5), pt(6), pt(7))
        index_dip = calculate_angle(pt(6), pt(7), pt(8))

        middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
        middle_pip = calculate_angle(pt(9), pt(10), pt(11))
        middle_dip = calculate_angle(pt(10), pt(11), pt(12))

        ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
        ring_pip = calculate_angle(pt(13), pt(14), pt(15))
        ring_dip = calculate_angle(pt(14), pt(15), pt(16))

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

    left_angles = hand_angles(left_hand_data)
    right_angles = hand_angles(right_hand_data)

    return annotated_image, pose_data, left_hand_data, left_angles, right_hand_data, right_angles, face_data

def flatten_landmarks(
    pose_data,
    left_hand_data,
    left_hand_angles,
    right_hand_data,
    right_hand_angles,
    face_data
):
    """Flatten all landmark data + angles into a single 1D list."""
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    left_angles_flat = left_hand_angles
    right_angles_flat = right_hand_angles
    face_flat = [val for landmark in face_data for val in landmark]

    return (
        pose_flat
        + left_hand_flat
        + left_angles_flat
        + right_hand_flat
        + right_angles_flat
        + face_flat
    )

# ---------------------------
# CSV Initialization
# ---------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state["csv_file"] = os.path.join(csv_folder, f"all_actions_recorded_{session_start_str}.csv")

csv_file = st.session_state["csv_file"]

@st.cache_data
def initialize_csv(file_name, header):
    with open(file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
    return True

if "csv_initialized" not in st.session_state:
    st.session_state["csv_initialized"] = initialize_csv(csv_file, header)

# ---------------------------
# Streamlit State
# ---------------------------
st.title("Record and Action")

# Keep track of user-defined states
if "actions" not in st.session_state:
    st.session_state["actions"] = {}
if "action_confirmed" not in st.session_state:
    st.session_state["action_confirmed"] = False
if "sequence_id" not in st.session_state:
    st.session_state["sequence_id"] = 0
if "record_in_progress" not in st.session_state:
    st.session_state["record_in_progress"] = False
if "record_start_time" not in st.session_state:
    st.session_state["record_start_time"] = 0
if "frames" not in st.session_state:
    st.session_state["frames"] = []  # store landmark data, etc.
if "record_duration" not in st.session_state:
    st.session_state["record_duration"] = 10  # default

col_controls, col_stream = st.columns([1, 2])

with col_controls:
    st.header("Controls")
    action_word = st.text_input("Enter the intended meaning for the action (e.g. I'm hungry)")
    record_time = st.number_input("Recording Time (seconds)", min_value=1, max_value=300, value=10)

    if st.button("Confirm Action") and action_word:
        st.session_state["actions"][action_word] = None
        st.session_state["action_confirmed"] = True
        st.session_state["record_duration"] = record_time
        st.success(f"Action '{action_word}' confirmed for {record_time} seconds!")

    # Button to start recording from WebRTC frames
    if st.session_state["action_confirmed"]:
        if not st.session_state["record_in_progress"]:
            if st.button("Start Recording"):
                st.session_state["record_in_progress"] = True
                st.session_state["record_start_time"] = time.time()
                st.session_state["frames"] = []
                st.info("Recording started...")
        else:
            st.warning("Recording is already in progress...")

with col_stream:
    st.subheader("Live Stream (Appears After Action Confirmation)")

# This callback collects frames from the live stream if "record_in_progress" is True
def webrtc_video_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame_bgr = frame.to_ndarray(format="bgr24")
    annotated_image, p_data, lh_data, lh_angles, rh_data, rh_angles, f_data = process_frame(frame_bgr)

    # If we are recording, collect the data
    if st.session_state["record_in_progress"]:
        elapsed = time.time() - st.session_state["record_start_time"]

        # Append frame data for potential CSV storage
        st.session_state["frames"].append((p_data, lh_data, lh_angles, rh_data, rh_angles, f_data))

        if elapsed >= st.session_state["record_duration"]:
            # Stop recording automatically
            st.session_state["record_in_progress"] = False
            st.success(f"Recording ended after {elapsed:.1f} seconds.")
            # Process frames if desired
            st.info(f"Captured {len(st.session_state['frames'])} frames. Ready for keyframe analysis or CSV export.")

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# Display WebRTC only if action confirmed
if st.session_state["action_confirmed"]:
    webrtc_streamer(
        key="record-actions",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=webrtc_video_callback,
        async_processing=True,
    )
else:
    st.info("Confirm an action to enable the live stream.")

# ---------------------------
# Display / Process Recorded Actions
# ---------------------------
st.header("Recorded Actions")
if st.session_state["frames"]:
    # Example code to flatten data for CSV / keyframe detection
    # (when the user is done recording).
    all_landmarks = []
    for f in st.session_state["frames"]:
        p_data, lh_data, lh_angles, rh_data, rh_angles, f_data = f
        row_data = flatten_landmarks(p_data, lh_data, lh_angles, rh_data, rh_angles, f_data)
        all_landmarks.append(row_data)

    # Keyframe detection, CSV writing, or other logic can go here
    st.write(f"Stored {len(all_landmarks)} frames of landmark data. (You can add keyframe detection or CSV saves here.)")
else:
    st.info("No frames recorded yet. Confirm an action and start recording.")
