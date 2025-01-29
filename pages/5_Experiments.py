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
# Initialization
# -----------------------------------
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")

# MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Landmark counts
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468

# Angle names setup
angle_names_base = [
    'thumb_mcp', 'thumb_ip', 'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'ring_mcp', 'ring_pip',
    'ring_dip', 'little_mcp', 'little_pip', 'little_dip'
]
left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]

# CSV header components
pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]

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
# Core Processing Functions
# -----------------------------------
@st.cache_resource
def load_mediapipe_model():
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
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def hand_angles(hand_landmarks):
    if not hand_landmarks or all(p[0]+p[1]+p[2] == 0 for p in hand_landmarks):
        return [0]*len(angle_names_base)
    
    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    pt = lambda i: [h[i][0], h[i][1]]
    
    return [
        calculate_angle(pt(1), pt(2), pt(3)),    # thumb_mcp
        calculate_angle(pt(2), pt(3), pt(4)),    # thumb_ip
        calculate_angle(pt(0), pt(5), pt(6)),    # index_mcp
        calculate_angle(pt(5), pt(6), pt(7)),    # index_pip
        calculate_angle(pt(6), pt(7), pt(8)),    # index_dip
        calculate_angle(pt(0), pt(9), pt(10)),   # middle_mcp
        calculate_angle(pt(9), pt(10), pt(11)),  # middle_pip
        calculate_angle(pt(10), pt(11), pt(12)), # middle_dip
        calculate_angle(pt(0), pt(13), pt(14)),  # ring_mcp
        calculate_angle(pt(13), pt(14), pt(15)), # ring_pip
        calculate_angle(pt(14), pt(15), pt(16)), # ring_dip
        calculate_angle(pt(0), pt(17), pt(18)),  # little_mcp
        calculate_angle(pt(17), pt(18), pt(19)), # little_pip
        calculate_angle(pt(18), pt(19), pt(20)), # little_dip
    ]

def process_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Initialize default values
    pose_data = [[0.0, 0.0, 0.0] for _ in range(num_pose_landmarks)]
    left_hand_data = [[0.0, 0.0, 0.0] for _ in range(num_hand_landmarks_per_hand)]
    right_hand_data = [[0.0, 0.0, 0.0] for _ in range(num_hand_landmarks_per_hand)]
    face_data = [[0.0, 0.0, 0.0] for _ in range(num_face_landmarks)]

    # Update with detected landmarks
    if results.pose_landmarks:
        pose_data = [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    if results.left_hand_landmarks:
        left_hand_data = [[lm.x, lm.y, lm.visibility] for lm in results.left_hand_landmarks.landmark]
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    if results.right_hand_landmarks:
        right_hand_data = [[lm.x, lm.y, lm.visibility] for lm in results.right_hand_landmarks.landmark]
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    if results.face_landmarks:
        face_data = [[lm.x, lm.y, lm.visibility] for lm in results.face_landmarks.landmark]
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    return (
        annotated_image,
        pose_data,
        left_hand_data,
        hand_angles(left_hand_data),
        right_hand_data,
        hand_angles(right_hand_data),
        face_data
    )

def flatten_landmarks(*args):
    return [
        val 
        for data in args 
        for sublist in (
            [item for landmark in data for item in landmark] 
            if isinstance(data[0], list) 
            else data
        ) 
        for val in sublist
    ]

# -----------------------------------
# Data Management
# -----------------------------------
if "data_queue" not in st.session_state:
    st.session_state.data_queue = Queue(maxsize=1000)

def initialize_session():
    if 'actions' not in st.session_state:
        st.session_state.actions = {}
    if 'sequence_id' not in st.session_state:
        st.session_state.sequence_id = 0
    if 'active_streamer_key' not in st.session_state:
        st.session_state.active_streamer_key = None
    if 'csv_file' not in st.session_state:
        csv_folder = Path("csv")
        csv_folder.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.csv_file = csv_folder / f"actions_{timestamp}.csv"
        with open(st.session_state.csv_file, 'w') as f:
            csv.writer(f).writerow(header)

initialize_session()

# -----------------------------------
# Video Processing
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    processed = process_frame(image)
    
    if st.session_state.get('action_confirmed', False):
        action_word = next(iter(st.session_state.actions))
        if st.session_state.actions[action_word] is None:
            st.session_state.actions[action_word] = []
        st.session_state.actions[action_word].append(processed[1:])  # Skip annotated image
    
    return av.VideoFrame.from_ndarray(processed[0], format="bgr24")

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("Action Recognition System")

left_col, right_col = st.columns([1, 2])
with left_col:
    st.header("Controls")
    action_word = st.text_input("Action label (e.g., 'Wave Hello')")
    
    if st.button("Start Recording") and action_word:
        sanitized = re.sub(r'[^\w_]', '_', action_word.strip())
        if st.session_state.get('active_streamer_key'):
            st.session_state.action_confirmed = False
            del st.session_state[st.session_state.active_streamer_key]
        
        st.session_state.actions[action_word] = []
        st.session_state.action_confirmed = True
        st.session_state.active_streamer_key = f"streamer_{sanitized}"
        st.success(f"Ready to record: {action_word}")

with right_col:
    if st.session_state.get('action_confirmed'):
        webrtc_streamer(
            key=st.session_state.active_streamer_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

# -----------------------------------
# Data Processing & Saving
# -----------------------------------
def process_recorded_actions():
    if not st.session_state.actions:
        return
    
    all_rows = []
    for action, frames in st.session_state.actions.items():
        if not frames or len(frames) < 2:
            continue
        
        flat_data = [
            flatten_landmarks(
                frame[0],  # pose_data
                frame[1],  # left_hand_data
                frame[2],  # left_angles
                frame[3],  # right_hand_data
                frame[4],  # right_angles
                frame[5]   # face_data
            )
            for frame in frames
        ]
        
        velocities = np.array([np.linalg.norm(flat_data[i]-flat_data[i-1]) 
                             for i in range(1, len(flat_data))])
        accelerations = np.abs(np.diff(velocities))
        
        keyframes = [
            i+1 for i, (v, a) in enumerate(zip(velocities, accelerations))
            if v > 0.1 or a > 0.1
        ]
        
        for kf in keyframes:
            if kf < len(frames):
                st.session_state.sequence_id += 1
                row = [action, st.session_state.sequence_id] + flat_data[kf]
                st.session_state.data_queue.put(row)

if st.button("Process & Save Actions"):
    process_recorded_actions()
    
    if st.session_state.data_queue.empty():
        st.warning("No data to save")
    else:
        rows = []
        while not st.session_state.data_queue.empty():
            rows.append(st.session_state.data_queue.get())
        
        with open(st.session_state.csv_file, 'a') as f:
            csv.writer(f).writerows(rows)
        st.success(f"Saved {len(rows)} sequences to {st.session_state.csv_file}")

if Path(st.session_state.csv_file).exists():
    st.subheader("Recorded Data Preview")
    df = pd.read_csv(st.session_state.csv_file)
    st.dataframe(df.head())
else:
    st.info("No recorded data yet")