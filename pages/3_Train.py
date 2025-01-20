import logging
import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import os
import csv
from datetime import datetime

# Logging setup
logger = logging.getLogger(__name__)

# Streamlit setup
st.set_page_config(layout="wide")

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Constants
CSV_FOLDER = "csv"
if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468

@st.cache_resource
def load_mediapipe_model():
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

holistic_model = load_mediapipe_model()

# Initialize session state
if "actions" not in st.session_state:
    st.session_state["actions"] = {}
if "recording" not in st.session_state:
    st.session_state["recording"] = False
if "current_action" not in st.session_state:
    st.session_state["current_action"] = None

# CSV Header
LANDMARK_HEADER = (
    ["class", "sequence_id"] +
    [f"pose_{i}_{axis}" for i in range(NUM_POSE_LANDMARKS) for axis in ["x", "y", "z", "v"]] +
    [f"left_hand_{i}_{axis}" for i in range(NUM_HAND_LANDMARKS) for axis in ["x", "y", "z"]] +
    [f"right_hand_{i}_{axis}" for i in range(NUM_HAND_LANDMARKS) for axis in ["x", "y", "z"]] +
    [f"face_{i}_{axis}" for i in range(NUM_FACE_LANDMARKS) for axis in ["x", "y", "z"]]
)

CSV_FILE = os.path.join(CSV_FOLDER, f"actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

@st.cache_data
def initialize_csv(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LANDMARK_HEADER)

initialize_csv(CSV_FILE)

# Process a frame using MediaPipe Holistic
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(rgb_frame)
    return results

# Extract landmarks
def extract_landmarks(results):
    def get_landmark_data(landmarks, num_landmarks, extra_dims=3):
        if landmarks:
            return [list(landmark) for landmark in landmarks.landmark]
        else:
            return [[0] * extra_dims] * num_landmarks

    pose_landmarks = get_landmark_data(results.pose_landmarks, NUM_POSE_LANDMARKS, 4)
    left_hand_landmarks = get_landmark_data(results.left_hand_landmarks, NUM_HAND_LANDMARKS)
    right_hand_landmarks = get_landmark_data(results.right_hand_landmarks, NUM_HAND_LANDMARKS)
    face_landmarks = get_landmark_data(results.face_landmarks, NUM_FACE_LANDMARKS)

    return pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks

# Flatten landmarks
def flatten_landmarks(action_name, sequence_id, pose, left_hand, right_hand, face):
    pose_flat = [value for sublist in pose for value in sublist]
    left_hand_flat = [value for sublist in left_hand for value in sublist]
    right_hand_flat = [value for sublist in right_hand for value in sublist]
    face_flat = [value for sublist in face for value in sublist]

    return [action_name, sequence_id] + pose_flat + left_hand_flat + right_hand_flat + face_flat

# WebRTC video callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    action_name = st.session_state["current_action"]
    recording = st.session_state["recording"]

    # Convert frame and process with MediaPipe
    frame_array = frame.to_ndarray(format="bgr24")
    results = process_frame(frame_array)

    # Annotate the frame with landmarks
    annotated_frame = frame_array.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    # Save landmarks to session state if recording
    if recording and action_name:
        pose, left_hand, right_hand, face = extract_landmarks(results)
        sequence_id = len(st.session_state["actions"].get(action_name, []))
        flat_landmarks = flatten_landmarks(action_name, sequence_id, pose, left_hand, right_hand, face)

        if action_name not in st.session_state["actions"]:
            st.session_state["actions"][action_name] = []

        st.session_state["actions"][action_name].append(flat_landmarks)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamlit UI
st.title("Action Recording with MediaPipe Holistic")

action_name = st.text_input("Enter the action name (e.g., 'wave')")

if not st.session_state["recording"]:
    if st.button("Start Recording") and action_name:
        st.session_state["current_action"] = action_name
        st.session_state["recording"] = True
        st.success(f"Recording started for action: '{action_name}'")
else:
    if st.button("Stop Recording"):
        st.session_state["recording"] = False
        st.success(f"Recording stopped for action: '{st.session_state['current_action']}'")

        # Save recorded data to CSV
        all_landmarks = st.session_state["actions"].get(st.session_state["current_action"], [])
        if all_landmarks:
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(all_landmarks)
            st.success(f"Data saved to {CSV_FILE}")
            st.session_state["actions"][st.session_state["current_action"]] = []

# Display WebRTC streamer
webrtc_streamer(
    key="holistic",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)
