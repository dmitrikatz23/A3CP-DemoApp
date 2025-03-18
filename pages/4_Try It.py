
#model selector and mediapipe working...no holistic

import mediapipe as mp
from pathlib import Path
from typing import List, NamedTuple
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from huggingface_hub import hf_hub_download, HfApi
import os
import re
import pandas as pd
import os
from collections import deque
import threading
import sys
from huggingface_hub import Repository, HfApi

sys.path.append(str(Path(__file__).resolve().parent.parent))
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

# -----------------------------------
# MediaPipe Initialization & Landmark Constants
# -----------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Number of landmarks in various MediaPipe models
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468


# -----------------------------
# Hugging Face Setup
# -----------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is set.")
    st.stop()

hf_api = HfApi()
model_repo_name = "dk23/A3CP_models"
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# -----------------------------
# Load MediaPipe Holistic Model (Cached)
# -----------------------------
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
    Returns:
        annotated_image: The original frame annotated with landmarks.
        pose_data: 2D + visibility for each pose landmark.
        left_hand_data: 2D + visibility for each landmark in the left hand.
        left_hand_angles_data: The angles computed for the left hand joints.
        right_hand_data: 2D + visibility for each landmark in the right hand.
        right_hand_angles_data: The angles computed for the right hand joints.
        face_data: 2D + visibility for each face landmark.
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

    # Extract data
    def extract_data(landmarks, count):
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count

    pose_data = extract_data(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = extract_data(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = extract_data(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = extract_data(results.face_landmarks, num_face_landmarks)

    # Compute joint angles for hands
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
    Expected input: NxM array (N frames, M features).
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
    `landmarks` is a NxM array representing frames (N) by flattened features (M).
    """
    velocities = calculate_velocity(landmarks)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i + 1)  # +1 offset because acceleration index starts at 1
    return keyframes

# -----------------------------
# WebRTC Frame Callback 
# -----------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process frame for real-time gesture prediction."""
    img_bgr = frame.to_ndarray(format="bgr24")

    # Extract landmarks using MediaPipe Holistic
    landmarks, results = extract_landmarks(img_bgr)

    if "tryit_model" in st.session_state and "tryit_encoder" in st.session_state:
        model = st.session_state["tryit_model"]
        encoder = st.session_state["tryit_encoder"]

        if landmarks is not None:
            # Preprocess input
            landmarks = np.expand_dims(landmarks, axis=0)
            landmarks = pad_sequences([landmarks], maxlen=100, padding='post', dtype='float32', value=-1.0)

            # Predict gesture
            predictions = model.predict(landmarks)
            predicted_label = np.argmax(predictions, axis=1)
            predicted_text = encoder.inverse_transform(predicted_label)[0]

            # Store in session state for UI display
            st.session_state["tryit_predicted_text"] = predicted_text
            cv2.putText(img_bgr, f"Prediction: {predicted_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            st.session_state["tryit_predicted_text"] = "No Gesture Detected"

    # Draw detected landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img_bgr, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -----------------------------
# Sidebar: Model Selection
# -----------------------------
@st.cache_data
def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs from Hugging Face."""
    repo_files = hf_api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]

    pairs = {}
    for mf in model_files:
        ts = mf[len("LSTM_model_"):-3]  # Extract timestamp
        pairs.setdefault(ts, {})["model"] = mf
    for ef in encoder_files:
        ts = ef[len("label_encoder_"):-4]  # Extract timestamp
        pairs.setdefault(ts, {})["encoder"] = ef

    valid_pairs = [(ts, items["model"], items["encoder"]) for ts, items in pairs.items() if "model" in items and "encoder" in items]
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

with st.sidebar:
    st.subheader("Select a Model/Encoder Pair")
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found.")
    else:
        pair_options = {f"{ts} | Model: {mf} | Encoder: {ef}": (mf, ef) for ts, mf, ef in model_encoder_pairs}
        selected_label = st.selectbox("Choose a matched pair:", list(pair_options.keys()))

        if selected_label:
            chosen_model, chosen_encoder = pair_options[selected_label]
            st.write("**Selected Model:**", chosen_model)
            st.write("**Selected Encoder:**", chosen_encoder)

        if st.button("Confirm Model") and selected_label:
            st.session_state["tryit_selected_pair"] = pair_options[selected_label]
            st.session_state["tryit_model_confirmed"] = True

            model_path = os.path.join(LOCAL_MODEL_DIR, chosen_model)
            encoder_path = os.path.join(LOCAL_MODEL_DIR, chosen_encoder)

            if not os.path.exists(model_path):
                hf_hub_download(model_repo_name, chosen_model, local_dir=LOCAL_MODEL_DIR, repo_type="model", token=hf_token)
            if not os.path.exists(encoder_path):
                hf_hub_download(model_repo_name, chosen_encoder, local_dir=LOCAL_MODEL_DIR, repo_type="model", token=hf_token)

            # Load Model & Encoder
            st.session_state["tryit_model"] = tf.keras.models.load_model(model_path)
            st.session_state["tryit_encoder"] = joblib.load(encoder_path)
            st.success("Model and encoder loaded successfully!")

# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("WebRTC Stream")
    if st.session_state.get("tryit_model_confirmed"):
        webrtc_streamer(
            key="tryit-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

# with right_col:
#     st.header("Predicted Gesture")
#     st.write(f"**Prediction:** {st.session_state.get('tryit_predicted_text', 'Waiting for input...')}")  
