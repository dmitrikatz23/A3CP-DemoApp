import mediapipe as mp
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

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

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
        min_tracking_confidence=0.5
    )

holistic_model = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils  # Drawing helper

# -----------------------------
# Initialize Session State for Debugging Holistic
# -----------------------------
if "holistic_status" not in st.session_state:
    st.session_state["holistic_status"] = {"pose": False, "left_hand": False, "right_hand": False}

# -----------------------------
# Helper Function: Extract Landmarks from Frame
# -----------------------------
def extract_landmarks(image):
    """Extract holistic landmarks from an image and store detection status in session state."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)

    # Store detection status in session state (no direct Streamlit calls here)
    st.session_state["holistic_status"] = {
        "pose": bool(results.pose_landmarks),
        "left_hand": bool(results.left_hand_landmarks),
        "right_hand": bool(results.right_hand_landmarks),
    }

    # If no landmarks detected, return None
    if not results.pose_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks:
        return None, results  

    landmarks = []
    
    def append_landmarks(landmark_list, count):
        if landmark_list:
            for lm in landmark_list.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0, 0.0, 0.0] * count)  # Fill missing landmarks

    append_landmarks(results.pose_landmarks, 33)  # Pose: 33 points
    append_landmarks(results.left_hand_landmarks, 21)  # Left Hand: 21 points
    append_landmarks(results.right_hand_landmarks, 21)  # Right Hand: 21 points

    return np.array(landmarks, dtype=np.float32), results

# -----------------------------
# WebRTC Frame Callback for Inference
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

            # Store prediction in session state
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
# Sidebar: Model Selection and Holistic Debugging
# -----------------------------
with st.sidebar:
    st.subheader("Holistic Detection Status")
    st.text(f"Pose Landmarks: {'✅' if st.session_state['holistic_status']['pose'] else '❌'}")
    st.text(f"Left Hand: {'✅' if st.session_state['holistic_status']['left_hand'] else '❌'}")
    st.text(f"Right Hand: {'✅' if st.session_state['holistic_status']['right_hand'] else '❌'}")

# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("WebRTC Stream")
    webrtc_streamer(
        key="tryit-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

with right_col:
    st.header("Predicted Gesture")
    st.write(f"**Prediction:** {st.session_state.get('tryit_predicted_text', 'Waiting for input...')}")  
