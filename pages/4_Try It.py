# -----------------------------------
# Imports and Logging Setup
# -----------------------------------
import os
import sys
import logging
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import av
import streamlit as st
import joblib
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model
import mediapipe as mp

# Optional: Append additional module paths if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

# -----------------------------------
# ICE Servers Function
# -----------------------------------
def get_ice_servers():
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:relay1.expressturn.com:3478"], "username": "user", "credential": "pass"}
    ]

# -----------------------------------
# Logging Configuration
# -----------------------------------
DEBUG_MODE = False  # Set to True for debugging

def debug_log(message):
    if DEBUG_MODE:
        logging.debug(message)

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app_debug.log", mode='w')]
)
logger = logging.getLogger(__name__)
logger.info("🚀 Logging initialized!")

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(layout="wide", page_title="Gesture Recognition")
st.title("Gesture Recognition System")

# -----------------------------------
# Hugging Face Token and API Setup
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()
hf_api = HfApi(token=hf_token)

# Define repository details for model/encoder
MODEL_REPO_NAME = "dk23/A3CP_models"
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# -----------------------------------
# Session State Initialization
# -----------------------------------
if "landmark_queue" not in st.session_state:
    st.session_state.landmark_queue = deque(maxlen=1000)
if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "recognized_action" not in st.session_state:
    st.session_state.recognized_action = "Waiting for prediction..."
if "action_confirmed" not in st.session_state:
    # For this example, we assume the action is confirmed by default.
    st.session_state.action_confirmed = True
if "active_streamer_key" not in st.session_state:
    st.session_state.active_streamer_key = "gesture_streamer"  # Unique static key for this session
if "streamer_running" not in st.session_state:
    st.session_state.streamer_running = False
if "action_word" not in st.session_state:
    st.session_state.action_word = "Gesture"

landmark_queue = st.session_state.landmark_queue
lock = st.session_state.lock

# -----------------------------------
# MediaPipe Initialization & Constants
# -----------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# -----------------------------------
# Video Processing Functions
# -----------------------------------
def process_frame(frame):
    """
    Process a single frame using MediaPipe Holistic.
    Draw landmarks on the frame and return the annotated image.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks if detected
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    return annotated_image

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Callback function for WebRTC video streaming.
    Processes the frame and returns the annotated frame.
    """
    input_bgr = frame.to_ndarray(format="bgr24")
    annotated_image = process_frame(input_bgr)
    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -----------------------------------
# Model & Encoder Selection UI (Sidebar)
# -----------------------------------
st.sidebar.header("Model & Encoder Selection")

@st.cache_data
def get_model_files():
    """Retrieve available models and encoders from Hugging Face."""
    repo_files = hf_api.list_repo_files(MODEL_REPO_NAME, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    return model_files, encoder_files

model_files, encoder_files = get_model_files()
selected_model_file = st.sidebar.selectbox("Select Model File", model_files)
selected_encoder_file = st.sidebar.selectbox("Select Encoder File", encoder_files)

if st.sidebar.button("Load Model & Encoder"):
    with st.spinner("Loading model and encoder..."):
        try:
            model_path = hf_hub_download(MODEL_REPO_NAME, selected_model_file, repo_type="model", local_dir=LOCAL_MODEL_DIR, token=hf_token)
            model = load_model(model_path)
            encoder_path = hf_hub_download(MODEL_REPO_NAME, selected_encoder_file, repo_type="model", local_dir=LOCAL_MODEL_DIR, token=hf_token)
            label_encoder = joblib.load(encoder_path)
            st.session_state.model = model
            st.session_state.label_encoder = label_encoder
            st.session_state.model_loaded = True
            st.sidebar.success("Model and encoder loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model/encoder: {e}")

# -----------------------------------
# Working WebRTC Streamer Code (from Record_Actions.py)
# -----------------------------------
if st.session_state.get('action_confirmed', False):
    streamer_key = st.session_state['active_streamer_key']
    st.info(f"Streaming activated! Perform the action: {st.session_state.get('action_word', 'your action')}")

    # Launch the WebRTC streamer using the working snippet
    webrtc_ctx = webrtc_streamer(
        key=streamer_key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

    # Update streamer_running flag based on streamer state
    if webrtc_ctx.state.playing:
        st.session_state['streamer_running'] = True
    else:
        if st.session_state.get('streamer_running', False):
            st.session_state['streamer_running'] = False
            # Optionally, snapshot the landmark queue here if needed:
            st.session_state["landmark_queue_snapshot"] = list(landmark_queue)
            debug_log(f"Snapshot taken with {len(st.session_state['landmark_queue_snapshot'])} frames.")
            st.success("Streaming has stopped. You can now save keyframes.")
else:
    st.info("Please confirm action to start streaming.")

# -----------------------------------
# Gesture Prediction Display
# -----------------------------------
# Process prediction only if a model is loaded and the stream is active.
if st.session_state.get("model_loaded", False) and 'webrtc_ctx' in locals() and webrtc_ctx.state.playing:
    st.subheader("Recognized Action:")
    try:
        # Here, row_data should be constructed from actual processed frame data.
        # For demonstration purposes, we use a placeholder.
        row_data = np.zeros((1, 1000))  # Replace with actual flattened landmark data
        prediction = st.session_state.model.predict(row_data)
        predicted_index = np.argmax(prediction)
        predicted_class = st.session_state.label_encoder.inverse_transform([predicted_index])[0]
        st.session_state.recognized_action = predicted_class
    except Exception as e:
        st.session_state.recognized_action = f"Error: {e}"
    st.write(st.session_state.get("recognized_action", "Waiting for prediction..."))
else:
    st.info("Load a model and encoder from the sidebar to enable gesture prediction.")
