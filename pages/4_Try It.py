import os
import sys
import logging
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
import numpy as np
import cv2
import av
import streamlit as st
import joblib
from huggingface_hub import HfApi, hf_hub_download
from streamlit_webrtc import webrtc_streamer, WebRtcMode, WebRtcStreamerContext
from tensorflow.keras.models import load_model
import mediapipe as mp

# -------------------------------
# Logging Configuration
# -------------------------------
DEBUG_MODE = False  # Set to True for debugging
def debug_log(message):
    if DEBUG_MODE:
        st.write(f"DEBUG: {message}")

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info("Logging initialized.")

# -------------------------------
# Session State Initialization
# -------------------------------
if "action_confirmed" not in st.session_state:
    st.session_state["action_confirmed"] = False  # Set to True after user confirms action
if "active_streamer_key" not in st.session_state:
    st.session_state["active_streamer_key"] = "gesture_streamer"  # Persistent key for the streamer
if "streamer_running" not in st.session_state:
    st.session_state["streamer_running"] = False
if "action_word" not in st.session_state:
    st.session_state["action_word"] = "Gesture"
if "landmark_queue" not in st.session_state:
    st.session_state["landmark_queue"] = deque(maxlen=1000)
if "landmark_queue_snapshot" not in st.session_state:
    st.session_state["landmark_queue_snapshot"] = []
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "recognized_action" not in st.session_state:
    st.session_state["recognized_action"] = "Waiting for prediction..."

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Gesture Recognition", layout="wide")
st.title("Gesture Recognition System")

# -------------------------------
# Sidebar: Model & Encoder Selection
# -------------------------------
st.sidebar.header("Model & Encoder Selection")
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.sidebar.error("Hugging Face token not found. Please add the 'Recorded_Datasets' secret.")
    st.stop()
hf_api = HfApi(token=hf_token)
MODEL_REPO_NAME = "dk23/A3CP_models"
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

@st.cache_data
def get_model_files():
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
            st.session_state["model"] = model
            st.session_state["label_encoder"] = label_encoder
            st.session_state["model_loaded"] = True
            st.sidebar.success("Model and encoder loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model/encoder: {e}")

# -------------------------------
# Action Confirmation UI
# -------------------------------
action_input = st.text_input("Enter action word (e.g., 'wave'):", value=st.session_state.get("action_word", "Gesture"))
if st.button("Confirm Action"):
    st.session_state["action_word"] = action_input
    st.session_state["action_confirmed"] = True
    st.session_state["active_streamer_key"] = "gesture_streamer"  # You can append a timestamp if desired
    st.success(f"Action '{action_input}' confirmed! Streaming will start now.")

# -------------------------------
# MediaPipe Initialization
# -------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# -------------------------------
# ICE Servers Function
# -------------------------------
def get_ice_servers():
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:relay1.expressturn.com:3478"], "username": "user", "credential": "pass"}
    ]

# -------------------------------
# Video Frame Callback Function
# -------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert the frame to a numpy array (BGR)
    image = frame.to_ndarray(format="bgr24")
    # Convert to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    # Draw landmarks if available
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# -------------------------------
# Working WebRTC Streamer Code
# -------------------------------
if st.session_state.get("action_confirmed", False):
    streamer_key = st.session_state["active_streamer_key"]
    st.info(f"Streaming activated! Perform the action: {st.session_state.get('action_word', 'your action')}")
    
    webrtc_ctx = webrtc_streamer(
        key=streamer_key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": get_ice_servers(),
            "iceTransportPolicy": "relay"
        },
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        st.session_state["streamer_running"] = True
    else:
        if st.session_state.get("streamer_running", False):
            st.session_state["streamer_running"] = False
            st.session_state["landmark_queue_snapshot"] = list(st.session_state["landmark_queue"])
            debug_log(f"Snapshot taken with {len(st.session_state['landmark_queue_snapshot'])} frames.")
            st.success("Streaming has stopped. You can now save keyframes.")

# -------------------------------
# (Optional) Prediction Display
# -------------------------------
# If a model is loaded and the streamer is running, attempt a prediction.
if st.session_state.get("model_loaded", False) and "webrtc_ctx" in locals() and webrtc_ctx.state.playing:
    st.subheader("Recognized Action:")
    try:
        # In a real app, you would extract processed landmark data.
        # Here we use a placeholder array for demonstration.
        row_data = np.zeros((1, 1000))
        prediction = st.session_state["model"].predict(row_data)
        predicted_index = np.argmax(prediction)
        predicted_class = st.session_state["label_encoder"].inverse_transform([predicted_index])[0]
        st.session_state["recognized_action"] = predicted_class
    except Exception as e:
        st.session_state["recognized_action"] = f"Error: {e}"
    st.write(st.session_state.get("recognized_action", "Waiting for prediction..."))

