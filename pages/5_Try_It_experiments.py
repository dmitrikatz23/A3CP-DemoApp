import logging
from pathlib import Path
import threading
from collections import deque
import os
import re
import numpy as np
import cv2
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from huggingface_hub import HfApi, hf_hub_download
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import mediapipe as mp

# -------------------------------------------------------------------
# Logging Setup (Optional Debugging)
# -------------------------------------------------------------------
DEBUG_MODE = False

def debug_log(msg):
    if DEBUG_MODE:
        logging.info(msg)

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
logger.info("✅ Logging is initialized!")

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="TryIt - Thread-Safe Holistic", layout="wide")
st.title("TryIt - Thread-Safe Holistic + Model Inference")

# -------------------------------------------------------------------
# Hugging Face Setup
# -------------------------------------------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please set the 'Recorded_Datasets' secret.")
    st.stop()

hf_api = HfApi()
model_repo_name = "dk23/A3CP_models"
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Threading + Session State
# -------------------------------------------------------------------
# We replicate RecordActions.py style concurrency:
if "lock" not in st.session_state:
    st.session_state.lock = threading.Lock()

lock = st.session_state.lock

# Store the final predicted text
# (We only store the last predicted label, no queue needed)
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = "Waiting..."

# -------------------------------------------------------------------
# MediaPipe Holistic (Cached Resource)
# -------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_mediapipe_holistic():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_holistic()

# -------------------------------------------------------------------
# Model + Encoder in Session State
# -------------------------------------------------------------------
if "tryit_model" not in st.session_state:
    st.session_state.tryit_model = None
if "tryit_encoder" not in st.session_state:
    st.session_state.tryit_encoder = None

# -------------------------------------------------------------------
# Helper: Flatten Landmarks
# -------------------------------------------------------------------
def flatten_landmarks(results):
    """
    Flatten pose(33), left(21), right(21) => 75 landmarks x 3 coords = 225 values
    If your model uses face landmarks, adapt accordingly.
    """
    def to_xyz(landmark_list, count):
        if not landmark_list:
            return [0.0, 0.0, 0.0] * count
        return [coord for lm in landmark_list.landmark for coord in (lm.x, lm.y, lm.z)]
    
    pose_vals = to_xyz(results.pose_landmarks, 33)
    left_vals = to_xyz(results.left_hand_landmarks, 21)
    right_vals = to_xyz(results.right_hand_landmarks, 21)

    merged = pose_vals + left_vals + right_vals
    if len(merged) == 0:
        return None  # No detection
    return np.array(merged, dtype=np.float32)

# -------------------------------------------------------------------
# Video Callback - Thread-Safe (No st.* calls here)
# -------------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This is called in a separate thread (async_media_processor_X).
    We do not call any st.* here to avoid "missing ScriptRunContext".
    """
    input_bgr = frame.to_ndarray(format="bgr24")
    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

    # Process with Holistic
    results = holistic_model.process(img_rgb)

    # Annotate for display
    annotated_image = input_bgr.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Flatten
    landmarks = flatten_landmarks(results)
    if landmarks is not None:
        # Predict if model + encoder loaded
        model = st.session_state.tryit_model
        encoder = st.session_state.tryit_encoder
        if (model is not None) and (encoder is not None):
            # Pad for LSTM
            data = np.expand_dims(landmarks, axis=0)
            data = pad_sequences([data], maxlen=100, padding='post', dtype='float32', value=-1.0)

            preds = model.predict(data)
            label_idx = np.argmax(preds, axis=1)
            pred_label = encoder.inverse_transform(label_idx)[0]

            # Thread-safe store in session_state
            with lock:
                st.session_state.predicted_label = pred_label

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# -------------------------------------------------------------------
# Model/Encoder Selector (Like RecordActions approach)
# -------------------------------------------------------------------
@st.cache_data
def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs from HF."""
    files = hf_api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in files if f.endswith(".h5")]
    encoder_files = [f for f in files if f.endswith(".pkl")]

    pairs = {}
    for mf in model_files:
        if mf.startswith("LSTM_model_") and mf.endswith(".h5"):
            ts = mf[len("LSTM_model_"):-3]
            pairs.setdefault(ts, {})["model"] = mf

    for ef in encoder_files:
        if ef.startswith("label_encoder_") and ef.endswith(".pkl"):
            ts = ef[len("label_encoder_"):-4]
            pairs.setdefault(ts, {})["encoder"] = ef

    valid = []
    for ts, item in pairs.items():
        if "model" in item and "encoder" in item:
            valid.append((ts, item["model"], item["encoder"]))
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid

model_encoder_pairs = get_model_encoder_pairs()

# -------------------------------------------------------------------
# Sidebar UI for Model Selection
# -------------------------------------------------------------------
with st.sidebar:
    st.subheader("Select Model/Encoder from HF")

    if not model_encoder_pairs:
        st.warning("No .h5 + .pkl pairs found in the repo.")
    else:
        pair_dict = {}
        for ts, mf, ef in model_encoder_pairs:
            label = f"{ts} => {mf} & {ef}"
            pair_dict[label] = (mf, ef)

        chosen_label = st.selectbox("Model/Encoder pairs:", list(pair_dict.keys()))
        if chosen_label:
            chosen_model, chosen_encoder = pair_dict[chosen_label]
            st.write("**Chosen Model:**", chosen_model)
            st.write("**Chosen Encoder:**", chosen_encoder)

        if st.button("Load Model & Encoder") and chosen_label:
            model_path = os.path.join(LOCAL_MODEL_DIR, chosen_model)
            encoder_path = os.path.join(LOCAL_MODEL_DIR, chosen_encoder)

            if not os.path.exists(model_path):
                hf_hub_download(model_repo_name, chosen_model,
                                local_dir=LOCAL_MODEL_DIR,
                                repo_type="model", token=hf_token)
            if not os.path.exists(encoder_path):
                hf_hub_download(model_repo_name, chosen_encoder,
                                local_dir=LOCAL_MODEL_DIR,
                                repo_type="model", token=hf_token)

            # Load them
            st.session_state.tryit_model = tf.keras.models.load_model(model_path)
            st.session_state.tryit_encoder = joblib.load(encoder_path)

            st.success("Model + Encoder loaded successfully!")

# -------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("WebRTC Stream (Holistic)")
    webrtc_streamer(
        key="threadsafe-tryit-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

with right_col:
    st.header("Predicted Gesture")
    # We read from session_state (protected by lock in callback)
    with lock:
        st.write(f"**Current Prediction:** {st.session_state.predicted_label}")
