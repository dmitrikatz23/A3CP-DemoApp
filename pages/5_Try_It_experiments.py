import streamlit as st
import mediapipe as mp
from streamlit_webrtc import WebRtcMode, webrtc_streamer, WebRtcStreamerContext
import cv2
import av
import numpy as np
import threading
import collections
import tensorflow as tf
import joblib
from huggingface_hub import hf_hub_download, HfApi
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------
# Page + Basic Config
# -------------------------------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Real-Time Gesture Inference")

hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure it's set.")
    st.stop()

# -------------------------------------------------------
# Lock + Session State
# -------------------------------------------------------
if "prediction_lock" not in st.session_state:
    st.session_state.prediction_lock = threading.Lock()

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = "Waiting..."

# Optional: Keep a queue if you want a history of predictions
if "prediction_queue" not in st.session_state:
    st.session_state.prediction_queue = collections.deque(maxlen=1000)

# -------------------------------------------------------
# MediaPipe Holistic Setup (like RecordActions.py)
# -------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_holistic():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_holistic()

def process_frame(frame_bgr):
    """Process a single BGR frame with the Holistic model."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(img_rgb)

    # Draw the landmarks onto a copy for display
    annotated = frame_bgr.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Extract just the pose + hands. (Face optional if your model uses it.)
    # For simplicity, assume your model wants 33 pose + 21 left + 21 right = 75 landmarks x 3 coords each = 225
    # If your model expects more, adapt accordingly.

    def flatten_landmarks(landmark_list, count):
        # Each landmark => (x, y, z)
        if landmark_list:
            return [coord for lm in landmark_list.landmark for coord in (lm.x, lm.y, lm.z)]
        else:
            return [0.0, 0.0, 0.0] * count

    pose_data = flatten_landmarks(results.pose_landmarks, 33)
    left_hand_data = flatten_landmarks(results.left_hand_landmarks, 21)
    right_hand_data = flatten_landmarks(results.right_hand_landmarks, 21)

    # Merge them
    full_landmarks = pose_data + left_hand_data + right_hand_data
    if len(full_landmarks) == 0:
        return annotated, None  # No detection

    return annotated, np.array(full_landmarks, dtype=np.float32)

# -------------------------------------------------------
# Model + Encoder
# -------------------------------------------------------
hf_api = HfApi()
model_repo = "dk23/A3CP_models"
LOCAL_MODEL_DIR = "local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None
if "loaded_encoder" not in st.session_state:
    st.session_state.loaded_encoder = None

@st.cache_data
def list_model_pairs():
    """List .h5 + .pkl from Hugging Face (like in RecordActions)."""
    files = hf_api.list_repo_files(repo_id=model_repo, repo_type="model", token=hf_token)
    model_files = [f for f in files if f.endswith(".h5")]
    encoder_files = [f for f in files if f.endswith(".pkl")]
    pairs = {}
    # Match them by timestamp
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

model_pairs = list_model_pairs()

# -------------------------------------------------------
# WebRTC Callback
# -------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """This runs in a separate thread (async_media_processor_X). 
       No direct Streamlit calls here to avoid missing ScriptRunContext."""
    bgr = frame.to_ndarray(format="bgr24")

    # 1) Process with Holistic
    annotated, landmarks = process_frame(bgr)

    # 2) If we have a loaded model + landmarks, do inference
    model = st.session_state.loaded_model
    encoder = st.session_state.loaded_encoder
    if (model is not None) and (encoder is not None) and (landmarks is not None):
        # Preprocess for LSTM
        landmarks = np.expand_dims(landmarks, axis=0)  # shape (1, N)
        # Our LSTM expects shape (batch, time=some_padding, features).
        # So we wrap with pad_sequences
        padded = pad_sequences([landmarks], maxlen=100, padding='post', value=-1.0, dtype='float32')
        # Model predicts
        preds = model.predict(padded)
        label_idx = np.argmax(preds, axis=1)
        pred_text = encoder.inverse_transform(label_idx)[0]

        # 3) Save into session_state with a lock
        with st.session_state.prediction_lock:
            st.session_state.current_prediction = pred_text

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -------------------------------------------------------
# Sidebar: Model Selection
# -------------------------------------------------------
with st.sidebar:
    st.subheader("Model Selection")
    if not model_pairs:
        st.write("No .h5/.pkl pairs found.")
    else:
        labels = [f"{t} => {m} & {e}" for t, m, e in model_pairs]
        choice = st.selectbox("Select Model/Encoder Pair:", labels)
        if choice:
            chosen_ts, chosen_model, chosen_encoder = None, None, None
            for t, m, e in model_pairs:
                label = f"{t} => {m} & {e}"
                if label == choice:
                    chosen_ts, chosen_model, chosen_encoder = t, m, e
                    break
            st.write(f"Chosen Model: {chosen_model}")
            st.write(f"Chosen Encoder: {chosen_encoder}")

            if st.button("Load This Model"):
                # Download if needed
                model_path = os.path.join(LOCAL_MODEL_DIR, chosen_model)
                enc_path = os.path.join(LOCAL_MODEL_DIR, chosen_encoder)

                if not os.path.exists(model_path):
                    hf_hub_download(model_repo, chosen_model, local_dir=LOCAL_MODEL_DIR,
                                    repo_type="model", token=hf_token)
                if not os.path.exists(enc_path):
                    hf_hub_download(model_repo, chosen_encoder, local_dir=LOCAL_MODEL_DIR,
                                    repo_type="model", token=hf_token)

                # Load them
                st.session_state.loaded_model = tf.keras.models.load_model(model_path)
                st.session_state.loaded_encoder = joblib.load(enc_path)
                st.success("Model & Encoder loaded!")

# -------------------------------------------------------
# Layout
# -------------------------------------------------------
left, right = st.columns([1, 2])

with left:
    st.header("Real-Time Stream")
    # No direct references to st.write in the callback => no ScriptRunContext error
    webrtc_streamer(
        key="tryit-livestream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

with right:
    st.header("Predicted Gesture (Thread-Safe)")
    # We read from session_state safely
    pred_text = st.session_state.current_prediction
    st.write(f"**Current Prediction:** {pred_text}")
