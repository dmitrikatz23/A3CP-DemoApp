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
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils  # Drawing helper

# -----------------------------
# Helper Function: Extract Landmarks from Frame
# -----------------------------
def extract_landmarks(image):
    """Extract holistic landmarks from an image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)

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

    return np.array(landmarks, dtype=np.float32) if landmarks else None, results

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
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

with right_col:
    st.header("Predicted Gesture")
    st.write(f"**Prediction:** {st.session_state.get('tryit_predicted_text', 'Waiting for input...')}")  
