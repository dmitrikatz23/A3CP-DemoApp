#on the left slide the user to choose a model and an encoder
#button to start the streamer
#uses holistic model to vectorize gestures
#uses chosen model to predict the meaning of gesture
#displays the predicted gesture

import os
import re
import cv2
import av
import numpy as np
import streamlit as st
from huggingface_hub import HfApi
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from sample_utils.turn import get_ice_servers

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

api = HfApi()
model_repo_name = "dk23/A3CP_models"  # Repository with .h5 (model) and .pkl (encoder) files

@st.cache_data
def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs from the HF repo."""
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]

    pairs = {}
    # Extract timestamps from names like: LSTM_model_{timestamp}.h5, label_encoder_{timestamp}.pkl
    for mf in model_files:
        if mf.startswith("LSTM_model_") and mf.endswith(".h5"):
            ts = mf[len("LSTM_model_"):-3]  # remove "LSTM_model_" + ".h5"
            pairs.setdefault(ts, {})["model"] = mf
    for ef in encoder_files:
        if ef.startswith("label_encoder_") and ef.endswith(".pkl"):
            ts = ef[len("label_encoder_"):-4]  # remove "label_encoder_" + ".pkl"
            pairs.setdefault(ts, {})["encoder"] = ef

    valid_pairs = []
    for ts, items in pairs.items():
        if "model" in items and "encoder" in items:
            valid_pairs.append((ts, items["model"], items["encoder"]))

    # Sort descending by timestamp
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

# -----------------------------
# Session State (TryIt-Specific)
# -----------------------------
if "tryit_model_confirmed" not in st.session_state:
    st.session_state["tryit_model_confirmed"] = False

if "tryit_selected_pair" not in st.session_state:
    st.session_state["tryit_selected_pair"] = None

if "tryit_streamer_key" not in st.session_state:
    st.session_state["tryit_streamer_key"] = None

if "tryit_streamer_running" not in st.session_state:
    st.session_state["tryit_streamer_running"] = False

# -----------------------------
# Sidebar: Model/Encoder Selector
# -----------------------------
with st.sidebar:
    st.subheader("Select a Model/Encoder Pair")
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found.")
    else:
        # Build a map for display
        pair_options = {}
        for ts, mf, ef in model_encoder_pairs:
            label = f"{ts} | Model: {mf} | Encoder: {ef}"
            pair_options[label] = (mf, ef)
        
        selected_label = st.selectbox(
            "Choose a matched pair:",
            list(pair_options.keys())
        )
        
        if selected_label:
            chosen_model, chosen_encoder = pair_options[selected_label]
            st.write("**Selected Model:**", chosen_model)
            st.write("**Selected Encoder:**", chosen_encoder)
        
        if st.button("Confirm Model") and selected_label:
            # Store the selection in session state
            st.session_state["tryit_selected_pair"] = pair_options[selected_label]
            # Create a unique, stable key for this session
            st.session_state["tryit_streamer_key"] = f"tryit-stream-{re.sub(r'[^a-zA-Z0-9]', '_', selected_label)}"
            st.session_state["tryit_model_confirmed"] = True
            st.success("Model confirmed! The WebRTC streamer will appear in the main area.")

# -----------------------------
# Example Frame Callback
# -----------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """You can insert inference code here if desired."""
    img_bgr = frame.to_ndarray(format="bgr24")
    cv2.putText(img_bgr, "TryIt Streaming",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -----------------------------
# Main Layout: 2 Columns
# -----------------------------
left_col, right_col = st.columns([1, 2])

# Left Column: WebRTC Streamer
with left_col:
    st.header("WebRTC Stream")
    if st.session_state["tryit_model_confirmed"] and st.session_state["tryit_streamer_key"]:
        webrtc_ctx = webrtc_streamer(
            key=st.session_state["tryit_streamer_key"],
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
        if webrtc_ctx.state.playing:
            st.session_state["tryit_streamer_running"] = True
        else:
            if st.session_state["tryit_streamer_running"]:
                st.session_state["tryit_streamer_running"] = False
                st.info("Streaming was stopped.")
    else:
        st.info("Please select and confirm a model in the sidebar.")

# Right Column: Optional Additional UI
with right_col:
    st.header("Inference / Status")
    if st.session_state["tryit_model_confirmed"]:
        mf, ef = st.session_state["tryit_selected_pair"]
        st.write(f"**Confirmed Model**: {mf}")
        st.write(f"**Confirmed Encoder**: {ef}")
        st.write("Add your inference logic or additional info here.")
    else:
        st.write("No model confirmed yet.")
