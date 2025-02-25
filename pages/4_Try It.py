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
import threading
from huggingface_hub import HfApi
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from sample_utils.turn import get_ice_servers  # Must return valid ICE servers

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

# -----------------------------------
# Hugging Face Setup
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

api = HfApi()
model_repo_name = "dk23/A3CP_models"  # Repository for trained model & encoder

@st.cache_data
def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs from the HF model repo."""
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]

    # Build dict of pairs based on timestamp in filenames
    pairs = {}
    for mf in model_files:
        if mf.startswith("LSTM_model_") and mf.endswith(".h5"):
            ts = mf[len("LSTM_model_"):-len(".h5")]
            pairs.setdefault(ts, {})["model"] = mf

    for ef in encoder_files:
        if ef.startswith("label_encoder_") and ef.endswith(".pkl"):
            ts = ef[len("label_encoder_"):-len(".pkl")]
            pairs.setdefault(ts, {})["encoder"] = ef

    # Keep only pairs that have both a model and an encoder
    valid_pairs = []
    for ts, files in pairs.items():
        if "model" in files and "encoder" in files:
            valid_pairs.append((ts, files["model"], files["encoder"]))

    # Sort by timestamp descending (most recent first)
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

# -----------------------------------
# Session State Initialization
# -----------------------------------
if "model_confirmed" not in st.session_state:
    st.session_state["model_confirmed"] = False
if "selected_pair" not in st.session_state:
    st.session_state["selected_pair"] = None
if "active_streamer_key" not in st.session_state:
    st.session_state["active_streamer_key"] = None

# -----------------------------------
# Sidebar: Model/Encoder Pair Selection
# -----------------------------------
with st.sidebar:
    st.header("Select a Model/Encoder Pair")
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found in the repository.")
    else:
        # Build user-friendly labels
        pair_options = {}
        for ts, model_file, encoder_file in model_encoder_pairs:
            label = f"{ts} | Model: {model_file} | Encoder: {encoder_file}"
            pair_options[label] = (model_file, encoder_file)

        # Dropdown for selecting a pair
        selected_label = st.selectbox("Model/Encoder Pair:", list(pair_options.keys()))
        selected_pair = pair_options[selected_label]

        # Show details of the currently selected pair
        st.write(f"**Model:** `{selected_pair[0]}`")
        st.write(f"**Encoder:** `{selected_pair[1]}`")

        # Confirm button
        if st.button("Confirm Model"):
            st.session_state["selected_pair"] = selected_pair
            # Generate a unique key for the streamer based on the selection
            key = f"tryit-stream-{re.sub(r'[^a-zA-Z0-9]', '_', selected_label)}"
            st.session_state["active_streamer_key"] = key
            st.session_state["model_confirmed"] = True
            st.success("Model/Encoder pair confirmed! The streamer will appear below.")

# -----------------------------------
# Video Frame Callback
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    # (Optional) Add inference logic here with the selected model/encoder, if needed

    # Simple overlay to show streaming is active
    cv2.putText(
        img_bgr,
        "Streaming Active",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -----------------------------------
# Main Area: Show Streamer (similar to RecordActions.py)
# -----------------------------------
left_col, right_col = st.columns([1, 2])
with left_col:
    st.subheader("Controls / Info")
    st.write("You can place additional controls or metadata here.")

with right_col:
    st.subheader("Live Stream")
    if st.session_state.get("model_confirmed", False) and st.session_state.get("active_streamer_key"):
        webrtc_ctx = webrtc_streamer(
            key=st.session_state["active_streamer_key"],
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
    else:
        st.info("Please select and confirm a model/encoder pair in the sidebar to start streaming.")
