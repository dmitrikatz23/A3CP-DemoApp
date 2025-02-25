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
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from huggingface_hub import HfApi
from sample_utils.turn import get_ice_servers

# -----------------------------------
# Debug Logging Setup (Optional)
# -----------------------------------
DEBUG_MODE = False  # Set True to see debug logs

def debug_log(msg):
    if DEBUG_MODE:
        st.write(f"[DEBUG] {msg}")

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

# -----------------------------------
# Hugging Face: Retrieve Model/Encoder Pairs
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is set.")
    st.stop()

api = HfApi()
model_repo_name = "dk23/A3CP_models"  # Where .h5 and .pkl files live

@st.cache_data
def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs from the HF repo."""
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    
    pairs = {}
    # Identify timestamps from filenames like 'LSTM_model_{ts}.h5', 'label_encoder_{ts}.pkl'
    for mf in model_files:
        if mf.startswith("LSTM_model_") and mf.endswith(".h5"):
            ts = mf[len("LSTM_model_"):-len(".h5")]
            pairs.setdefault(ts, {})["model"] = mf
    for ef in encoder_files:
        if ef.startswith("label_encoder_") and ef.endswith(".pkl"):
            ts = ef[len("label_encoder_"):-len(".pkl")]
            pairs.setdefault(ts, {})["encoder"] = ef
    
    valid_pairs = []
    for ts, items in pairs.items():
        if "model" in items and "encoder" in items:
            valid_pairs.append((ts, items["model"], items["encoder"]))
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
if "streamer_running" not in st.session_state:
    st.session_state["streamer_running"] = False

# -----------------------------------
# Sidebar: Model/Encoder Selection
# -----------------------------------
with st.sidebar:
    st.subheader("Model/Encoder Pair Selection")
    
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found.")
    else:
        # Build a label -> (model, encoder) mapping
        pair_options = {}
        for ts, model_file, encoder_file in model_encoder_pairs:
            label = f"{ts} | Model: {model_file} | Encoder: {encoder_file}"
            pair_options[label] = (model_file, encoder_file)
        
        selected_label = st.selectbox(
            "Choose a matched pair", 
            options=list(pair_options.keys())
        )
        
        if selected_label:
            mfile, efile = pair_options[selected_label]
            st.write("**Selected Model:**", mfile)
            st.write("**Selected Encoder:**", efile)
        
        # Button to confirm model
        if st.button("Confirm Model") and selected_label:
            st.session_state["selected_pair"] = pair_options[selected_label]
            # Create a unique key for the streamer
            key = f"tryit-stream-{re.sub(r'[^a-zA-Z0-9]', '_', selected_label)}"
            st.session_state["active_streamer_key"] = key
            st.session_state["model_confirmed"] = True
            st.success("Model confirmed! The WebRTC streamer will appear in the main area.")

# -----------------------------------
# Example Video Frame Callback
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    debug_log("📷 video_frame_callback triggered")
    
    # Example overlay text (you can insert inference code here)
    cv2.putText(img_bgr, "Streaming Active", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -----------------------------------
# Main Layout: 2 Columns
# -----------------------------------
left_col, right_col = st.columns([1, 2])

# Left Column: WebRTC Streamer (Only if Model Confirmed)
with left_col:
    st.header("WebRTC Stream")
    if st.session_state["model_confirmed"] and st.session_state["active_streamer_key"]:
        webrtc_ctx = webrtc_streamer(
            key=st.session_state["active_streamer_key"],
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

        # Track if streamer is active
        if webrtc_ctx.state.playing:
            st.session_state["streamer_running"] = True
        else:
            # If streamer was playing and just stopped, handle cleanup if needed
            if st.session_state["streamer_running"]:
                st.session_state["streamer_running"] = False
                st.success("Streaming was stopped.")
    else:
        st.info("Please select and confirm a model in the sidebar to start streaming.")

# Right Column: Placeholder for Additional Info
with right_col:
    st.header("Model & Encoder Info")
    if st.session_state["model_confirmed"]:
        model_file, encoder_file = st.session_state["selected_pair"]
        st.write(f"**Confirmed Model**: {model_file}")
        st.write(f"**Confirmed Encoder**: {encoder_file}")
        st.write("You can add inference logic here.")
    else:
        st.write("No model confirmed yet.")

