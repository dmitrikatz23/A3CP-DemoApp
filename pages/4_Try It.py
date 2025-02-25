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
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

# -----------------------------------
# Hugging Face Setup and Model/Encoder Pair Retrieval
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is set.")
    st.stop()

api = HfApi()
model_repo_name = "dk23/A3CP_models"  # Your model repo containing .h5 and .pkl files

@st.cache_data
def get_model_encoder_pairs():
    """Retrieve matched model/encoder pairs (by timestamp) from the HF repo."""
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    
    pairs = {}
    # Extract timestamps from filenames like 'LSTM_model_{timestamp}.h5' or 'label_encoder_{timestamp}.pkl'
    for mf in model_files:
        if mf.startswith("LSTM_model_") and mf.endswith(".h5"):
            ts = mf[len("LSTM_model_"):-len(".h5")]
            pairs.setdefault(ts, {})["model"] = mf
            
    for ef in encoder_files:
        if ef.startswith("label_encoder_") and ef.endswith(".pkl"):
            ts = ef[len("label_encoder_"):-len(".pkl")]
            pairs.setdefault(ts, {})["encoder"] = ef

    valid_pairs = []
    for ts, files in pairs.items():
        if "model" in files and "encoder" in files:
            valid_pairs.append((ts, files["model"], files["encoder"]))
    
    # Sort by timestamp descending
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

# -----------------------------------
# Session State for Model Confirmation & Streamer
# -----------------------------------
if "model_confirmed" not in st.session_state:
    st.session_state["model_confirmed"] = False
if "selected_pair" not in st.session_state:
    st.session_state["selected_pair"] = None
if "active_streamer_key" not in st.session_state:
    st.session_state["active_streamer_key"] = None

# -----------------------------------
# Sidebar: Model/Encoder Pair Selector
# -----------------------------------
with st.sidebar:
    st.subheader("Select a Model/Encoder Pair")
    
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found in the repository.")
    else:
        pair_options = {}
        for ts, model_file, encoder_file in model_encoder_pairs:
            label = f"{ts} | Model: {model_file} | Encoder: {encoder_file}"
            pair_options[label] = (model_file, encoder_file)
        
        selected_label = st.selectbox(
            "Choose a matched pair",
            list(pair_options.keys()) if pair_options else []
        )
        
        if selected_label:
            st.write("**Selected Model:**", pair_options[selected_label][0])
            st.write("**Selected Encoder:**", pair_options[selected_label][1])
        
        if st.button("Confirm Model") and selected_label:
            st.session_state["selected_pair"] = pair_options[selected_label]
            # Create a unique key for the streamer based on selection
            key = f"tryit-stream-{re.sub(r'[^a-zA-Z0-9]', '_', selected_label)}"
            st.session_state["active_streamer_key"] = key
            st.session_state["model_confirmed"] = True
            st.success("Model confirmed! The WebRTC streamer will appear in the main area.")

# -----------------------------------
# Example Video Frame Callback
# -----------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Example callback function to demonstrate WebRTC streaming.
    You can insert inference logic here if desired.
    """
    # Convert to BGR for OpenCV
    img = frame.to_ndarray(format="bgr24")
    # (Optional) Overlay a simple text
    cv2.putText(img, "Streaming Active", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------
# Main Area Layout (Two Columns)
# -----------------------------------
left_col, right_col = st.columns([1, 2])

# Left Column: WebRTC Streamer (only if model confirmed)
with left_col:
    st.header("Streaming")
    if st.session_state.get("model_confirmed") and st.session_state.get("active_streamer_key"):
        webrtc_ctx = webrtc_streamer(
            key=st.session_state["active_streamer_key"],
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
    else:
        st.info("Please select and confirm a model/encoder pair in the sidebar.")

# Right Column: Additional Info/Output
with right_col:
    st.header("Additional Information")
    st.write("Once streaming is active, you can implement further inference logic.")
    if st.session_state.get("model_confirmed"):
        model_file, encoder_file = st.session_state["selected_pair"]
        st.write(f"**Using Model**: {model_file}")
        st.write(f"**Using Encoder**: {encoder_file}")
    else:
        st.write("No model confirmed yet.")
