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
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from huggingface_hub import HfApi
from sample_utils.turn import get_ice_servers  # Assumes this returns a valid ICE servers list

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference & Streaming Interface")

# -----------------------------------
# Hugging Face and Model/Encoder Pair Selection
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

api = HfApi()
model_repo_name = "dk23/A3CP_models"  # Repository containing trained model and encoder files

@st.cache_data
def get_model_encoder_pairs():
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    
    # Build pairs based on timestamp extracted from filenames
    pairs = {}
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
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

# Initialize session state for confirmation
if "model_confirmed" not in st.session_state:
    st.session_state["model_confirmed"] = False
if "selected_pair" not in st.session_state:
    st.session_state["selected_pair"] = None
if "active_streamer_key" not in st.session_state:
    st.session_state["active_streamer_key"] = None

# -----------------------------------
# Layout: Left and Right Columns
# -----------------------------------
left_col, right_col = st.columns([1, 2])

# -----------------------------------
# Left Column: Model Selection and Streaming
# -----------------------------------
with left_col:
    if not st.session_state["model_confirmed"]:
        st.header("Select Model/Encoder Pair")
        if not model_encoder_pairs:
            st.warning("No valid model/encoder pairs found in the repository.")
        else:
            pair_options = {}
            for ts, model_file, encoder_file in model_encoder_pairs:
                label = f"{ts} | Model: {model_file} | Encoder: {encoder_file}"
                pair_options[label] = (model_file, encoder_file)
            selected_label = st.selectbox("Select a Model/Encoder Pair", list(pair_options.keys()))
            st.write("**Selected Pair:**")
            st.write(f"Model File: `{pair_options[selected_label][0]}`")
            st.write(f"Encoder File: `{pair_options[selected_label][1]}`")
            
            if st.button("Confirm Model"):
                st.session_state["selected_pair"] = pair_options[selected_label]
                # Create a unique key for the streamer based on the selection
                key = f"tryit-stream-{re.sub(r'[^a-zA-Z0-9]', '_', selected_label)}"
                st.session_state["active_streamer_key"] = key
                st.session_state["model_confirmed"] = True
                st.success("Model confirmed! Streaming will now start.")
    else:
        st.header("Streaming")
        # Launch the WebRTC streamer in the left column using the active key
        webrtc_ctx = webrtc_streamer(
            key=st.session_state["active_streamer_key"],
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=lambda frame: av.VideoFrame.from_ndarray(
                cv2.putText(
                    frame.to_ndarray(format="bgr24"),
                    "Streaming Active",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                ),
                format="bgr24"
            ),
            async_processing=True,
        )

# -----------------------------------
# Right Column: Additional Interface or Output
# -----------------------------------
with right_col:
    st.header("Try It Interface")
    st.write("The selected model/encoder pair will be used for inference.")
    # TODO: Add further inference or visualization logic here.
