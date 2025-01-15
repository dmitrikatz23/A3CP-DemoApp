import logging
import os
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from sample_utils.turn import get_ice_servers  # Import the TURN server utility

# Configure logging
logger = logging.getLogger(__name__)

# Set up Streamlit page
st.set_page_config(page_title="Record Actions", page_icon="🎥")
st.title("Record Actions")
st.info("This application uses WebRTC to stream and process video in real-time.")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # Mirror image
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer configuration
webrtc_streamer(
    key="record_actions",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
)


