import logging
import os
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from turn import get_ice_servers  # Import the TURN server utility

# Configure logging
logger = logging.getLogger(__name__)

# Set up Streamlit page
st.set_page_config(page_title="Record Actions", page_icon="🎥")
st.title("Record Actions")
st.info("This application uses WebRTC to stream and process video in real-time.")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Log function calls for debugging
    print("Processing frame...")

    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # Mirror the image

    # Add a message to indicate that the frame is being processed
    cv2.putText(
        img,
        "Processing Frame",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC streamer configuration
webrtc_streamer(
    key="record_actions",
    mode=WebRtcMode.SENDRECV,  # Allow sending and receiving video
    rtc_configuration={"iceServers": get_ice_servers()},  # Configure TURN servers
    media_stream_constraints={"video": True, "audio": False},  # Enable video, disable audio
    video_frame_callback=video_frame_callback,  # Process video frames
    async_processing=True,  # Enable asynchronous processing
)

