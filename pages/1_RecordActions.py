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

# Define the video frame callback function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Process each video frame from the webcam.

    Parameters:
    - frame (av.VideoFrame): The input video frame.

    Returns:
    - av.VideoFrame: The processed video frame.
    """
    # Convert the video frame to a NumPy array
    img = frame.to_ndarray(format="bgr24")

    # Flip the image horizontally for a mirrored view
    img = cv2.flip(img, 1)

    # Optional: Add some text to the video frame
    cv2.putText(
        img,
        "Recording in Progress",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Return the processed video frame
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

