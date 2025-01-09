import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import time

# Set the page configuration
st.set_page_config(layout="wide")

# Title of the app
st.title("Timed Live Video Feed")

# Instructions
st.write(
    "Click the 'Start' button to activate your webcam feed for 10 seconds. "
    "The feed will stop automatically after 10 seconds."
)

# Define a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each video frame.
        Currently, it simply returns the frame without any modifications.
        """
        return frame

# Initialize session state to track feed activity
if "feed_active" not in st.session_state:
    st.session_state["feed_active"] = False
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None

# Start button logic
if st.button("Start"):
    st.session_state["feed_active"] = True
    st.session_state["start_time"] = time.time()  # Record the start time

# Display the video feed if active and within the 10-second limit
if st.session_state["feed_active"]:
    elapsed_time = time.time() - st.session_state["start_time"]
    if elapsed_time <= 10:
        st.info(f"Webcam feed is active. Time remaining: {10 - int(elapsed_time)} seconds.")
        webrtc_streamer(
            key="live-video-feed",
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},  # Video only
        )
    else:
        st.session_state["feed_active"] = False  # Stop the feed after 10 seconds
        st.session_state["start_time"] = None
        st.info("The webcam feed has stopped automatically after 10 seconds.")

# Display a message if the feed is inactive
if not st.session_state["feed_active"]:
    st.info("Webcam feed is inactive. Click 'Start' to activate.")
