import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set the page configuration
st.set_page_config(layout="wide")

# Title of the app
st.title("Live Video Feed")

# Instructions
st.write("Click the 'Start' button below to activate your webcam feed.")

# Define a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each video frame.
        Currently, it simply returns the frame without any modifications.
        """
        return frame

# Initialize session state to track if the feed has started
if "feed_started" not in st.session_state:
    st.session_state["feed_started"] = False

# Start button
if st.button("Start") and not st.session_state["feed_started"]:
    st.session_state["feed_started"] = True  # Mark feed as started

# Display the video feed if started
if st.session_state["feed_started"]:
    st.info("Webcam feed is active. Close the browser tab or app to stop.")
    webrtc_streamer(
        key="live-video-feed",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Video only
    )
else:
    st.info("Webcam feed is inactive. Click 'Start' to activate.")

# Footer
st.write("The video feed will remain active until the app is stopped.")
