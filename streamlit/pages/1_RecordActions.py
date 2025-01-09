import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set the page configuration
st.set_page_config(layout="wide")

# Title of the app
st.title("Live Video Feed with Toggle Button")

# Instructions
st.write(
    "Click the 'Start/Stop' button to toggle the webcam feed."
)

# Define a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each video frame.
        Currently, it simply returns the frame without any modifications.
        """
        return frame

# Initialize session state for the video feed
if "feed_active" not in st.session_state:
    st.session_state["feed_active"] = False

# Toggle button logic
if st.button("Start/Stop"):
    st.session_state["feed_active"] = not st.session_state["feed_active"]  # Toggle state

# Display the video feed if active
if st.session_state["feed_active"]:
    st.info("Webcam feed is active. Click 'Start/Stop' to deactivate.")
    webrtc_streamer(
        key="live-video-feed",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},  # Video only
    )
else:
    st.info("Webcam feed is inactive. Click 'Start/Stop' to activate.")

# Footer
st.write("Use the 'Start/Stop' button above to toggle the webcam feed.")
