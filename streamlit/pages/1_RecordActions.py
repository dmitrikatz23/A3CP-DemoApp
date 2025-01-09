import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set the page configuration
st.set_page_config(layout='wide')

# Title of the app
st.title("Start Live Video Feed")

# Instructions
st.write(
    "Click the 'Start' button below to access and display your webcam feed. "
    "Ensure your browser permissions allow access to your webcam."
)

# Define a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each video frame.
        For now, it simply returns the frame without modification.
        """
        return frame

# Button to start the webcam feed
if st.button('Start'):
    # Show the live video feed
    st.info("Webcam feed started. Close the browser tab or stop the app to stop the feed.")
    webrtc_streamer(
        key="live-video-feed",
        video_transformer_factory=VideoTransformer,  # Processes video frames
        media_stream_constraints={"video": True, "audio": False},  # Access only the video
    )

# Footer message
st.write(
    "Your webcam feed will remain active as long as the app is running and the tab is open."
)
