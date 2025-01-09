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
    "Click stop to turn it off"
)

# Define a custom video transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each video frame.
        For now, it simply returns the frame without modification.
        """
        return frame

#iniitalize session state for video feed toggle
if "feed_active" not in st.session_state:
    st.session_state["feed_active"] = False

col1,col2 = st.columns(2)

with col1:
    if st.button("Start"):
        st.session_state["feed_active"] = True

with col2:
    if st.button("Stop"):
        st.session_state["feed_active"]= False

# display the video feed if active
if st.session_state["feed_active"]:
    st.info ("webcam feed is active")
    webrtc_streamer(
        key="live-video-feed",
        video_transformer_factory=VideoTransformer,  # Processes video frames
        media_stream_constraints={"video": True, "audio": False},  # Access only the video
    )
else:
    st.info("Webcam is inactive. Click 'Start' to activate.")

# Footer message
st.write(
    "Toggle the feed using the Start/Stop buttons."
)
