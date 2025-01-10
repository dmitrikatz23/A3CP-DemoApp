import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to ndarray
        return frame.from_ndarray(img, format="bgr24")  # Return the same frame

st.title("Minimal WebRTC Test")

webrtc_streamer(
    key="test",
    video_processor_factory=VideoProcessor,  # Process frames using the VideoProcessor
)
