import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Set the page configuration
st.set_page_config(layout='wide')

# Title of the app
st.title('Continuous Webcam Check using Streamlit-WeRTC')

# Instructions
st.write(
    "This app keeps your webcam active continuously using `streamlit-webrtc`. "
    "Ensure you allow camera access when prompted by your browser."
)

# Continuous webcam feed
webrtc_streamer(
    key="camera-continuous", 
    video_transformer_factory=None,
    media_stream_constraints={"video": True, "audio": False},  # Enables only video
)

# Footer message
st.write(
    "Your webcam feed will remain active until you stop the Streamlit app or close the browser tab."
)
