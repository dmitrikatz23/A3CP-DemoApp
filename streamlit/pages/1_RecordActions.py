import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Set the page configuration
st.set_page_config(layout='wide')

# Title of the app
st.title('Continuous Webcam Check using Streamlit-WeRTC')

# Instructions
st.write(
    "This app uses `streamlit-webrtc` to continuously access your webcam. "
    "Ensure your browser permissions allow camera access."
)

# Webcam streaming function
def continuous_camera_check():
    st.info("Your webcam is active. Close the app or browser tab to stop the feed.")
    webrtc_streamer(key="camera-check", video_transformer_factory=None)

# Button to start continuous webcam check
if st.button('Start Webcam Check'):
    continuous_camera_check()

# Footer message
st.write(
    "Your webcam will remain active until you close the browser tab or stop the Streamlit app."
)
