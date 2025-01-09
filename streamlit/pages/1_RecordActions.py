import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Set the page configuration
st.set_page_config(layout='wide')

# Title of the app
st.title('Webcam Permission Test using Streamlit-WeRTC')

# Instructions
st.write(
    "This app uses `streamlit-webrtc` to access your webcam. "
    "Ensure your browser permissions allow camera access."
)

# Webcam permission check using streamlit-webrtc
def check_camera_access():
    st.info("Press the 'Start' button below to test webcam access.")
    webrtc_streamer(key="camera-test")

# Button to test webcam
if st.button('Check Webcam'):
    check_camera_access()

# Footer message
st.write("If the camera feed is displayed above, your webcam is accessible!")
