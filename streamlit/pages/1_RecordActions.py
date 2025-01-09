import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set the page configuration
st.set_page_config(layout='wide')

# Title of the app
st.title('Live video feed using Streamlit-WeRTC')

# Instructions
st.write(
    "Click the 'Start' button to access and display your webcam feed. "
    "Ensure your brwoser permissions allow access to your webcam."
)

#define a custom video transformer class

class VideoTransformer (VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        #receives video frame and returns unchanged
        return frame
    

# Button to start the webcam feed

if st.button('Start'):
    #show the live video feed
    webrtc_streamer(
    key="live-video-feed", 
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},  # Enables only video
)


# Footer message
st.write(
    "Your webcam feed will remain active until you stop the Streamlit app or close the browser tab."
)
