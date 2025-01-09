import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Set the page configuration
st.set_page_config(layout='wide')

# Title of the app
st.title('Live video feed using Streamlit-WeRTC')

# Instructions
st.write(
    "This app demonstrates a live video feed using `streamlit-webrtc`. "
    "Ensure you allow camera access when prompted by your browser."
)

#define a custom video transformer class

class VideoTransformer (VideoTransformerBase):
    def transform(self, frame):
        #receives video frame and returns unchanged
        return frame
    

# Start the live video feed
webrtc_streamer(
    key="live-video-feed", 
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},  # Enables only video
)


# Footer message
st.write(
    "Your webcam feed will remain active until you stop the Streamlit app or close the browser tab."
)
