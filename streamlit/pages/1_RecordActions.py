import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Define a custom video transformer
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Perform any processing on the video frames here if needed
        img = frame.to_ndarray(format="bgr24")
        return img

# Streamlit app
def main():
    st.title("Video Streaming App with Streamlit-WebRTC")

    st.write("Click the 'Start' button to begin video streaming.")
    # Start the video stream
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
