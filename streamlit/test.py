import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2  # For image processing

# Define the Video Processor class
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert the frame to a NumPy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        
        # Add a debug message in the terminal for each frame
        print("Frame received and processed")
        
        # Add an overlay text to the frame
        img = cv2.putText(
            img,  # Frame
            "Streaming",  # Text to overlay
            (50, 50),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Font scale
            (0, 255, 0),  # Color (Green)
            2,  # Thickness
        )
        
        # Return the modified frame
        return frame.from_ndarray(img, format="bgr24")

# Streamlit app
def main():
    st.title("Streamlit WebRTC Debugging Test")

    # Create the WebRTC streamer
    webrtc_streamer(
    key="test",
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]},
        ]
    },
    video_processor_factory=VideoProcessor,
)

if __name__ == "__main__":
    main()
