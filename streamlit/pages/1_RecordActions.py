import streamlit as st
import cv2
import threading
import numpy as np

# Function to handle the video stream
def video_stream():
    global stop_stream, frame_placeholder
    cap = cv2.VideoCapture(0)  # Open the webcam (camera 0)

    while not stop_stream:
        ret, frame = cap.read()
        if ret:
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame in the Streamlit app
            frame_placeholder.image(frame, channels="RGB")
        else:
            st.error("Failed to capture video.")
            break

    cap.release()  # Release the webcam

# Streamlit app
def main():
    global stop_stream, frame_placeholder
    st.title("Video Stream Toggle App")

    # Button to start/stop video stream
    toggle_button = st.button("Start/Stop Video Stream")

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    if toggle_button:
        st.session_state.streaming = not st.session_state.streaming

    if st.session_state.streaming:
        stop_stream = False
        frame_placeholder = st.empty()  # Placeholder for video frames
        threading.Thread(target=video_stream).start()  # Start video stream in a thread
    else:
        stop_stream = True

if __name__ == "__main__":
    stop_stream = True
    frame_placeholder = None
    main()
