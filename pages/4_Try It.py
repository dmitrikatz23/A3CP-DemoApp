import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import av
import mediapipe as mp
from collections import deque
import threading

# -------------------------------
# Helper Functions & Configuration
# -------------------------------

def debug_log(message):
    # For debugging purposes, we simply write messages to the Streamlit app.
    st.write(f"DEBUG: {message}")

def get_ice_servers():
    # Return a list of ICE servers for WebRTC connections.
    return [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["turn:relay1.expressturn.com:3478"], "username": "user", "credential": "pass"}
    ]

# -------------------------------
# Session State Initialization
# -------------------------------

# For a working example, we set these defaults.
if "action_confirmed" not in st.session_state:
    st.session_state["action_confirmed"] = True  # In practice, set this when the user confirms an action.
if "active_streamer_key" not in st.session_state:
    st.session_state["active_streamer_key"] = "gesture_streamer"  # Use a persistent key to avoid reinitialization.
if "streamer_running" not in st.session_state:
    st.session_state["streamer_running"] = False
if "action_word" not in st.session_state:
    st.session_state["action_word"] = "Gesture"
if "landmark_queue" not in st.session_state:
    st.session_state["landmark_queue"] = deque(maxlen=1000)

# Create a local reference for convenience.
landmark_queue = st.session_state["landmark_queue"]

# -------------------------------
# Streamlit Page Configuration
# -------------------------------

st.set_page_config(page_title="Gesture Recognition", layout="wide")
st.title("Gesture Recognition System")

# -------------------------------
# MediaPipe Initialization
# -------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# -------------------------------
# Video Frame Callback Function
# -------------------------------

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This function is called for every video frame.
    It converts the frame to an array, processes it with MediaPipe,
    draws landmarks, and then returns the annotated frame.
    """
    # Convert the incoming frame to a numpy array (BGR format)
    image = frame.to_ndarray(format="bgr24")
    
    # Convert BGR to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    
    # Optionally, draw landmarks on the image if they exist
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    
    # Return the annotated frame to the WebRTC streamer
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# -------------------------------
# Working WebRTC Streamer Code
# -------------------------------

if st.session_state.get('action_confirmed', False):
    # Retrieve the persistent streamer key from session state.
    streamer_key = st.session_state['active_streamer_key']
    
    # Inform the user that streaming has been activated.
    st.info(f"Streaming activated! Perform the action: {st.session_state.get('action_word', 'your action')}")
    
    # Launch the WebRTC streamer.
    webrtc_ctx = webrtc_streamer(
        key=streamer_key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": get_ice_servers(),
            "iceTransportPolicy": "relay"
        },
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )
    
    # Check the state of the streamer and update session state accordingly.
    if webrtc_ctx.state.playing:
        st.session_state['streamer_running'] = True
    else:
        # If the streamer was running but now stopped, snapshot the landmark queue.
        if st.session_state.get('streamer_running', False):
            st.session_state['streamer_running'] = False
            st.session_state["landmark_queue_snapshot"] = list(landmark_queue)
            debug_log(f"Snapshot taken with {len(st.session_state['landmark_queue_snapshot'])} frames.")
            st.success("Streaming has stopped. You can now save keyframes.")

