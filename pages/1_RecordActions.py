import logging
import queue
from typing import List, NamedTuple
import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from sample_utils.turn import get_ice_servers

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert the input video frame to a NumPy array
    img = frame.to_ndarray(format="bgr24")

    # Flip the image horizontally for a mirrored view
    img = cv2.flip(img, 1)

    # Return the processed frame
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Streamer
webrtc_streamer(
    key="record_actions",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)
