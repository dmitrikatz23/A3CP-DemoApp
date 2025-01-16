import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ..sample_utils.download import download_file
from ..sample_utils.turn import get_ice_servers

# Logging setup
logger = logging.getLogger(__name__)

# Streamlit setup
st.set_page_config(page_title="AI Squat Detection", page_icon="🏋️")
st.markdown(
    """<style>
    .status-box {
        background: #f7f7f7;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        font-size: 18px;
    }
    .title {
        color: #2E86C1;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .info {
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
        color: #333;
    }
    </style>""", unsafe_allow_html=True)

st.markdown('<div class="title">AI Squat Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="info">Use your webcam for real-time squat detection.</div>', unsafe_allow_html=True)

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Detection Queue
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# Initialize MediaPipe Pose once
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global counterL, correct, incorrect, stage
    if 'stage' not in globals():
        stage = 'up'
        correct = 0
        incorrect = 0

    image = frame.to_ndarray(format="bgr24")
    # Mirror the image horizontally
    image = cv2.flip(image, 1)  # Flip code 1 means horizontal flip
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []

    detections = [
        Detection(
            class_id=0, label="Pose", score=0.5, box=np.array([0, 0, w, h])
        )
    ] if landmarks else []

    if landmarks:
        hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        angleKneeL = calculate_angle(hipL, kneeL, ankleL)
        angleHipL = calculate_angle(shoulderL, hipL, [hipL[0], 0])

        rel_point1 = (int(w * 0), int(h - h * 0.65))
        rel_point2 = (int(w * 0.17), int(h - h * 0.65))

        cv2.rectangle(image, (0, 90), (200, 175), (127, 248, 236), -1)
        cv2.rectangle(image, (0, 93), (197, 173), (12, 85, 61), -1)
        cv2.putText(image, 'HipL', (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'KneeL', (125, 122),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angleHipL)), rel_point1, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angleKneeL)), rel_point2, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        if angleKneeL > 110 and stage == 'down':
            stage = 'up'
            if 18 < angleHipL < 40:
                correct += 1

        if angleKneeL < 110 and stage == 'up':
            stage = 'down'

    cv2.rectangle(image, (0, 0), (200, 83), (127, 248, 236), -1)
    cv2.rectangle(image, (0, 3), (197, 80), (12, 85, 61), -1)

    cv2.putText(image, 'Left', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(correct), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'STAGE', (110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, stage, (77, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 175, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=2)
    )

    result_queue.put(detections)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="squat-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

