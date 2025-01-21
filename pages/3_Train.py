import streamlit as st
import csv
import numpy as np
import cv2
import mediapipe as mp
import time
import pandas as pd
import os
from datetime import datetime

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av

# ---------------------------
# Additional Hugging Face Imports (Optional)
# ---------------------------
from huggingface_hub import Repository

st.set_page_config(layout="wide")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ---------------------------
# Define the header structure
# ---------------------------
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468

angle_names_base = [
    'thumb_mcp', 'thumb_ip', 
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip', 
    'ring_mcp', 'ring_pip', 'ring_dip', 
    'little_mcp', 'little_pip', 'little_dip'
]

left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]

pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]

@st.cache_data
def load_csv_header():
    return (
        ['class', 'sequence_id'] 
        + pose_landmarks 
        + left_hand_landmarks 
        + left_hand_angle_names 
        + right_hand_landmarks 
        + right_hand_angle_names 
        + face_landmarks
    )

header = load_csv_header()

# ---------------------------
# Mediapipe & Angle Functions
# ---------------------------
@st.cache_resource
def load_mediapipe_model():
    """Load and cache the MediaPipe Holistic model with optimized settings."""
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_model()

def calculate_velocity(landmarks):
    """Calculate Euclidean velocity between consecutive frames."""
    velocities = []
    for i in range(1, len(landmarks)):
        velocity = np.linalg.norm(landmarks[i] - landmarks[i-1])
        velocities.append(velocity)
    return np.array(velocities)

def calculate_acceleration(velocities):
    """Calculate absolute acceleration between consecutive velocity frames."""
    accelerations = []
    for i in range(1, len(velocities)):
        acceleration = np.abs(velocities[i] - velocities[i-1])
        accelerations.append(acceleration)
    return np.array(accelerations)

def identify_keyframes(landmarks, velocity_threshold=0.1, acceleration_threshold=0.1):
    """
    Identify frames where velocity or acceleration exceed given thresholds.
    Returns a list of frame indices considered keyframes.
    """
    velocities = calculate_velocity(landmarks)
    accelerations = calculate_acceleration(velocities)
    keyframes = []
    for i in range(len(accelerations)):
        if velocities[i] > velocity_threshold or accelerations[i] > acceleration_threshold:
            keyframes.append(i + 1)  # +1 because acceleration index starts from 1
    return keyframes

def calculate_angle(a, b, c):
    """Calculate angle (in degrees) at joint b formed by points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ---------------------------
# Processing and Data Helpers
# ---------------------------
def process_frame(frame_bgr):
    """Process a single frame with MediaPipe Holistic, returning landmarks + an annotated image."""
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks for visualization
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Extract pose data
    if results.pose_landmarks:
        pose_data = [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
    else:
        pose_data = [[0, 0, 0]] * num_pose_landmarks

    # Extract left and right hand data
    def extract_hand_data(hand_landmarks):
        if hand_landmarks:
            return [[lm.x, lm.y, lm.visibility] for lm in hand_landmarks.landmark]
        else:
            return [[0, 0, 0]] * num_hand_landmarks_per_hand

    left_hand_data = extract_hand_data(results.left_hand_landmarks)
    right_hand_data = extract_hand_data(results.right_hand_landmarks)

    # Extract face data
    if results.face_landmarks:
        face_data = [[lm.x, lm.y, lm.visibility] for lm in results.face_landmarks.landmark]
    else:
        face_data = [[0, 0, 0]] * num_face_landmarks

    # Calculate hand angles
    def hand_angles(hand_landmarks):
        if all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
            return [0] * len(angle_names_base)

        h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
        def pt(i): return [h[i][0], h[i][1]]

        thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
        thumb_ip = calculate_angle(pt(2), pt(3), pt(4))

        index_mcp = calculate_angle(pt(0), pt(5), pt(6))
        index_pip = calculate_angle(pt(5), pt(6), pt(7))
        index_dip = calculate_angle(pt(6), pt(7), pt(8))

        middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
        middle_pip = calculate_angle(pt(9), pt(10), pt(11))
        middle_dip = calculate_angle(pt(10), pt(11), pt(12))

        ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
        ring_pip = calculate_angle(pt(13), pt(14), pt(15))
        ring_dip = calculate_angle(pt(14), pt(15), pt(16))

        little_mcp = calculate_angle(pt(0), pt(17), pt(18))
        little_pip = calculate_angle(pt(17), pt(18), pt(19))
        little_dip = calculate_angle(pt(18), pt(19), pt(20))

        return [
            thumb_mcp, thumb_ip,
            index_mcp, index_pip, index_dip,
            middle_mcp, middle_pip, middle_dip,
            ring_mcp, ring_pip, ring_dip,
            little_mcp, little_pip, little_dip
        ]

    left_hand_angles_data = hand_angles(left_hand_data)
    right_hand_angles_data = hand_angles(right_hand_data)

    return (
        annotated_image,
        pose_data,
        left_hand_data,
        left_hand_angles_data,
        right_hand_data,
        right_hand_angles_data,
        face_data
    )

def flatten_landmarks(
    pose_data, 
    left_hand_data, 
    left_hand_angles_data, 
    right_hand_data, 
    right_hand_angles_data, 
    face_data
):
    """Flatten all landmark data + angles into a single 1D list."""
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    left_hand_angles_flat = left_hand_angles_data
    right_hand_angles_flat = right_hand_angles_data
    face_flat = [val for landmark in face_data for val in landmark]

    return (
        pose_flat 
        + left_hand_flat 
        + left_hand_angles_flat 
        + right_hand_flat 
        + right_hand_angles_flat 
        + face_flat
    )

# ---------------------------
# CSV Initialization
# ---------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state["csv_file"] = os.path.join(csv_folder, f"all_actions_recorded_{session_start_str}.csv")

csv_file = st.session_state["csv_file"]

@st.cache_data
def initialize_csv(file_name, header):
    """
    Overwrites (or creates) the CSV file with a new header 
    at the start of the session.
    """
    with open(file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
    return True

if "csv_initialized" not in st.session_state:
    st.session_state["csv_initialized"] = initialize_csv(csv_file, header)

# ---------------------------
# WebRTC VideoProcessor
# ---------------------------
class HolisticVideoProcessor(VideoProcessorBase):
    """
    Custom processor for streamlit-webrtc that collects frames and 
    processes them with MediaPipe Holistic.
    """
    def __init__(self):
        self.collected_frames = []
        self.recording = False

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        (
            annotated_image,
            pose_data,
            left_hand_data,
            left_hand_angles_data,
            right_hand_data,
            right_hand_angles_data,
            face_data
        ) = process_frame(frame_bgr)

        # If currently recording, store the results
        if self.recording:
            self.collected_frames.append((
                pose_data,
                left_hand_data,
                left_hand_angles_data,
                right_hand_data,
                right_hand_angles_data,
                face_data
            ))

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# ---------------------------
# Hugging Face Integration (Optional)
# ---------------------------
hf_token = os.getenv("Recorded_Datasets")  # If set in your Space secrets
repo = None
if hf_token:
    try:
        repo_name = "dk23/A3CP_actions"
        local_repo_path = "local_repo"
        git_user = "A3CP_bot"
        git_email = "no-reply@huggingface.co"
        repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")
        repo.git_config_username_and_email(git_user, git_email)
    except Exception as e:
        st.error(f"Error setting up Hugging Face repository: {e}")

def push_to_huggingface():
    """Commit/push CSV to HF if repo is set up."""
    if not repo:
        st.warning("No Hugging Face repo configured. Skipping push.")
        return
    if not os.path.exists(csv_file):
        st.warning("No CSV file found to push.")
        return

    try:
        df = pd.read_csv(csv_file)
        os.makedirs(repo.local_dir, exist_ok=True)
        repo_csv_path = os.path.join(repo.local_dir, os.path.basename(csv_file))
        df.to_csv(repo_csv_path, index=False)

        repo.git_add(os.path.basename(csv_file))
        repo.git_commit("Update A3CP actions CSV")
        repo.git_push()
        st.success(f"CSV pushed to {repo.local_dir} -> {repo_name}")
    except Exception as ex:
        st.error(f"Failed to push to Hugging Face: {ex}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("A3CP: Recording Actions")

# Maintain session state
if "sequence_id" not in st.session_state:
    st.session_state["sequence_id"] = 0
if "actions" not in st.session_state:
    st.session_state["actions"] = {}

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Controls")
    action_word = st.text_input("Enter an action meaning (e.g. 'I'm hungry')")

    # Step 1: Confirm action
    if st.button("Confirm Action") and action_word:
        st.session_state["actions"][action_word] = []
        st.success(f"Action '{action_word}' confirmed!")

    # Step 2: Initialize WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="holistic-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HolisticVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
    processor = webrtc_ctx.video_processor

    if processor and action_word:
        # Step 3: Start/Stop recording
        if st.button("Start Recording"):
            processor.recording = True
            processor.collected_frames = []
            st.success("Recording started.")

        if st.button("Stop Recording"):
            processor.recording = False
            all_frames = processor.collected_frames
            st.success(f"Recording for '{action_word}' completed with {len(all_frames)} frames.")
            # Save frames to session state for this action
            st.session_state["actions"][action_word] = all_frames

    # Step 4: Push to Hugging Face if desired
    if st.button("Push CSV to Hugging Face"):
        push_to_huggingface()

with col_right:
    st.header("Recorded Actions / Keyframes")

# If we have recorded frames for an action, identify keyframes & write CSV
all_rows = []
for action, frames in st.session_state["actions"].items():
    if frames and len(frames) > 1:
        # Flatten each frame’s landmarks
        flat_landmarks_per_frame = []
        for f in frames:
            (pose_data, 
             left_hand_data, 
             left_hand_angles_data, 
             right_hand_data, 
             right_hand_angles_data, 
             face_data) = f
            flattened = flatten_landmarks(
                pose_data,
                left_hand_data,
                left_hand_angles_data,
                right_hand_data,
                right_hand_angles_data,
                face_data
            )
            flat_landmarks_per_frame.append(flattened)

        # Identify keyframes
        flat_landmarks_per_frame = np.array(flat_landmarks_per_frame)
        keyframes = identify_keyframes(
            flat_landmarks_per_frame, 
            velocity_threshold=0.1, 
            acceleration_threshold=0.1
        )

        # For each keyframe, store data in CSV
        for kf in keyframes:
            if kf < len(frames):
                st.session_state["sequence_id"] += 1
                (pose_data, 
                 left_hand_data, 
                 left_hand_angles_data, 
                 right_hand_data, 
                 right_hand_angles_data, 
                 face_data) = frames[kf]

                row_data = flatten_landmarks(
                    pose_data, 
                    left_hand_data, 
                    left_hand_angles_data, 
                    right_hand_data, 
                    right_hand_angles_data, 
                    face_data
                )
                row = [action, st.session_state["sequence_id"]] + row_data
                all_rows.append(row)

if all_rows:
    # Append new rows to the CSV for this session
    with open(csv_file, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(all_rows)
    st.success(f"All keyframes appended to '{csv_file}'")

    # Show summary of recorded classes
    df = pd.read_csv(csv_file)
    unique_actions = df['class'].unique()
    num_actions = len(unique_actions)
    num_cols = 8
    num_rows = (num_actions + num_cols - 1) // num_cols

    for r in range(num_rows):
        row_actions = unique_actions[r*num_cols:(r+1)*num_cols]
        cols = st.columns(num_cols)
        for col, a in zip(cols, row_actions):
            if a:
                col.markdown(
                    f"<h4 style='margin:10px; text-align:center; font-family:sans-serif;'>{a}</h4>",
                    unsafe_allow_html=True
                )
else:
    st.info("No keyframes identified or no actions recorded yet.")
