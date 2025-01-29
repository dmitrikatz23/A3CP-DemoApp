import logging
from pathlib import Path
import re
import csv
import time
import pandas as pd
import os
from datetime import datetime
from queue import Queue

import mediapipe as mp
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# sample_utils imports (if needed for ICE servers)
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

# Optional: Hugging Face
from huggingface_hub import Repository

# ------------------------------------------------
# Configure Logging
# ------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("twilio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("Holistic Demo with Queuing & CSV Saving")

# ------------------------------------------------
# 1) Initialize Session State
# ------------------------------------------------
# We store our queue in session state so Streamlit doesn't recreate it on each rerun.
if "frame_queue" not in st.session_state:
    st.session_state.frame_queue = Queue()

if "actions" not in st.session_state:
    st.session_state.actions = {}

if "sequence_id" not in st.session_state:
    st.session_state.sequence_id = 0

if "action_confirmed" not in st.session_state:
    st.session_state.action_confirmed = False

if "active_streamer_key" not in st.session_state:
    st.session_state.active_streamer_key = None

# ------------------------------------------------
# 2) CSV Setup
# ------------------------------------------------
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
    st.write(f"Created CSV folder: {csv_folder}")

# If a CSV file hasn't been set in session state, create one with a timestamped name
if "csv_file" not in st.session_state:
    session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.session_state.csv_file = os.path.join(csv_folder, f"all_actions_recorded_{session_start_str}.csv")

csv_file = st.session_state.csv_file

# ------------------------------------------------
# 3) Generate CSV Header
# ------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

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

def initialize_csv(file_name, header):
    """
    Initialize the CSV file with the header row if it doesn't exist.
    """
    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
        st.write(f"CSV initialized at {file_name} with header.")
    else:
        st.write(f"CSV file already exists: {file_name}")

if "csv_initialized" not in st.session_state:
    initialize_csv(csv_file, header)
    st.session_state.csv_initialized = True

# ------------------------------------------------
# 4) MediaPipe Model Loader
# ------------------------------------------------
@st.cache_resource
def load_mediapipe_model():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

holistic_model = load_mediapipe_model()

# ------------------------------------------------
# 5) Helper Functions
# ------------------------------------------------
def validate_frame_data(frame_data):
    """
    Ensure frame_data has all required keys and that they are not empty.
    """
    required_keys = [
        "pose_data",
        "left_hand_data",
        "left_hand_angles_data",
        "right_hand_data",
        "right_hand_angles_data",
        "face_data"
    ]
    if not isinstance(frame_data, dict):
        return False, "frame_data is not a dictionary"

    for key in required_keys:
        if key not in frame_data or not frame_data[key]:
            return False, key  # Missing or empty key

    return True, None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def hand_angles(hand_landmarks):
    """
    Calculate angles for each finger joint using hand landmarks.
    """
    if not hand_landmarks or all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)

    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i): return [h[i][0], h[i][1]]

    # Thumb angles
    thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
    thumb_ip  = calculate_angle(pt(2), pt(3), pt(4))

    # Index finger
    index_mcp = calculate_angle(pt(0), pt(5), pt(6))
    index_pip = calculate_angle(pt(5), pt(6), pt(7))
    index_dip = calculate_angle(pt(6), pt(7), pt(8))

    # Middle finger
    middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
    middle_pip = calculate_angle(pt(9), pt(10), pt(11))
    middle_dip = calculate_angle(pt(10), pt(11), pt(12))

    # Ring finger
    ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
    ring_pip = calculate_angle(pt(13), pt(14), pt(15))
    ring_dip = calculate_angle(pt(14), pt(15), pt(16))

    # Little finger
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

def process_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks if available
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    def extract_data(landmarks, count):
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark] if landmarks else [[0, 0, 0]] * count

    pose_data = extract_data(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = extract_data(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = extract_data(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = extract_data(results.face_landmarks, num_face_landmarks)

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
    pose_flat = [val for landmark in pose_data for val in landmark]
    left_hand_flat = [val for landmark in left_hand_data for val in landmark]
    right_hand_flat = [val for landmark in right_hand_data for val in landmark]
    left_hand_angles_flat = left_hand_angles_data
    right_hand_angles_flat = right_hand_angles_data
    face_flat = [val for landmark in face_data for val in landmark]

    return (
        pose_flat +
        left_hand_flat +
        left_hand_angles_flat +
        right_hand_flat +
        right_hand_angles_flat +
        face_flat
    )

# ------------------------------------------------
# 6) Video Frame Callback
# ------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    input_bgr = frame.to_ndarray(format="bgr24")

    (annotated_image,
     pose_data,
     left_hand_data,
     left_hand_angles_data,
     right_hand_data,
     right_hand_angles_data,
     face_data) = process_frame(input_bgr)

    frame_data = {
        "pose_data": pose_data,
        "left_hand_data": left_hand_data,
        "left_hand_angles_data": left_hand_angles_data,
        "right_hand_data": right_hand_data,
        "right_hand_angles_data": right_hand_angles_data,
        "face_data": face_data,
    }

    # Enqueue frame_data into session state queue
    st.session_state.frame_queue.put(frame_data)
    logging.debug(f"Frame added to queue. Queue size: {st.session_state.frame_queue.qsize()}")

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# ------------------------------------------------
# 7) Hugging Face Integration (Optional)
# ------------------------------------------------
hf_token = os.getenv("Recorded_Datasets")  
repo = None
if hf_token:
    try:
        repo_name = "dk23/A3CP_actions"
        local_repo_path = "local_repo"
        git_user = "A3CP_bot"
        git_email = "no-reply@huggingface.co"

        repo = Repository(
            local_dir=local_repo_path,
            clone_from=repo_name,
            use_auth_token=hf_token,
            repo_type="dataset"
        )
        repo.git_config_username_and_email(git_user, git_email)
    except Exception as e:
        st.error(f"Error setting up Hugging Face repository: {e}")
else:
    st.warning("Hugging Face token not found. CSV will only be saved locally.")

def save_csv_to_huggingface():
    if not repo:
        st.info("No Hugging Face repository configured; skipping push.")
        return
    if not os.path.exists(csv_file):
        st.warning("No CSV file found to save.")
        return

    try:
        df = pd.read_csv(csv_file)
        os.makedirs(repo.local_dir, exist_ok=True)
        csv_repo_path = os.path.join(repo.local_dir, os.path.basename(csv_file))
        df.to_csv(csv_repo_path, index=False)

        repo.git_add(csv_file)
        repo.git_commit("Update A3CP actions CSV")
        repo.git_push()
        st.success(f"CSV successfully pushed to Hugging Face repository: {repo_name}")
    except Exception as ex:
        st.error(f"Error saving to repository: {ex}")

# ------------------------------------------------
# 8) Streamlit UI and Logic
# ------------------------------------------------
st.subheader("WebRTC Stream")

# Controls Column
left_col, right_col = st.columns([1, 2])
with left_col:
    st.header("Controls")
    action_word = st.text_input("Enter the intended meaning for the action (e.g., I'm hungry)")

    # Confirm Action button
    if st.button("Confirm Action") and action_word:
        sanitized_action_word = re.sub(r'[^a-zA-Z0-9_]', '_', action_word.strip())

        # Reset or clear old streamer, if any
        if st.session_state.get('active_streamer_key') is not None:
            st.session_state['action_confirmed'] = False
            old_key = st.session_state['active_streamer_key']
            if old_key in st.session_state:
                del st.session_state[old_key]

        # Prepare for a new action
        st.session_state['current_action'] = action_word
        st.session_state['actions'][action_word] = []
        st.session_state['action_confirmed'] = True
        st.session_state['active_streamer_key'] = f"record-actions-{sanitized_action_word}"
        st.success(f"Action '{action_word}' confirmed!")

    # Start the WebRTC streamer if action confirmed
    if st.session_state.get('action_confirmed', False):
        streamer_key = st.session_state['active_streamer_key']
        st.info(f"Streaming activated! Perform the action: {action_word}")

        webrtc_streamer(
            key=streamer_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers(), "iceTransportPolicy": "relay"},
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )

# Main display areas
FRAME_WINDOW = right_col.image([])
status_bar = right_col.empty()

st.subheader("Process & Save Frames")

# ------------------------------------------------
# Buttons to Process & Save
# ------------------------------------------------
if st.button("Process Frames"):
    """
    Move frames from st.session_state.frame_queue into st.session_state.actions
    for the currently confirmed action.
    """
    qsize = st.session_state.frame_queue.qsize()
    st.write(f"Queue size before processing: {qsize}")
    logging.debug(f"[Process Frames] Queue size: {qsize}")

    while not st.session_state.frame_queue.empty():
        frame_data = st.session_state.frame_queue.get()

        # Validate frame
        is_valid, missing_key = validate_frame_data(frame_data)
        if not is_valid:
            logging.warning(f"Skipping invalid frame. Missing or empty key: {missing_key}")
            continue

        if st.session_state.get("action_confirmed") and st.session_state.get("current_action"):
            action_word = st.session_state["current_action"]
            st.session_state["actions"][action_word].append(frame_data)

    if st.session_state.get("current_action"):
        final_count = len(st.session_state["actions"][st.session_state["current_action"]])
        st.write(f"Processed frames. Total frames for action '{st.session_state['current_action']}': {final_count}")

if st.button("Save to CSV"):
    """
    Flatten all stored frames in st.session_state['actions'] into rows 
    and write them to the CSV file.
    """
    all_rows = []

    # Loop over each action and each frame
    for action, frames in st.session_state["actions"].items():
        if frames:
            for index, frame_data in enumerate(frames):
                try:
                    row_data = flatten_landmarks(
                        frame_data["pose_data"],
                        frame_data["left_hand_data"],
                        frame_data["left_hand_angles_data"],
                        frame_data["right_hand_data"],
                        frame_data["right_hand_angles_data"],
                        frame_data["face_data"]
                    )
                    st.session_state["sequence_id"] += 1
                    row = [action, st.session_state["sequence_id"]] + row_data
                    all_rows.append(row)
                except Exception as e:
                    logging.error(f"Error flattening frame {index} for action '{action}': {e}")

    if all_rows:
        # Write to CSV
        try:
            with open(csv_file, mode="a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(all_rows)
            st.success(f"Wrote {len(all_rows)} rows to CSV: {csv_file}")
            logging.info(f"[Save to CSV] Successfully wrote {len(all_rows)} rows.")
        except Exception as e:
            st.error(f"Error writing to CSV: {e}")

        # Optionally push to Hugging Face
        if all_rows:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    save_csv_to_huggingface()
            except Exception as e:
                st.error(f"Error reading CSV for repository push: {e}")
    else:
        st.warning("No rows to write to CSV.")

# ------------------------------------------------
# Display Recorded CSV
# ------------------------------------------------
st.subheader("Recorded Actions Summary (Current CSV)")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if not df.empty:
        # Display unique actions
        unique_actions = df['class'].unique()
        st.write(f"**Unique Actions in CSV**: {unique_actions}")

        # Display the entire CSV for reference
        st.subheader("Full CSV Data")
        st.dataframe(df.reset_index(drop=True))
    else:
        st.info("CSV is initialized but has no data rows yet.")
else:
    st.info("No CSV file found yet.")
