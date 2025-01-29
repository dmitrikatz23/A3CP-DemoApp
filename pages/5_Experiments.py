import streamlit as st
import cv2
import av
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import os
from pathlib import Path
from datetime import datetime
from queue import Queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import re

################################################################################
# 1) Session State Setup: Queue & CSV Filename
################################################################################

# Initialize the queue in session state once
if "data_queue" not in st.session_state:
    st.session_state["data_queue"] = Queue(maxsize=2000)

if "action_label" not in st.session_state:
    st.session_state["action_label"] = "unlabeled_action"

if "sequence_id" not in st.session_state:
    st.session_state["sequence_id"] = 0

# Create a CSV folder (if it doesn't exist) and define a timestamped CSV path
def init_csv_path():
    """
    Ensure there's a 'csv' folder in the current directory, and create a
    unique timestamped CSV file name if one doesn't exist in session_state.
    """
    current_folder = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    csv_folder = current_folder / "csv"
    csv_folder.mkdir(exist_ok=True)

    if "csv_file" not in st.session_state:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state["csv_file"] = csv_folder / f"actions_{timestamp_str}.csv"

    return st.session_state["csv_file"]

csv_file_path = init_csv_path()

################################################################################
# 2) CSV Header Definition & Initialization
################################################################################

# Set constants for number of landmarks
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468

# Define angle names
angle_names_base = [
    "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "little_mcp", "little_pip", "little_dip"
]
left_hand_angle_names = [f"left_{name}" for name in angle_names_base]
right_hand_angle_names = [f"right_{name}" for name in angle_names_base]

# Build the CSV column headers
pose_headers = [f"pose_{axis}{i}" for i in range(1, NUM_POSE_LANDMARKS + 1) for axis in ["x", "y", "v"]]
left_hand_headers = [f"left_hand_{axis}{i}" for i in range(1, NUM_HAND_LANDMARKS + 1) for axis in ["x", "y", "v"]]
right_hand_headers = [f"right_hand_{axis}{i}" for i in range(1, NUM_HAND_LANDMARKS + 1) for axis in ["x", "y", "v"]]
face_headers = [f"face_{axis}{i}" for i in range(1, NUM_FACE_LANDMARKS + 1) for axis in ["x", "y", "v"]]

CSV_HEADER = (
    ["class", "sequence_id"]
    + pose_headers
    + left_hand_headers
    + left_hand_angle_names
    + right_hand_headers
    + right_hand_angle_names
    + face_headers
)

def initialize_csv_if_needed(file_path, header):
    """
    If the CSV doesn't exist, create it and write the header row.
    """
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
        st.write(f"Created CSV with header at: {file_path}")
    else:
        st.write(f"CSV file already exists: {file_path}")

initialize_csv_if_needed(csv_file_path, CSV_HEADER)

################################################################################
# 3) MediaPipe Holistic Setup
################################################################################

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_mediapipe_model():
    """Load the Holistic model once and reuse."""
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

holistic_model = load_mediapipe_model()

################################################################################
# 4) Landmark & Angle Calculation Helpers
################################################################################

def calculate_angle(a, b, c):
    """
    Calculate the angle in degrees formed at point b, using the points:
       a -> b -> c
    Each point is a 2D (x,y).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def compute_hand_angles(hand_landmarks):
    """
    Given a list of 21 landmarks [(x,y,v), (x,y,v), ...],
    compute the angles for each finger joint.
    Returns a list of angles in the order of angle_names_base.
    """
    if (not hand_landmarks) or all((p[0] == 0 and p[1] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)

    # Make a dict for easy referencing
    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}

    # Helper to get (x,y) only
    def xy(i):
        return [h[i][0], h[i][1]]

    # Thumb
    thumb_mcp = calculate_angle(xy(1), xy(2), xy(3))
    thumb_ip  = calculate_angle(xy(2), xy(3), xy(4))

    # Index
    index_mcp = calculate_angle(xy(0), xy(5), xy(6))
    index_pip = calculate_angle(xy(5), xy(6), xy(7))
    index_dip = calculate_angle(xy(6), xy(7), xy(8))

    # Middle
    middle_mcp = calculate_angle(xy(0), xy(9), xy(10))
    middle_pip = calculate_angle(xy(9), xy(10), xy(11))
    middle_dip = calculate_angle(xy(10), xy(11), xy(12))

    # Ring
    ring_mcp = calculate_angle(xy(0), xy(13), xy(14))
    ring_pip = calculate_angle(xy(13), xy(14), xy(15))
    ring_dip = calculate_angle(xy(14), xy(15), xy(16))

    # Little
    little_mcp = calculate_angle(xy(0), xy(17), xy(18))
    little_pip = calculate_angle(xy(17), xy(18), xy(19))
    little_dip = calculate_angle(xy(18), xy(19), xy(20))

    return [
        thumb_mcp, thumb_ip,
        index_mcp, index_pip, index_dip,
        middle_mcp, middle_pip, middle_dip,
        ring_mcp, ring_pip, ring_dip,
        little_mcp, little_pip, little_dip
    ]

def flatten_landmarks(pose_data, lhand_data, lhand_angles,
                      rhand_data, rhand_angles, face_data):
    """
    Flatten all data into a single list, matching CSV_HEADER order.
    """
    pose_flat = [val for lm in pose_data for val in lm]  # each lm: [x,y,v]
    lhand_flat = [val for lm in lhand_data for val in lm]
    rhand_flat = [val for lm in rhand_data for val in lm]
    face_flat = [val for lm in face_data for val in lm]
    return pose_flat + lhand_flat + lhand_angles + rhand_flat + rhand_angles + face_flat

################################################################################
# 5) Video Frame Callback
################################################################################

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This function is called on each video frame from the webcam.
    We'll run MediaPipe Holistic, annotate the frame, and queue
    the flattened landmark data.
    """
    # Convert to BGR (as typically used by OpenCV)
    frame_bgr = frame.to_ndarray(format="bgr24")

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(frame_rgb)

    # Draw annotations on a copy
    annotated_image = frame_bgr.copy()

    # Draw the landmarks if present
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    # Extract raw landmarks as (x,y,visibility)
    def extract_landmark_data(landmark_list, total_count):
        if landmark_list is None:
            return [[0, 0, 0]] * total_count
        return [
            [lm.x, lm.y, lm.visibility] for lm in landmark_list.landmark
        ]

    pose_data = extract_landmark_data(results.pose_landmarks, NUM_POSE_LANDMARKS)
    left_hand_data = extract_landmark_data(results.left_hand_landmarks, NUM_HAND_LANDMARKS)
    right_hand_data = extract_landmark_data(results.right_hand_landmarks, NUM_HAND_LANDMARKS)
    face_data = extract_landmark_data(results.face_landmarks, NUM_FACE_LANDMARKS)

    # Compute finger joint angles for each hand
    left_hand_angles = compute_hand_angles(left_hand_data)
    right_hand_angles = compute_hand_angles(right_hand_data)

    # Flatten all landmarks
    row_data = flatten_landmarks(
        pose_data,
        left_hand_data,
        left_hand_angles,
        right_hand_data,
        right_hand_angles,
        face_data
    )

    # Insert class label & sequence_id at the front
    row = [
        st.session_state["action_label"],
        st.session_state["sequence_id"]  # could increment per-frame or per-action
    ] + row_data

    # Push this row into the queue
    try:
        st.session_state["data_queue"].put_nowait(row)
    except:
        # If queue is full, skip
        pass

    # Return the annotated frame
    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

################################################################################
# 6) Streamlit UI
################################################################################
st.title("MediaPipe Holistic Example with Queue-to-CSV")

# ----- Action input -----
st.subheader("1. Enter and Confirm Your Action Label")
user_input = st.text_input("Enter your action label (e.g., 'hungry')", "")
if st.button("Confirm Action") and user_input.strip():
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", user_input.strip())
    st.session_state["action_label"] = sanitized
    st.success(f"Action label set to: {sanitized}")

# ----- Start the video stream -----
st.subheader("2. Start Video Stream")
st.write("Click the button below to start the webcam feed.")

webrtc_ctx = webrtc_streamer(
    key="holistic-webrtc",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
)

# ----- Save data to CSV -----
st.subheader("3. Save Recorded Data to CSV")
st.write("When you're done recording, click the button below to save all queued data to the CSV.")

if st.button("Save Data to CSV"):
    # Pull everything from the queue
    rows_to_write = []
    while not st.session_state["data_queue"].empty():
        rows_to_write.append(st.session_state["data_queue"].get())

    if rows_to_write:
        # Increment the sequence_id if you want a unique ID each time you press 'Save'
        # or each row. This is optional. Example:
        for row in rows_to_write:
            # row[1] is sequence_id in the current code
            # If you'd like each "save" to jump the sequence ID:
            st.session_state["sequence_id"] += 1
            row[1] = st.session_state["sequence_id"]

        try:
            # Append to CSV
            with open(csv_file_path, mode="a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(rows_to_write)

            st.success(f"Wrote {len(rows_to_write)} rows to {csv_file_path}")
        except Exception as e:
            st.error(f"Error writing to CSV: {e}")
    else:
        st.warning("No data in the queue to write.")

# ----- Display CSV contents -----
st.subheader("4. Current CSV Contents")
if os.path.exists(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        st.write(df)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.info("CSV file does not exist yet. Start streaming and save data first.")
