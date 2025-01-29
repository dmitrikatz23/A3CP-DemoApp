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

# ---------------------------------------------------------------------
# SETUP: CSV initialization and queue management
# ---------------------------------------------------------------------

# Initialize queue if not already in session state
if "data_queue" not in st.session_state:
    st.session_state["data_queue"] = Queue(maxsize=2000)  # Increase as needed

# Folder and CSV file initialization
def initialize_csv_folder_and_file():
    """
    Create a 'csv' folder (if it doesn't exist) in the same directory as this script.
    Generate a timestamped CSV file name (only once per session).
    """
    # You can adjust this path logic as needed for your directory structure
    current_folder = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    csv_folder = current_folder / "csv"
    csv_folder.mkdir(exist_ok=True)

    if "csv_file" not in st.session_state:
        session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state["csv_file"] = csv_folder / f"actions_{session_start_str}.csv"

    return st.session_state["csv_file"]

def initialize_csv_file(file_path, header):
    """
    Creates a CSV file with the specified header if it doesn't exist.
    """
    if not file_path.exists():
        with open(file_path, mode="w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
        st.write(f"Created and initialized CSV at {file_path}")
    else:
        st.write(f"CSV file already exists: {file_path}")

def add_to_queue(data):
    """
    Push processed frame data (flattened landmarks, angles, etc.) into the queue.
    """
    if st.session_state["data_queue"].full():
        st.warning("Queue is full. Data not added.")
        return
    st.session_state["data_queue"].put(data)

def write_queue_to_csv(file_path):
    """
    Writes all data in the queue to CSV, then clears the queue.
    """
    if st.session_state["data_queue"].empty():
        st.warning("No data in queue to write.")
        return

    rows = []
    while not st.session_state["data_queue"].empty():
        rows.append(st.session_state["data_queue"].get())

    try:
        with open(file_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)
        st.success(f"Successfully wrote {len(rows)} row(s) to {file_path}")
    except Exception as e:
        st.error(f"Error writing to CSV: {e}")

def display_csv(file_path):
    """Display the entire CSV file in a Streamlit table."""
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                st.info("The CSV file is empty.")
            else:
                st.write("### CSV File Contents")
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    else:
        st.info("CSV file does not exist yet.")

# ---------------------------------------------------------------------
# DEFINE HEADERS & MediaPipe Holistic Setup
# ---------------------------------------------------------------------

# We define the CSV header. This is a sample that includes:
# [class, sequence_id, pose_x1, pose_y1, pose_v1, ..., hand/face data, angles...]
# Adjust as needed based on your model and desired output.

num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468
angle_names_base = [
    "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "little_mcp", "little_pip", "little_dip"
]
left_hand_angle_names = [f"left_{name}" for name in angle_names_base]
right_hand_angle_names = [f"right_{name}" for name in angle_names_base]

pose_landmarks_header = [
    f"pose_{axis}{i}" for i in range(1, num_pose_landmarks + 1)
    for axis in ["x", "y", "v"]
]
left_hand_landmarks_header = [
    f"left_hand_{axis}{i}" for i in range(1, num_hand_landmarks_per_hand + 1)
    for axis in ["x", "y", "v"]
]
right_hand_landmarks_header = [
    f"right_hand_{axis}{i}" for i in range(1, num_hand_landmarks_per_hand + 1)
    for axis in ["x", "y", "v"]
]
face_landmarks_header = [
    f"face_{axis}{i}" for i in range(1, num_face_landmarks + 1)
    for axis in ["x", "y", "v"]
]

csv_header = (
    ["class", "sequence_id"]
    + pose_landmarks_header
    + left_hand_landmarks_header
    + left_hand_angle_names
    + right_hand_landmarks_header
    + right_hand_angle_names
    + face_landmarks_header
)

# Initialize MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_mediapipe_model():
    """Load and cache the MediaPipe Holistic model."""
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

holistic_model = load_mediapipe_model()

# ---------------------------------------------------------------------
# LANDMARK & ANGLE CALCULATIONS
# ---------------------------------------------------------------------

def calculate_angle(a, b, c):
    """
    Calculate the angle formed by points a->b->c (in degrees).
    Each point is (x,y).
    """
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
    Compute angles for each finger joint of the hand.
    Returns angles in a fixed order.
    """
    if not hand_landmarks or all((p[0] == 0 and p[1] == 0) for p in hand_landmarks):
        return [0] * len(angle_names_base)

    # Convert list of (x, y, visibility) to a dict for easy indexing
    h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}
    def pt(i):
        return [h[i][0], h[i][1]]

    # Thumb angles
    thumb_mcp = calculate_angle(pt(1), pt(2), pt(3))
    thumb_ip  = calculate_angle(pt(2), pt(3), pt(4))

    # Index finger angles
    index_mcp = calculate_angle(pt(0), pt(5), pt(6))
    index_pip = calculate_angle(pt(5), pt(6), pt(7))
    index_dip = calculate_angle(pt(6), pt(7), pt(8))

    # Middle finger angles
    middle_mcp = calculate_angle(pt(0), pt(9), pt(10))
    middle_pip = calculate_angle(pt(9), pt(10), pt(11))
    middle_dip = calculate_angle(pt(10), pt(11), pt(12))

    # Ring finger angles
    ring_mcp = calculate_angle(pt(0), pt(13), pt(14))
    ring_pip = calculate_angle(pt(13), pt(14), pt(15))
    ring_dip = calculate_angle(pt(14), pt(15), pt(16))

    # Little finger angles
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

def extract_landmarks(results):
    """
    Extract pose, left hand, right hand, face landmarks from the
    holistic results, each as a list of [x, y, visibility].
    """
    def get_coords(landmarks, count):
        if landmarks is None:
            return [[0, 0, 0]] * count
        return [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark]

    pose_data = get_coords(results.pose_landmarks, num_pose_landmarks)
    left_hand_data = get_coords(results.left_hand_landmarks, num_hand_landmarks_per_hand)
    right_hand_data = get_coords(results.right_hand_landmarks, num_hand_landmarks_per_hand)
    face_data = get_coords(results.face_landmarks, num_face_landmarks)

    return pose_data, left_hand_data, right_hand_data, face_data

def flatten_landmarks(pose_data, lhand_data, lhand_angles, rhand_data, rhand_angles, face_data):
    """
    Flatten all landmarks + angles into a single list, matching the CSV header order.
    """
    # Flatten [x,y,v] for each landmark
    pose_flat = [val for landmark in pose_data for val in landmark]
    lhand_flat = [val for landmark in lhand_data for val in landmark]
    rhand_flat = [val for landmark in rhand_data for val in landmark]
    face_flat = [val for landmark in face_data for val in landmark]

    # Angles are already a flat list
    return pose_flat + lhand_flat + lhand_angles + rhand_flat + rhand_angles + face_flat

# ---------------------------------------------------------------------
# VIDEO FRAME CALLBACK
# ---------------------------------------------------------------------

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This callback is invoked for each frame from the webcam.
    We run the MediaPipe Holistic model, annotate the frame,
    and also store the processed landmark data in the queue.
    """
    input_bgr = frame.to_ndarray(format="bgr24")
    image_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)

    # Annotate frame
    annotated_image = input_bgr.copy()
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    # Extract and flatten data
    pose_data, left_hand_data, right_hand_data, face_data = extract_landmarks(results)
    lhand_angles = hand_angles(left_hand_data)
    rhand_angles = hand_angles(right_hand_data)

    # Each row will have the form [class_label, sequence_id, ...all landmarks...]
    # For now, we place placeholders for action label and sequence_id. We'll fill them in later.
    # Or you can store them immediately if your use case allows.
    action_label = st.session_state.get("action_label", "unconfirmed_action")
    seq_id = 0  # You may store an incremental sequence ID in session_state if you wish

    # Flatten
    row_data = flatten_landmarks(
        pose_data, left_hand_data, lhand_angles, right_hand_data, rhand_angles, face_data
    )
    row_to_queue = [action_label, seq_id] + row_data

    # Add the data row to the queue for later saving
    add_to_queue(row_to_queue)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# ---------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------

st.title("Demo: MediaPipe Recording with Queue-to-CSV")

# 1) CSV Setup
csv_file_path = initialize_csv_folder_and_file()
initialize_csv_file(csv_file_path, csv_header)

# 2) Action Input & Confirmation
action_input = st.text_input("Enter the intended meaning for the action (e.g., I'm hungry)", "")
if st.button("Confirm Action") and action_input.strip():
    # Sanitize the action label
    sanitized_action_label = re.sub(r"[^a-zA-Z0-9_]", "_", action_input.strip())
    st.session_state["action_label"] = sanitized_action_label
    st.success(f"Action label confirmed: {sanitized_action_label}")

st.write("---")

# 3) Start/Stop Video Stream
st.write("### Video Stream")
if "streamer_started" not in st.session_state:
    st.session_state["streamer_started"] = False

if st.button("Start Video Stream"):
    st.session_state["streamer_started"] = True

if st.session_state["streamer_started"]:
    # Start the WebRTC streamer
    webrtc_streamer(
        key="mediapipe-holistic",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback
    )

# 4) When user is done, let them save the queue to CSV
st.write("---")
st.write("### Save Recorded Data to CSV")

# Because we continuously push frames to the queue, you might want to
# handle the notion of 'stop recording' or 'stop streaming' first.
# The user can simply stop the streamer. Then they can click 'Save'.
if st.button("Save Data to CSV"):
    # Write everything from the queue to the CSV
    write_queue_to_csv(csv_file_path)

st.write("---")
st.write("### Current CSV Contents")
display_csv(csv_file_path)
