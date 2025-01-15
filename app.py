import streamlit as st
import csv
import numpy as np
import cv2
import mediapipe as mp
import time
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer

#set the page configuration
st.set_page_config(layout = 'wide')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ---------------------------
# Define the header structure
# ---------------------------
num_pose_landmarks = 33
num_hand_landmarks_per_hand = 21
num_face_landmarks = 468

# Base angle names for one hand
angle_names_base = [
    'thumb_mcp', 'thumb_ip', 
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip', 
    'ring_mcp', 'ring_pip', 'ring_dip', 
    'little_mcp', 'little_pip', 'little_dip'
]

# Generate angle names for left and right hands
left_hand_angle_names = [f'left_{name}' for name in angle_names_base]
right_hand_angle_names = [f'right_{name}' for name in angle_names_base]

# Generate coordinate labels for pose, hands, and face
pose_landmarks = [f'pose_{axis}{i}' for i in range(1, num_pose_landmarks+1) for axis in ['x', 'y', 'v']]
left_hand_landmarks = [f'left_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
right_hand_landmarks = [f'right_hand_{axis}{i}' for i in range(1, num_hand_landmarks_per_hand+1) for axis in ['x', 'y', 'v']]
face_landmarks = [f'face_{axis}{i}' for i in range(1, num_face_landmarks+1) for axis in ['x', 'y', 'v']]

# Assemble the header
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
    """
    Load and cache the MediaPipe Holistic model with optimized settings for video processing.
    - min_detection_confidence: Minimum confidence value for the detection to be considered successful.
    - min_tracking_confidence: Minimum confidence value for the tracking to be considered successful.
    - static_image_mode: Set to False to optimize for video input.
    """
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False  # Optimize for continuous video input
    )

# Load the MediaPipe holistic model once and cache it
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
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ---------------------------
# Processing and Data Helpers
# ---------------------------

def process_frame(frame):
    """Process a single frame with MediaPipe Holistic."""
    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks for visualization
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

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
        # If the hand is not detected, return an array of 0 angles
        if all((p[0] == 0 and p[1] == 0 and p[2] == 0) for p in hand_landmarks):
            return [0] * len(angle_names_base)

        h = {i: hand_landmarks[i] for i in range(len(hand_landmarks))}

        def pt(i):
            return [h[i][0], h[i][1]]

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
        image, 
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
# Create/Initialize CSV
# ---------------------------

# Ensure the "csv" folder exists
csv_folder = "csv"
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Create a unique CSV file name for this session if not already created
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

# Overwrite (or create) the file with a header at the start of each new session
if "csv_initialized" not in st.session_state:
    st.session_state["csv_initialized"] = initialize_csv(csv_file, header)

# ---------------------------
# Streamlit and Logic
# ---------------------------

st.title("A3CP: Personalised Communication Mapping Interface")
st.markdown("Define an action, demonstrate it via webcam, and train a machine learning model.")

# JavaScript to request camera permissions
st.components.v1.html("""
<script>
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
        document.body.innerHTML += "<p style='color: green;'>Camera access granted!</p>";
    })
    .catch(function(err) {
        document.body.innerHTML += "<p style='color: red;'>Camera access denied or unavailable. Please check permissions and try again.</p>";
        console.error("Camera access error:", err);
    });
</script>
""")


#check webcam
def check_camera_access():
    cap = cv2.VideoCapture(0) # attempt to access webcam
    if not cap.isOpened():
        st.error('webcam is not accessible. please chec browser and system permissions')
    else:
        st.success ('webcam is accessible')
        cap.release()

st.title('webcam permission test')

if st.button ('check Webcam'):
    check_camera_access()




# Maintain state
if 'actions' not in st.session_state:
    st.session_state['actions'] = {}
if 'record_started' not in st.session_state:
    st.session_state['record_started'] = False
if 'sequence_id' not in st.session_state:
    st.session_state['sequence_id'] = 0

left_col, right_col = st.columns([1, 2])
FRAME_WINDOW = right_col.image([])
status_bar = right_col.empty()

file_exists = os.path.isfile(csv_file)

with left_col:
    st.header("Controls")
    action_word = st.text_input("Enter the intended meaning for the action e.g. I'm hungry")

    if st.button("Confirm Action") and action_word:
        st.session_state['actions'][action_word] = None
        st.success(f"Action '{action_word}' confirmed!")

    if action_word in st.session_state['actions']:
        if st.button("Start Recording", key=f"start_recording_{action_word}"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam")
            else:
                st.session_state['record_started'] = True
                start_time = time.time()
                all_frames = []
                stop_button_pressed = False

                stop_button = st.button("Stop Recording", key=f"stop_recording_{action_word}")

                while st.session_state['record_started']:
                    elapsed_time = int(time.time() - start_time)
                    status_bar.text(f"Time Elapsed: {elapsed_time} seconds")

                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break

                    # Process each frame
                    (processed_image,
                     pose_data,
                     left_hand_data,
                     left_hand_angles_data,
                     right_hand_data,
                     right_hand_angles_data,
                     face_data) = process_frame(frame)

                    FRAME_WINDOW.image(processed_image, channels="BGR")

                    all_frames.append((
                        pose_data, 
                        left_hand_data, 
                        left_hand_angles_data, 
                        right_hand_data, 
                        right_hand_angles_data, 
                        face_data
                    ))

                    # Stop automatically after 10 seconds or when "Stop Recording" is pressed
                    if stop_button or elapsed_time >= 10:
                        st.session_state['actions'][action_word] = all_frames
                        st.session_state['record_started'] = False
                        stop_button_pressed = True
                        break

                cap.release()
                FRAME_WINDOW.image([])

                if stop_button_pressed:
                    st.success(f"Recording for '{action_word}' saved!")
                    st.info("Camera turned off.")

st.header("Recorded Actions")
if st.session_state['actions']:
    all_rows = []
    for action, all_frames in st.session_state['actions'].items():
        if all_frames is not None and len(all_frames) > 1:
            # Flatten each frame’s landmarks
            flat_landmarks_per_frame = []
            for f in all_frames:
                (pose_data, 
                 left_hand_data, 
                 left_hand_angles_data, 
                 right_hand_data, 
                 right_hand_angles_data, 
                 face_data) = f

                flat_landmarks_per_frame.append(
                    flatten_landmarks(
                        pose_data, 
                        left_hand_data, 
                        left_hand_angles_data,
                        right_hand_data, 
                        right_hand_angles_data, 
                        face_data
                    )
                )

            flat_landmarks_per_frame = np.array(flat_landmarks_per_frame)

            # Identify keyframes based on velocity & acceleration thresholds
            keyframes = identify_keyframes(
                flat_landmarks_per_frame, 
                velocity_threshold=0.1, 
                acceleration_threshold=0.1
            )

            # For each keyframe, store data in the CSV
            for kf in keyframes:
                if kf < len(all_frames):
                    st.session_state['sequence_id'] += 1
                    (pose_data, 
                     left_hand_data, 
                     left_hand_angles_data, 
                     right_hand_data, 
                     right_hand_angles_data, 
                     face_data) = all_frames[kf]
                    
                    row_data = flatten_landmarks(
                        pose_data, 
                        left_hand_data, 
                        left_hand_angles_data, 
                        right_hand_data, 
                        right_hand_angles_data, 
                        face_data
                    )
                    row = [action, st.session_state['sequence_id']] + row_data
                    all_rows.append(row)
                    
    # Append new rows to the CSV file for this session
    if all_rows:
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(all_rows)  # Append rows instead of rewriting the file
        st.success(f"All recorded actions appended to '{csv_file}'")

        # Display a quick summary of recorded actions
        df = pd.read_csv(csv_file)

        unique_actions = df['class'].unique()
        num_actions = len(unique_actions)
        num_cols = 8
        num_rows = (num_actions + num_cols - 1) // num_cols

        # Display actions in an 8-across grid
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
        st.warning("No keyframes were identified. Try recording a clearer gesture.")
else:
    st.info("No actions recorded yet.")
