import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from huggingface_hub import HfApi
import matplotlib.animation as animation

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(layout="wide", page_title="Dataset Visualizer")

# Hugging Face repository details
HF_REPO_NAME = "dk23/A3CP_actions"
LOCAL_DATASET_DIR = "local_repo"

# Load Hugging Face token from environment variables
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

# Initialize Hugging Face API
hf_api = HfApi()

# -----------------------------------
# Fetch Dataset List from Hugging Face
# -----------------------------------
#@st.cache_data
def fetch_datasets():
    """Retrieve dataset filenames from Hugging Face repository."""
    try:
        files = hf_api.list_repo_files(HF_REPO_NAME, repo_type="dataset", token=hf_token)
        csv_files = sorted([f for f in files if f.endswith(".csv")], reverse=True)
        return csv_files
    except Exception as e:
        st.error(f"Failed to fetch datasets: {e}")
        return []

# Fetch dataset list
dataset_files = fetch_datasets()

# -----------------------------------
# Left Column: Dataset Selector
# -----------------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Available Datasets")
    
    if dataset_files:
        selected_dataset = st.selectbox("Select a dataset to visualize:", dataset_files)

        # Download the selected dataset if it does not exist locally
        dataset_path = os.path.join(LOCAL_DATASET_DIR, selected_dataset)
        if not os.path.exists(dataset_path):
            with st.spinner(f"Downloading {selected_dataset}..."):
                hf_api.hf_hub_download(HF_REPO_NAME, selected_dataset, local_dir=LOCAL_DATASET_DIR, repo_type="dataset", token=hf_token)

        st.success(f"Selected dataset: {selected_dataset}")

    else:
        st.warning("No datasets found in the repository.")
        st.stop()

# -----------------------------------
# Right Column: Visualization
# -----------------------------------
def animate_landmarks(data, save_path, frame_skip=2):
    """Generate an animation and save as a GIF with faster processing."""
    
    # Reduce the number of frames processed
    data = data.iloc[::frame_skip, :].reset_index(drop=True)
    num_frames = len(data)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Gesture Landmark Animation")
    ax.invert_yaxis()  # Flip the y-axis to match MediaPipe coordinate system

    scatter = ax.scatter([], [], c='blue', marker='o', alpha=0.5)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Gesture Landmark Animation - Frame {frame+1}/{num_frames}")
        ax.invert_yaxis()  # Keep y-axis flipped

        # Convert data to NumPy array for faster indexing
        frame_data = data.iloc[frame].to_numpy()

        x_vals = frame_data[1::3]  # Extract x values (every third column starting from index 1)
        y_vals = frame_data[2::3]  # Extract y values (every third column starting from index 2)

        ax.scatter(x_vals, y_vals, c='blue', marker='o', alpha=0.5)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    # Save animation as GIF using a faster backend (`imagemagick`)
    ani.save(save_path, writer='imagemagick', fps=15)
    plt.close(fig)  # Close figure to prevent Streamlit from rendering a static plot

# -----------------------------------
# Playback Controls
# -----------------------------------
if "playing" not in st.session_state:
    st.session_state["playing"] = False

with right_col:
    st.header("Dataset Visualization")

    if st.button("Start Animation"):
        st.session_state["playing"] = True

    if st.button("Stop Animation"):
        st.session_state["playing"] = False

    if st.session_state["playing"]:
        with st.spinner("Processing dataset..."):
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Extract landmark data (skip first two columns: 'class' and 'sequence_id')
            landmark_data = df.iloc[:, 2:]

            # Define GIF save path
            gif_path = "landmark_animation.gif"

            # Generate and save animation (skip every 2nd frame for speed)
            animate_landmarks(landmark_data, gif_path, frame_skip=2)

            # Display the saved GIF in Streamlit
            st.image(gif_path)
