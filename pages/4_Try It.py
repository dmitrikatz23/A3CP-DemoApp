import streamlit as st
from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
from queue import Queue

# Initialize Queue
data_queue = Queue()

# Function to initialize CSV folder and file
def initialize_csv_folder_and_file():
    """
    Initializes the CSV folder and creates a new CSV file if it doesn't exist.
    Returns the path to the CSV file.
    """
    pages_folder = Path(__file__).resolve().parent  # Location of the current file (assumed to be in `pages`)
    csv_folder = pages_folder.parent / "csv"  # One level up from `pages`

    # Ensure the folder exists
    if not csv_folder.exists():
        csv_folder.mkdir(parents=True)
        st.write(f"Created CSV folder: {csv_folder}")

    if "csv_file" not in st.session_state:
        session_start_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state["csv_file"] = csv_folder / f"all_actions_recorded_{session_start_str}.csv"

    return st.session_state["csv_file"]

# Function to initialize CSV file with a header
def initialize_csv_file(file_path, header):
    """
    Creates a CSV file with the specified header if it doesn't exist.
    """
    if not file_path.exists():
        with open(file_path, mode="w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
        st.write(f"CSV initialized with header at {file_path}")
    else:
        st.write(f"CSV file already exists: {file_path}")

# Function to add data to the queue
def add_to_queue(data):
    """
    Adds data to the queue.
    """
    data_queue.put(data)
    st.write(f"Added to queue: {data}")
    st.write(f"Queue size: {data_queue.qsize()}")

# Function to write data from the queue to the CSV
def write_queue_to_csv(file_path):
    """
    Writes all data in the queue to the CSV file.
    """
    if data_queue.empty():
        st.warning("No data in queue to write.")
        return

    rows = []
    while not data_queue.empty():
        rows.append(data_queue.get())

    try:
        with open(file_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)
        st.success(f"Written {len(rows)} rows from queue to {file_path}")
    except Exception as e:
        st.error(f"Error writing to CSV: {e}")

# Function to display the contents of the CSV file
def display_csv(file_path):
    """
    Displays the contents of the CSV file using Streamlit's DataFrame viewer.
    """
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
        st.info("The CSV file does not exist yet.")

# Main Execution
csv_file = initialize_csv_folder_and_file()

# Define the CSV header
csv_header = ["class", "sequence_id", "pose_x1", "pose_y1", "pose_v1"]  # Example header
initialize_csv_file(csv_file, csv_header)

# Example data to add to the queue
example_data_1 = ["wave", 1, 0.5, 0.5, 1]
example_data_2 = ["wave", 2, 0.6, 0.5, 1]

# UI Components
if st.button("Add Example Data to Queue"):
    add_to_queue(example_data_1)
    add_to_queue(example_data_2)

if st.button("Write Queue to CSV"):
    write_queue_to_csv(csv_file)

# Display the CSV contents
display_csv(csv_file)
