import streamlit as st
from pathlib import Path
from datetime import datetime
import csv
import pandas as pd

# Function to initialize CSV folder and file
def initialize_csv_folder_and_file():
    """
    Initializes the CSV folder and creates a new CSV file if it doesn't exist.
    Returns the path to the CSV file.
    """
    # Set up the path for the CSV folder to be one level up from the `pages` folder
    pages_folder = Path(__file__).resolve().parent  # Location of the current file (assumed to be in `pages`)
    csv_folder = pages_folder.parent / "csv"  # One level up from `pages`

    # Ensure the folder exists
    if not csv_folder.exists():
        csv_folder.mkdir(parents=True)
        st.write(f"Created CSV folder: {csv_folder}")

    # Initialize the CSV file path
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

# Function to append data to the CSV file
def append_to_csv(file_path, rows):
    """
    Appends rows of data to the specified CSV file.
    """
    try:
        with open(file_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)
        st.success(f"Successfully appended {len(rows)} rows to {file_path}")
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

# Example data to append
example_rows = [
    ["wave", 1, 0.5, 0.5, 1],
    ["wave", 2, 0.6, 0.5, 1],
]

# UI Components
if st.button("Append Example Data to CSV"):
    append_to_csv(csv_file, example_rows)

# Display the CSV contents
display_csv(csv_file)
