import streamlit as st
from pathlib import Path
from datetime import datetime
import os
import csv

def initialize_csv_folder_and_file():
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

# Initialize the folder and file only after Streamlit is running
csv_file = initialize_csv_folder_and_file()
st.write(f"CSV file path: {csv_file}")

