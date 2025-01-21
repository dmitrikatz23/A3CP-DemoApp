import os
import pandas as pd
import random
import streamlit as st
from huggingface_hub import Repository

# Get the Hugging Face token from the "Recorded_Datasets" secret
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

# Hugging Face repository details
repo_name = "dk23/A3CP_actions"  # Your dataset repository
local_repo_path = "local_repo"

# CSV file name
csv_file_name = "A3CP_actions.csv"
csv_file_local = csv_file_name  # Local file
csv_file_repo = os.path.join(local_repo_path, csv_file_name)  # File in the repo directory

# Create CSV if it doesn't exist
if not os.path.exists(csv_file_local):
    df = pd.DataFrame(columns=[str(i) for i in range(1, 11)])  # Create 10 columns, 1-10
    df.to_csv(csv_file_local, index=False)

# Load the existing CSV
df = pd.read_csv(csv_file_local)

# Clone or create the Hugging Face repository
repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")

# Function to save CSV permanently
def save_to_repo():
    # Ensure the repo directory exists
    os.makedirs(local_repo_path, exist_ok=True)
    # Copy the local CSV to the repo directory
    df.to_csv(csv_file_repo, index=False)
    # Add, commit, and push changes
    repo.git_add(csv_file_name)  # Use the relative path within the repo
    repo.git_commit("Update A3CP actions CSV")
    repo.git_push()

# Streamlit app
st.title("Save A3CP Actions to Hugging Face")

# Button to add a new row to the CSV
if st.button("Add Row to CSV"):
    new_row = pd.DataFrame([{str(i): random.randint(1, 10) for i in range(1, 11)}])  # Generate random numbers 1-10
    df = pd.concat([df, new_row], ignore_index=True)  # Add the new row
    df.to_csv(csv_file_local, index=False)  # Save locally
    st.success("Row added!")

# Button to save the CSV to Hugging Face repository
if st.button("Save to Hugging Face Repository"):
    try:
        save_to_repo()
        st.success(f"CSV saved to Hugging Face repository: {repo_name}")
    except Exception as e:
        st.error(f"Failed to save to repository: {e}")

# Display the current CSV in the app
st.write("Current CSV:")
st.dataframe(df)
