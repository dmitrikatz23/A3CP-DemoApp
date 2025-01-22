import os
import random
import pandas as pd
import streamlit as st
from huggingface_hub import Repository

# Hugging Face Repository Configuration
repo_name = "your-username/your-dataset-repo"  # Replace with your Hugging Face repo name
hf_token = "your_hf_token"  # Replace with your Hugging Face token
local_repo_path = "./hf_repo"  # Local directory for the repo

# CSV File Configuration
csv_file_name = "A3CP_actions.csv"
csv_file_local = csv_file_name  # Local file
csv_file_repo = os.path.join(local_repo_path, csv_file_name)  # File in the repo directory

# Create the CSV locally if it doesn't exist
if not os.path.exists(csv_file_local):
    df = pd.DataFrame(columns=[str(i) for i in range(1, 11)])  # Create 10 columns labeled 1-10
    df.to_csv(csv_file_local, index=False)

# Load the existing CSV
df = pd.read_csv(csv_file_local)

# Clone or create the Hugging Face repository
repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")

# Function to save the CSV to the Hugging Face repository
def save_to_repo():
    # Ensure the repo directory exists
    os.makedirs(local_repo_path, exist_ok=True)
    # Save the CSV to the repo directory
    df.to_csv(csv_file_repo, index=False)
    # Add, commit, and push changes to the repository
    repo.git_add(csv_file_name)  # Use the relative path within the repo
    repo.git_commit("Update A3CP actions CSV")
    repo.git_push()

# Streamlit app
st.title("Save A3CP Actions to Hugging Face")

# Button to add a new row to the CSV
if st.button("Add Row to CSV"):
    # Generate a new row with random integers 1-10
    new_row = pd.DataFrame([{str(i): random.randint(1, 10) for i in range(1, 11)}])
    df = pd.concat([df, new_row], ignore_index=True)  # Add the new row
    df.to_csv(csv_file_local, index=False)  # Save locally
    st.success("Row added!")

# Button to save the CSV to the Hugging Face repository
if st.button("Save to Hugging Face Repository"):
    try:
        save_to_repo()
        st.success(f"CSV saved to Hugging Face repository: {repo_name}")
    except Exception as e:
        st.error(f"Failed to save to repository: {e}")

# Display the current CSV in the app
st.write("Current CSV:")
st.dataframe(df)
