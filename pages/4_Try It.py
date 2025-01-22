import os
import pandas as pd
import random
import streamlit as st
from huggingface_hub import Repository

# Load Hugging Face token from environment variables
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

# Hugging Face repository details
repo_name = "dk23/A3CP_actions"
local_repo_path = "local_repo"

# Configure generic Git identity
git_user = "A3CP_bot"
git_email = "no-reply@huggingface.co"

# CSV file paths
csv_file_name = "A3CP_actions.csv"
csv_file_local = csv_file_name
csv_file_repo = os.path.join(local_repo_path, csv_file_name)

# Create CSV if it doesn't exist
if not os.path.exists(csv_file_local):
    df = pd.DataFrame(columns=[str(i) for i in range(1, 11)])
    df.to_csv(csv_file_local, index=False)

# Load the existing CSV
df = pd.read_csv(csv_file_local)

# Clone or create the Hugging Face repository
repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")

# Configure Git user details (Positional Arguments)
repo.git_config_username_and_email(git_user, git_email)

# Function to save CSV permanently
def save_to_repo():
    os.makedirs(local_repo_path, exist_ok=True)
    df.to_csv(csv_file_repo, index=False)
    repo.git_add(csv_file_name)
    repo.git_commit("Update A3CP actions CSV")
    repo.git_push()

# Streamlit app
st.title("Save A3CP Actions to Hugging Face")

if st.button("Add Row to CSV"):
    new_row = pd.DataFrame([{str(i): random.randint(1, 10) for i in range(1, 11)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file_local, index=False)
    st.success("Row added!")

if st.button("Save to Hugging Face Repository"):
    try:
        save_to_repo()
        st.success(f"CSV saved to Hugging Face repository: {repo_name}")
    except Exception as e:
        st.error(f"Failed to save to repository: {e}")

st.write("Current CSV:")
st.dataframe(df)