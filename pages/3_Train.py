import os
import pandas as pd
import random
import streamlit as st
from huggingface_hub import Repository

# Get the Hugging Face token from the environment variable
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face token not found. Please add it as a secret in your Space settings.")
    st.stop()

# Hugging Face repository details
repo_name = "your-username/troubleshooting-dataset"
local_repo_path = "local_repo"

# CSV file path
csv_file = "troubleshooting.csv"

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=[str(i) for i in range(1, 11)])
    df.to_csv(csv_file, index=False)

# Load the existing CSV
df = pd.read_csv(csv_file)

# Clone or create the Hugging Face repository
repo = Repository(local_dir=local_repo_path, clone_from=repo_name, use_auth_token=hf_token, repo_type="dataset")

# Function to save CSV permanently
def save_to_repo():
    csv_repo_path = os.path.join(local_repo_path, "troubleshooting.csv")
    df.to_csv(csv_repo_path, index=False)
    repo.git_add(csv_repo_path)
    repo.git_commit("Update troubleshooting CSV")
    repo.git_push()

# Streamlit app
st.title("Save CSV to Hugging Face")

if st.button("Add Row to CSV"):
    new_row = pd.DataFrame([{str(i): random.randint(1, 10) for i in range(1, 11)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)
    st.success("Row added!")

if st.button("Save to Hugging Face Repository"):
    save_to_repo()
    st.success("CSV saved to Hugging Face repository!")

st.dataframe(df)
