import os
import pandas as pd
import streamlit as st

# Optional: set page config for this "Data Visualizer" page
st.set_page_config(page_title="Data Visualizer", layout="centered")

st.title("Data Visualizer")

# Attempt to retrieve the CSV file path from session state
# (as set in 1_RecordActions.py)
csv_file = st.session_state.get("csv_file", None)

if not csv_file:
    # If there's no file in session state, 
    # fallback to a default path or prompt the user
    st.warning("No CSV file found in session state. Please record actions first!")
else:
    # Check if the file actually exists on the filesystem
    if not os.path.exists(csv_file):
        st.error(f"CSV file not found at: {csv_file}")
    else:
        # Load the data
        df = pd.read_csv(csv_file)

        st.subheader("Recorded Actions Data")
        st.write(f"**File Path:** `{csv_file}`")
        
        # Show a preview of the data
        st.dataframe(df.head(20))

        st.markdown("---")
        st.subheader("Dataset Summary")
        st.write(df.describe())

        # Optional: Basic filtering or selection
        # For example, select a specific action
        actions_available = df['class'].unique().tolist() if 'class' in df.columns else []
        if actions_available:
            selected_action = st.selectbox("Filter by Action:", options=["All"] + actions_available)
            
            if selected_action != "All":
                df_filtered = df[df['class'] == selected_action]
            else:
                df_filtered = df
        else:
            df_filtered = df

        st.markdown("### Filtered Data Preview")
        st.dataframe(df_filtered.head(20))

        # You could add basic charts here, e.g., a bar chart of how many rows per action
        if 'class' in df.columns:
            action_counts = df['class'].value_counts()
            st.bar_chart(action_counts)
        
        # Any other charts / analyses you want to display
        st.markdown("---")
        st.info("Feel free to add more custom visualizations based on your CSV data!")


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