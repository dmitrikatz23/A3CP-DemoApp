from huggingface_hub import list_repo_files
import streamlit as st

# Hugging Face repository details
repo_name = "dk23/A3CP_actions"

# Fetch all CSV files from the repository
@st.cache_data
def get_huggingface_csv_files():
    files = list_repo_files(repo_id=repo_name, repo_type="dataset")
    csv_files = [f for f in files if f.endswith(".csv")]
    
    # Sort files by date in descending order (most recent first)
    csv_files.sort(reverse=True)  
    return csv_files

# Display saved datasets in the sidebar
st.sidebar.header("📁 Saved Datasets (Hugging Face)")

csv_files = get_huggingface_csv_files()

if csv_files:
    st.sidebar.write("🔹 Available datasets:")
    for file in csv_files:
        st.sidebar.write(f"📄 {file}")
else:
    st.sidebar.write("⚠️ No saved datasets found in the repository.")
