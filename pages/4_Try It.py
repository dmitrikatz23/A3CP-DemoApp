#on the left slide the user to choose a model and an encoder
#button to start the streamer
#uses holistic model to vectorize gestures
#uses chosen model to predict the meaning of gesture
#displays the predicted gesture


import os
import streamlit as st
from huggingface_hub import HfApi

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference Interface")

# -----------------------------------
# Hugging Face Authentication
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

# -----------------------------------
# Initialize Hugging Face API and Repository Details
# -----------------------------------
api = HfApi()
model_repo_name = "dk23/A3CP_models"  # Repository containing trained model and encoder files

# -----------------------------------
# Helper Function: Retrieve and Match Model/Encoder Pairs
# -----------------------------------
@st.cache_data
def get_model_encoder_pairs():
    repo_files = api.list_repo_files(model_repo_name, repo_type="model", token=hf_token)
    model_files = [f for f in repo_files if f.endswith(".h5")]
    encoder_files = [f for f in repo_files if f.endswith(".pkl")]
    
    # Build a dictionary keyed by timestamp extracted from file names
    pairs = {}
    for mf in model_files:
        # Expected format: "LSTM_model_{timestamp}.h5"
        if mf.startswith("LSTM_model_") and mf.endswith(".h5"):
            ts = mf[len("LSTM_model_"):-len(".h5")]
            pairs.setdefault(ts, {})["model"] = mf

    for ef in encoder_files:
        # Expected format: "label_encoder_{timestamp}.pkl"
        if ef.startswith("label_encoder_") and ef.endswith(".pkl"):
            ts = ef[len("label_encoder_"):-len(".pkl")]
            pairs.setdefault(ts, {})["encoder"] = ef

    # Only keep pairs that have both model and encoder files
    valid_pairs = []
    for ts, files in pairs.items():
        if "model" in files and "encoder" in files:
            valid_pairs.append((ts, files["model"], files["encoder"]))
    
    # Sort pairs by timestamp in descending order (most recent first)
    valid_pairs.sort(key=lambda x: x[0], reverse=True)
    return valid_pairs

model_encoder_pairs = get_model_encoder_pairs()

# -----------------------------------
# Sidebar: Single Dropdown for Matched Pair Selection
# -----------------------------------
with st.sidebar:
    st.header("Configuration")
    if not model_encoder_pairs:
        st.warning("No valid model/encoder pairs found in the repository.")
    else:
        # Build a dictionary of labels to paired files
        pair_options = {}
        for ts, model_file, encoder_file in model_encoder_pairs:
            label = f"{ts} | Model: {model_file} | Encoder: {encoder_file}"
            pair_options[label] = (model_file, encoder_file)
        
        selected_label = st.selectbox("Select a Model/Encoder Pair", list(pair_options.keys()))
        selected_pair = pair_options[selected_label]
        st.markdown("---")
        st.write("**Selected Pair:**")
        st.write(f"Model File: `{selected_pair[0]}`")
        st.write(f"Encoder File: `{selected_pair[1]}`")

# -----------------------------------
# Main Content: TryIt Interface
# -----------------------------------
st.write("### Try It Interface")
st.write("Use the selected model and encoder pair to run inference.")
# TODO: Add further logic to download the selected files and run your inference pipeline.
