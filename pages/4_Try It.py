#on the left slide the user to choose a model and an encoder
#button to start the streamer
#uses holistic model to vectorize gestures
#uses chosen model to predict the meaning of gesture
#displays the predicted gesture


import streamlit as st
import os
from huggingface_hub import list_models

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="TryIt", layout="wide")
st.title("TryIt - Inference Page")

# -----------------------------------
# Hugging Face Authentication
# -----------------------------------
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

# -----------------------------------
# Retrieve HF Models from Repository
# -----------------------------------
@st.cache_data
def get_hf_models():
    # Optionally filter by author or organization (e.g., "dk23")
    models_list = list_models(author="dk23")  # Remove or adjust the filter as needed
    return [model.modelId for model in models_list]

hf_model_list = get_hf_models()

# -----------------------------------
# Sidebar Configuration
# -----------------------------------
with st.sidebar:
    st.header("Configuration")
    
    # Local model and encoder choices
    local_model_choice = st.selectbox("Choose a Local Model", options=["Model A", "Model B", "Model C"])
    encoder_choice = st.selectbox("Choose an Encoder", options=["Encoder X", "Encoder Y", "Encoder Z"])
    
    # HF model selection from repository
    hf_model_choice = st.selectbox("Choose a HF Model", options=hf_model_list if hf_model_list else ["No models found"])
    
    st.markdown("---")
    st.write("**Local Model:**", local_model_choice)
    st.write("**Encoder:**", encoder_choice)
    st.write("**HF Model:**", hf_model_choice)

# -----------------------------------
# Main Content Area
# -----------------------------------
st.write("### Try It Interface")
st.write("Use the selections from the sidebar to run inference or further processing.")
# TODO: Add your inference logic here using the selected models/encoder.
