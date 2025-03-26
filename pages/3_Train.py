import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import datetime
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam

# Repository details
dataset_repo_name = "dk23/A3CP_actions"  # Dataset repo containing CSVs
model_repo_name = "dk23/A3CP_models"    # Model repo for saving the trained model

# Local folder for temporary files
LOCAL_DATASET_DIR = "local_data"
os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

st.title("Train Gesture Recognition Model")

# Load Hugging Face token from environment variables (secret: "Recorded_Datasets")
hf_token = os.getenv("Recorded_Datasets")
if not hf_token:
    st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
    st.stop()

# Initialize Hugging Face API
hf_api = HfApi()

# Fetch available CSV files from the dataset repository
def get_csv_files():
    try:
        repo_files = hf_api.list_repo_files(dataset_repo_name, repo_type="dataset", token=hf_token)
        return sorted([f for f in repo_files if f.endswith(".csv")], reverse=True)
    except Exception as e:
        st.error(f"Error fetching dataset files: {e}")
        return []

csv_files = get_csv_files()
selected_csvs = st.multiselect("Select CSV files for training:", csv_files)

if st.button("Download Selected CSVs"):
    downloaded_files = []
    missing_files = []
    
    for csv in selected_csvs:
        local_csv_path = os.path.join(LOCAL_DATASET_DIR, csv)

        if not os.path.exists(local_csv_path):  # Avoid redundant downloads
            with st.spinner(f"Downloading {csv}..."):
                try:
                    file_path = hf_hub_download(dataset_repo_name, csv, repo_type="dataset", local_dir=LOCAL_DATASET_DIR, token=hf_token)
                    downloaded_files.append(file_path)
                except Exception as e:
                    missing_files.append(csv)
                    st.error(f"Failed to download {csv}: {e}")

        else:
            st.info(f"{csv} already exists locally.")

    if downloaded_files:
        st.success(f"Successfully downloaded {len(downloaded_files)} files!")
    
    if missing_files:
        st.error(f"Failed to download the following files: {missing_files}. Check if they exist in the Hugging Face repo.")



if st.button("Train Model") and selected_csvs:
    # Read and combine CSVs into one DataFrame
    #all_dataframes = [pd.read_csv(os.path.join(data_path, os.path.basename(csv))) for csv in selected_csvs]
    all_dataframes = [pd.read_csv(os.path.join(LOCAL_DATASET_DIR, csv)) for csv in selected_csvs]
    df = pd.concat(all_dataframes, ignore_index=True)

    # Create a unique identifier per sequence

    df['sequence_id'] = df['sequence_id'].astype(str).replace('nan', 'unknown')
    df['unique_id'] = df['class'].astype(str) + '_' + df['sequence_id']
    unique_ids = df['unique_id'].unique()

    sequences, labels = [], []
    for unique_id in unique_ids:
        sequence_df = df[df['unique_id'] == unique_id]
        # Drop non-feature columns
        sequence_data = sequence_df.drop(columns=['class', 'sequence_id', 'unique_id']).values
        sequences.append(sequence_data)
        labels.append(sequence_df['class'].iloc[0])

    # Pad sequences
    X = pad_sequences(sequences, maxlen= 30, padding='post', dtype='float32', value=-1.0)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_onehot = to_categorical(y_encoded)

    # Normalize hand joint positions (adjust indices if needed)
    left_hand_positions = np.arange(162, 176)
    right_hand_positions = np.arange(239, 253)
    X[:, :, left_hand_positions] /= 180.0
    X[:, :, right_hand_positions] /= 180.0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, stratify=y_encoded
    )
    
    # Compute class weights for imbalance handling
    class_weights = compute_class_weight("balanced", classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = dict(enumerate(class_weights))

    # Build the LSTM model
    model = Sequential([
        Masking(mask_value=-1.0, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, 
              validation_data=(X_test, y_test), 
              class_weight=class_weights_dict)

    # Generate timestamped filenames for saving the model and label encoder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"LSTM_model_{timestamp}.h5"
    encoder_filename = f"label_encoder_{timestamp}.pkl"
    
    model_path = os.path.join(LOCAL_DATASET_DIR, model_filename)
    encoder_path = os.path.join(LOCAL_DATASET_DIR, encoder_filename)
    
    # Save model and encoder locally
    model.save(model_path)
    joblib.dump(le, encoder_path)

    # Ensure the model repository exists; if not, create it
    if not hf_api.repo_exists(model_repo_name, repo_type="model", token=hf_token):
        st.info(f"Repository '{model_repo_name}' not found. Creating it now...")
        hf_api.create_repo(repo_id=model_repo_name, repo_type="model", private=False, token=hf_token)

    # Upload the files to the Hugging Face model repository
    hf_api.upload_file(
        path_or_fileobj=model_path, 
        path_in_repo=model_filename, 
        repo_id=model_repo_name, 
        repo_type="model",
        token=hf_token
    )
    hf_api.upload_file(
        path_or_fileobj=encoder_path, 
        path_in_repo=encoder_filename, 
        repo_id=model_repo_name, 
        repo_type="model",
        token=hf_token
    )

    st.success(
        f"Model and label encoder saved and uploaded to Hugging Face as '{model_filename}' and '{encoder_filename}'!"
    )

###temp to check if classes in .pkl
import joblib
import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "local_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Try downloading the most recent files just after upload
try:
    model_path = hf_hub_download(
        repo_id=model_repo_name,
        filename=model_filename,
        repo_type="model",
        local_dir=MODEL_DIR,
        token=hf_token
    )
    encoder_path = hf_hub_download(
        repo_id=model_repo_name,
        filename=encoder_filename,
        repo_type="model",
        local_dir=MODEL_DIR,
        token=hf_token
    )
except Exception as e:
    st.warning(f"Could not download most recent model: {e}")

# Persistent debug panel that works across sessions
st.subheader("🔍 Model Debug Info")
if os.path.exists(MODEL_DIR):
    encoder_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if encoder_files:
        latest_encoder = max(encoder_files, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
        encoder_path = os.path.join(MODEL_DIR, latest_encoder)
        encoder = joblib.load(encoder_path)
        st.write("**Encoder Classes:**", list(encoder.classes_))
    else:
        st.warning("No encoder files (.pkl) found in local_models.")
else:
    st.warning("⚠️ 'local_models' folder does not exist yet. Train a model first.")
