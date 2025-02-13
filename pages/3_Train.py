import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam

# Hugging Face repo details
repo_name = "dk23/A3CP_actions"
data_path = "local_data"
os.makedirs(data_path, exist_ok=True)

st.title("Train Gesture Recognition Model")

# Fetch available CSV files from Hugging Face
def get_csv_files():
    api = HfApi()
    repo_files = api.list_repo_files(repo_name, repo_type="dataset")
    return [f for f in repo_files if f.endswith(".csv")]

csv_files = get_csv_files()
selected_csvs = st.multiselect("Select CSV files for training:", csv_files)

if st.button("Download Selected CSVs"):
    downloaded_files = []
    for csv in selected_csvs:
        file_path = hf_hub_download(repo_name, csv, repo_type="dataset", local_dir=data_path)
        downloaded_files.append(file_path)
    st.success(f"Downloaded {len(downloaded_files)} files!")

if st.button("Train Model") and selected_csvs:
    all_dataframes = [pd.read_csv(os.path.join(data_path, csv)) for csv in selected_csvs]
    df = pd.concat(all_dataframes, ignore_index=True)

    df['unique_id'] = df['class'] + '_' + df['sequence_id'].astype(str)
    unique_ids = df['unique_id'].unique()

    sequences, labels = [], []
    for unique_id in unique_ids:
        sequence_df = df[df['unique_id'] == unique_id]
        sequence_data = sequence_df.drop(columns=['class', 'sequence_id', 'unique_id']).values
        sequences.append(sequence_data)
        labels.append(sequence_df['class'].iloc[0])

    X = pad_sequences(sequences, padding='post', dtype='float32', value=-1.0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y_onehot = to_categorical(y_encoded)

    left_hand_positions = np.arange(162, 176)
    right_hand_positions = np.arange(239, 253)
    X[:, :, left_hand_positions] /= 180.0
    X[:, :, right_hand_positions] /= 180.0

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, stratify=y_encoded)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = dict(enumerate(class_weights))

    model = Sequential([
        Masking(mask_value=-1.0, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)

    model.save(os.path.join(data_path, "gesture_model.h5"))
    joblib.dump(le, os.path.join(data_path, "label_encoder.pkl"))
    st.success("Model and label encoder saved!")
