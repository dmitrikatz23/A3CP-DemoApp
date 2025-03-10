# import os
# import pandas as pd
# import numpy as np
# import streamlit as st
# import joblib
# import datetime
# from huggingface_hub import HfApi, hf_hub_download
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
# from tensorflow.keras.optimizers import Adam

# # Repository details
# dataset_repo_name = "dk23/A3CP_actions"  # Dataset repo containing CSVs
# model_repo_name = "dk23/A3CP_models"    # Model repo for saving the trained model

# # Local folder for temporary files
# data_path = "local_data"
# os.makedirs(data_path, exist_ok=True)

# st.title("Train Gesture Recognition Model")

# # Load Hugging Face token from environment variables (secret: "Recorded_Datasets")
# hf_token = os.getenv("Recorded_Datasets")
# if not hf_token:
#     st.error("Hugging Face token not found. Please ensure the 'Recorded_Datasets' secret is added in the Space settings.")
#     st.stop()

# # Initialize Hugging Face API
# api = HfApi()

# # Fetch available CSV files from the dataset repository
# @st.cache_data
# def get_csv_files():
#     repo_files = api.list_repo_files(dataset_repo_name, repo_type="dataset", token=hf_token)
#     return [f for f in repo_files if f.endswith(".csv")]

# csv_files = get_csv_files()
# selected_csvs = st.multiselect("Select CSV files for training:", csv_files)

# if st.button("Download Selected CSVs"):
#     downloaded_files = []
#     for csv in selected_csvs:
#         file_path = hf_hub_download(dataset_repo_name, csv, repo_type="dataset", local_dir=data_path, token=hf_token)
#         downloaded_files.append(file_path)
#     st.success(f"Downloaded {len(downloaded_files)} files!")

# if st.button("Train Model") and selected_csvs:
#     # Read and combine CSVs into one DataFrame
#     all_dataframes = [pd.read_csv(os.path.join(data_path, os.path.basename(csv))) for csv in selected_csvs]
#     df = pd.concat(all_dataframes, ignore_index=True)

#     # Create a unique identifier per sequence
#     df['unique_id'] = df['class'] + '_' + df['sequence_id'].astype(str)
#     unique_ids = df['unique_id'].unique()

#     sequences, labels = [], []
#     for unique_id in unique_ids:
#         sequence_df = df[df['unique_id'] == unique_id]
#         # Drop non-feature columns
#         sequence_data = sequence_df.drop(columns=['class', 'sequence_id', 'unique_id']).values
#         sequences.append(sequence_data)
#         labels.append(sequence_df['class'].iloc[0])

#     # Pad sequences
#     X = pad_sequences(sequences, padding='post', dtype='float32', value=-1.0)
    
#     # Encode labels
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(labels)
#     y_onehot = to_categorical(y_encoded)

#     # Normalize hand joint positions (adjust indices if needed)
#     left_hand_positions = np.arange(162, 176)
#     right_hand_positions = np.arange(239, 253)
#     X[:, :, left_hand_positions] /= 180.0
#     X[:, :, right_hand_positions] /= 180.0

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_onehot, test_size=0.2, stratify=y_encoded
#     )
    
#     # Compute class weights for imbalance handling
#     class_weights = compute_class_weight("balanced", classes=np.unique(y_encoded), y=y_encoded)
#     class_weights_dict = dict(enumerate(class_weights))

#     # Build the LSTM model
#     model = Sequential([
#         Masking(mask_value=-1.0, input_shape=(X_train.shape[1], X_train.shape[2])),
#         LSTM(128, return_sequences=True),
#         Dropout(0.5),
#         LSTM(128),
#         Dropout(0.5),
#         Dense(y_train.shape[1], activation='softmax')
#     ])

#     model.compile(optimizer=Adam(learning_rate=0.001), 
#                   loss='categorical_crossentropy', 
#                   metrics=['accuracy'])
    
#     # Train the model
#     model.fit(X_train, y_train, epochs=50, batch_size=32, 
#               validation_data=(X_test, y_test), 
#               class_weight=class_weights_dict)

#     # Generate timestamped filenames for saving the model and label encoder
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_filename = f"LSTM_model_{timestamp}.h5"
#     encoder_filename = f"label_encoder_{timestamp}.pkl"
    
#     model_path = os.path.join(data_path, model_filename)
#     encoder_path = os.path.join(data_path, encoder_filename)
    
#     # Save model and encoder locally
#     model.save(model_path)
#     joblib.dump(le, encoder_path)

#     # Ensure the model repository exists; if not, create it
#     if not api.repo_exists(model_repo_name, repo_type="model", token=hf_token):
#         st.info(f"Repository '{model_repo_name}' not found. Creating it now...")
#         api.create_repo(repo_id=model_repo_name, repo_type="model", private=False, token=hf_token)

#     # Upload the files to the Hugging Face model repository
#     api.upload_file(
#         path_or_fileobj=model_path, 
#         path_in_repo=model_filename, 
#         repo_id=model_repo_name, 
#         repo_type="model",
#         token=hf_token
#     )
#     api.upload_file(
#         path_or_fileobj=encoder_path, 
#         path_in_repo=encoder_filename, 
#         repo_id=model_repo_name, 
#         repo_type="model",
#         token=hf_token
#     )

#     st.success(
#         f"Model and label encoder saved and uploaded to Hugging Face as '{model_filename}' and '{encoder_filename}'!"
#     )
