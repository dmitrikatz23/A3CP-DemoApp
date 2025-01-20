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
