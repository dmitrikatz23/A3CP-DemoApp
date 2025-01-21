import streamlit as st
import pandas as pd
import os
import random

# CSV file path
csv_file = "troubleshooting.csv"

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    # Create an empty DataFrame with columns numbered 1 to 10
    df = pd.DataFrame(columns=[str(i) for i in range(1, 11)])
    df.to_csv(csv_file, index=False)

# Load the existing CSV
df = pd.read_csv(csv_file)

st.title("CSV Troubleshooting App")

# Button to add a new row
if st.button("Make a CSV"):
    # Generate a new row with random numbers from 1 to 10
    new_row = {str(i): random.randint(1, 10) for i in range(1, 11)}
    # Append the new row to the DataFrame
    df = df.append(new_row, ignore_index=True)
    # Save the updated DataFrame to the CSV
    df.to_csv(csv_file, index=False)
    st.success("New row added to the CSV!")

# Display the DataFrame
st.write("Current CSV Data:")
st.dataframe(df)

# Download button for the CSV
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="troubleshooting.csv",
    mime="text/csv",
)
