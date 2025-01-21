import streamlit as st
from pathlib import Path

# --- Layout Configuration ---
st.set_page_config(
    page_title="A3CP Action Recording App",
    layout="centered",
    initial_sidebar_state="auto",
)
# hamburger menu more prominent
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <p style="font-size: 0.9em; color: #1E3A8A;">
            📖 Use the menu in the top-left corner to navigate!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    .landing-title {
        font-size: 3em;
        font-weight: bold;
        color: #4B9CD3;
        margin-bottom: 0.2em;
        text-align: center;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
    .landing-subtitle {
        font-size: 1.2em;
        margin-top: 0.2em;
        text-align: center;
        color: #555555;
    }
    .landing-container {
        max-width: 700px;
        margin: auto;
        margin-top: 4rem;
        padding: 2rem;
        background-color: #fefefe;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .landing-text {
        font-size: 1.1em;
        margin-top: 1.5em;
        text-align: justify;
        color: #333333;
        line-height: 1.6em;
    }
    .landing-button {
        display: block;
        margin: 2em auto;
        width: 60%;
        font-size: 1.2em;
        padding: 0.6em;
        background-color: white; /* White background */
        color: #1E3A8A; /* Dark blue text */
        border-radius: 8px;
        border: 2px solid #1E3A8A; /* Blue outline */
        text-align: center;
        font-weight: bold;
        cursor: pointer;
        text-decoration: none;
    }
    .landing-button:hover {
        background-color: #E5EFFF; /* Light blue on hover */
        color: #1E3A8A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Container ---
with st.container():
    st.markdown("<div class='landing-container'>", unsafe_allow_html=True)

    # Title & Subtitle
    st.markdown("<h1 class='landing-title'>Welcome to the Action Recording App!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='landing-subtitle'>Capture and analyze actions with MediaPipe Holistic</h3>", unsafe_allow_html=True)

    # Introductory Text
    st.markdown(
        """
        <p class='landing-text'>
        This application allows you to name an action, record it (gestures, poses, etc.) using your webcam and 
        automatically detect keyframes using velocity and acceleration thresholds.  
        Each recorded frame is converted to numbers (vectorized), then stored in a CSV file for future analysis. No video is stored.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Instructions
    st.markdown(
        """
        <p class='landing-text'>
        <b>How it works:</b><br>
        1. Navigate to the Record Actions page.<br>
        2. Enter your intended action meaning (e.g., "I’m hungry").<br>
        3. Confirm the action.<br>
        4. The app will open a webcam stream for you to perform and record.<br>
        5. After processing, the app will store frames in the CSV folder.<br>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Button to Go to Record Actions Page
    st.markdown(
        "<a class='landing-button' href='pages/1_RecordActions.py'>Go to Recording Page to Start</a>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
