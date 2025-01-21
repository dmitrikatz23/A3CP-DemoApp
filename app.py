import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# --- Layout Configuration ---
st.set_page_config(
    page_title="A3CP App",
    layout="centered",
    initial_sidebar_state="auto",
)

# Hamburger menu prompt
st.markdown(
    """
    <div style="text-align: center; margin-top: 14px;">
        <p style="font-size: 1em; color: #1E3A8A;">
            <- Use the menu in the top-left corner to navigate!
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
        color: #1E3A8A; /* Updated to dark blue */
        margin-bottom: 0.2em;
        text-align: center;
        font-family: "Helvetica Neue", Arial, sans-serif;
    }    
    .stButton > button {
        display: block;
        margin: 20px auto; /* Center the button */
        padding: 10px 20px;
        font-size: 16px;
        color: #1E3A8A; /* Dark blue text */
        border: 2px solid #1E3A8A; /* Dark blue border */
        border-radius: 8px;
        background-color: white; /* White background */
        text-align: center;
        font-weight: normal;
        cursor: pointer;
    }
    .stButton > button:hover {
        font-weight: bold; /* Bold text on hover */
        color: green; /* green text on hover */
        border: 3px solid green; /* green border */
    }
    .stButton > button:active {
        background-color: #DFF2DF; /* Light green background on mouse down */
        color: #4CAF50; /* Green text */
        border: 2px solid #4CAF50; /* Green border */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Container ---
with st.container():
    # Title & Subtitle
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 class='landing-title'>Welcome to the A3CP Action Mapping App!</h1>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<h3 class='landing-subtitle'>Capture and visualize non-verbal gestural communication</h3>", unsafe_allow_html=True)

    # Introductory Text
    st.markdown(
        """
        <p class='landing-text'>
        This application allows you to name a non-verbal expression of an action, record it using your webcam. 
        Each recorded frame is converted to numbers (vectorized), then stored for future analysis. No video is stored.
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
        2. Enter name of action (Hello!).<br>
        3. Confirm the action.<br>
        4. The app will open a webcam stream for you to perform and record.<br>
        5. After processing, the app will store key frames as numbers.<br>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Button to Go to Record Actions Page
    if st.button("Go to Recording Page to Start"):
        switch_page("RecordActions")
