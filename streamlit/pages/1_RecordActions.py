import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Live Camera Feed")

webrtc_streamer(key="example")