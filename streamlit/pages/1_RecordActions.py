import streamlit as st
import csv
import numpy as np
import cv2
import mediapipe as mp
import time
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer

#set the page configuration
st.set_page_config(layout = 'wide')

# JavaScript to request camera permissions
st.components.v1.html("""
<script>
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
        document.body.innerHTML += "<p style='color: green;'>Camera access granted!</p>";
    })
    .catch(function(err) {
        document.body.innerHTML += "<p style='color: red;'>Camera access denied or unavailable. Please check permissions and try again.</p>";
        console.error("Camera access error:", err);
    });
</script>
""")


#check webcam
def check_camera_access():
    cap = cv2.VideoCapture(0) # attempt to access webcam
    if not cap.isOpened():
        st.error('webcam is not accessible. please chec browser and system permissions')
    else:
        st.success ('webcam is accessible')
        cap.release()

st.title('webcam permission test')

if st.button ('check Webcam'):
    check_camera_access()
