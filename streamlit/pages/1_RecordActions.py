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
