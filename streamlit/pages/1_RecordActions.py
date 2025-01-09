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


