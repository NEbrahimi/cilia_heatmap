import streamlit as st
import cv2
from nd2reader import ND2Reader
import os
import tempfile
import shutil
import subprocess
import numpy as np

# Set page configuration and styles
st.set_page_config(page_title="Cilia Analysis Tool", layout="wide")
st.markdown(
    """
    <style>
    .css-18e3th9 {background-color: #F8F9FA;}
    .css-1d391kg {background-color: white;}
    </style>
    """, unsafe_allow_html=True)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def convert_nd2_to_video(input_file, output_dir, fps, gamma=1.0):
    with ND2Reader(input_file) as images:
        height, width = images.metadata['height'], images.metadata['width']
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
        video_filename = os.path.splitext(os.path.basename(input_file.name))[0] + '.avi'
        video_path = os.path.join(output_dir, video_filename)
        out = cv2.VideoWriter(video_path, fourcc, int(fps), (width, height))

        for frame in images:
            frame_8bit = cv2.normalize(frame, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_adjusted = adjust_gamma(frame_8bit, gamma)
            frame_color = cv2.cvtColor(frame_adjusted, cv2.COLOR_GRAY2BGR)
            out.write(frame_color)
        out.release()
        return video_path

def convert_to_h264(input_path):
    output_path = input_path.replace('.avi', '.mp4')
    subprocess.run(['ffmpeg', '-i', input_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path], check=True)
    return output_path

# Input widgets
with st.sidebar:
    st.title("File Upload and Review")
    uploaded_file = st.file_uploader("Upload a .nd2 file", type=["nd2"])
    exposure_time = st.number_input("Exposure Time (seconds)", min_value=0.0001, value=0.003, step=0.0001, format="%.3f")
    gamma_value = st.slider("Adjust Gamma", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

# Processing uploaded file
if uploaded_file and exposure_time > 0:
    fps = 1 / exposure_time
    temp_dir = tempfile.mkdtemp()
    try:
        video_path = convert_nd2_to_video(uploaded_file, temp_dir, fps, gamma=gamma_value)
        h264_video_path = convert_to_h264(video_path)
        with open(h264_video_path, "rb") as file:
            file_data = file.read()
        st.download_button("Download Original Video", file_data, h264_video_path.split(os.path.sep)[-1])
        col1, col2 = st.columns(2)
        with col1:
            # Using HTML to embed video with loop attribute
            st.video(h264_video_path, format='video/mp4', start_time=0, loop=True)
    finally:
        shutil.rmtree(temp_dir)

