import streamlit as st
import cv2
from nd2reader import ND2Reader
import os
import tempfile
import subprocess
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt


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
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
    subprocess.run(['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path], check=True)
    return output_path

def pixel_wise_fft_filtered_and_masked(video_path, fps, freq_min, freq_max, mag_threshold):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return None, None  # Return paths for mask and magnitude images
    frames = []
    while ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        ret, frame = cap.read()
    cap.release()

    data_cube = np.stack(frames, axis=2)
    fft_cube = fftshift(fft(data_cube, axis=2), axes=(2,))
    magnitude = np.abs(fft_cube)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(frames), d=1 / fps))
    valid_indices = (freqs > freq_min) & (freqs < freq_max)
    magnitude_filtered = magnitude[:, :, valid_indices]

    dominant_indices = np.argmax(magnitude_filtered, axis=2)
    dominant_magnitude = np.max(magnitude_filtered, axis=2)

    mask = (dominant_magnitude >= mag_threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 10  # Minimum area threshold to keep a contour

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)  # fill with 255 where the ciliated areas are significant


    mask_path = os.path.join(tempfile.gettempdir(), 'mask.png')
    cv2.imwrite(mask_path, filtered_mask)

    # Plot and save the magnitude map
    plt.figure(figsize=(10, 8))
    plt.imshow(dominant_magnitude, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Dominant Frequency Magnitude Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    magnitude_path = os.path.join(tempfile.gettempdir(), 'magnitude_map.png')
    plt.savefig(magnitude_path)
    plt.close()

    return mask_path, magnitude_path


def apply_mask_to_video(video_path, mask_path, output_video_path, fps):
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        # Update codec to 'X264'
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_video_path, fourcc, fps,
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        ret, frame = cap.read()
        while ret:
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            out.write(masked_frame)
            ret, frame = cap.read()
        cap.release()
        out.release()
    else:
        print("Error: Mask file could not be loaded.")


def save_masked_frames(video_path, mask_path, output_dir, fps):
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        frame_index = 0
        ret, frame = cap.read()
        while ret:
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            frame_filename = f"frame_{frame_index:04d}.png"
            cv2.imwrite(os.path.join(output_dir, frame_filename), masked_frame)
            ret, frame = cap.read()
            frame_index += 1
        cap.release()
    else:
        print("Error: Mask file could not be loaded.")

def create_video_from_frames(output_dir, output_video_path, fps):
    command = [
        'ffmpeg',
        '-r', str(fps),  # Frame rate
        '-f', 'image2',  # Force format
        '-s', '1920x1080',  # Size of the images
        '-i', os.path.join(output_dir, 'frame_%04d.png'),  # Input format
        '-vcodec', 'libx264',  # Output codec
        '-crf', '23',  # Quality
        '-preset', 'fast',  # Compression speed
        output_video_path
    ]
    subprocess.run(command, check=True)

# Initialize session state with default values if they don't exist
if 'freq_min' not in st.session_state:
    st.session_state.freq_min = 2  # Default minimum frequency

if 'freq_max' not in st.session_state:
    st.session_state.freq_max = 30  # Default maximum frequency

if 'mag_threshold' not in st.session_state:
    st.session_state.mag_threshold = 300  # Default magnitude threshold

# Input widgets
with st.sidebar:
    st.title("Step 1: File Upload and Review")
    uploaded_file = st.file_uploader("Upload a .nd2 file", type=["nd2"])
    exposure_time = st.number_input("Exposure Time (seconds)", min_value=0.0001, value=0.003, step=0.0001, format="%.3f")
    gamma_value = st.slider("Adjust Gamma", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    run_step_1 = st.button("Run Step 1")

    st.title("Step 2: Masking")
    freq_min = st.number_input("Min Frequency", value=st.session_state.freq_min, help="Set the minimum frequency for filtering.")
    freq_max = st.number_input("Max Frequency", value=st.session_state.freq_max, help="Set the maximum frequency for filtering.")
    mag_threshold = st.number_input("Magnitude Threshold", value=st.session_state.mag_threshold, help="Set the threshold for background detection sensitivity.")
    run_step_2 = st.button("Run Step 2")

# Update session state when values change
st.session_state.freq_min = freq_min
st.session_state.freq_max = freq_max
st.session_state.mag_threshold = mag_threshold

# Step 1
# Step 1
if uploaded_file and exposure_time > 0 and run_step_1:
    print("File uploaded successfully:", uploaded_file.name)  # Debugging statement 1

    fps = 1 / exposure_time
    # Maintain the temporary directory across reruns
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()

    video_path = convert_nd2_to_video(uploaded_file, st.session_state['temp_dir'], fps, gamma=gamma_value)

    if video_path:
        print("Video created at:", video_path)  # Debugging statement 2
        if os.path.exists(video_path):
            print("Video file exists.")
        else:
            print("Video file does not exist.")

    h264_video_path = convert_to_h264(video_path)

    if h264_video_path:
        print("Converted to H264 at:", h264_video_path)  # Debugging statement 3
        if os.path.exists(h264_video_path):
            print("H264 video file exists.")
        else:
            print("H264 video file does not exist.")

    st.session_state['original_video_path'] = h264_video_path
    with open(h264_video_path, "rb") as file:
        file_data = file.read()
    st.download_button("Download Original Video", file_data, h264_video_path.split(os.path.sep)[-1])

    col1, col2 = st.columns(2)
    with col1:
        st.video(h264_video_path, format='video/mp4', start_time=0, loop=True)


# Step 2 processing after user triggers
if 'original_video_path' in st.session_state and run_step_2:
    fps = 1 / exposure_time  # Calculate fps based on the exposure_time
    mask_path, magnitude_path = pixel_wise_fft_filtered_and_masked(st.session_state['original_video_path'], fps,
                                                                   st.session_state.freq_min, st.session_state.freq_max,
                                                                   st.session_state.mag_threshold)

    # Set up directories and output paths
    frames_output_dir = tempfile.mkdtemp()  # Temp directory for frames
    masked_video_path = os.path.join(frames_output_dir, 'masked_video.mp4')

    # Save masked frames and create video
    save_masked_frames(st.session_state['original_video_path'], mask_path, frames_output_dir, fps)
    create_video_from_frames(frames_output_dir, masked_video_path, fps)

    col1, col2 = st.columns(2)
    with col1:
        st.video(st.session_state['original_video_path'], format='video/mp4', start_time=0, loop=True)
        with open(st.session_state['original_video_path'], "rb") as file:
            st.download_button("Download Original Video", file.read(), file_name='Original_Video.mp4', key="download_orig_video")
        st.image(mask_path, caption="Mask Image")
        with open(mask_path, "rb") as file:
            st.download_button("Download Mask Image", file.read(), file_name='Mask.png', key="download_mask_image")

    with col2:
        st.video(masked_video_path, format='video/mp4', start_time=0, loop=True)
        with open(masked_video_path, "rb") as file:
            st.download_button("Download Masked Video", file.read(), file_name='Masked_Video.mp4', key="download_masked_video")
        st.image(magnitude_path, caption="Magnitude Map")
        with open(magnitude_path, "rb") as file:
            st.download_button("Download Magnitude Map", file.read(), file_name='Magnitude_Map.png', key="download_magnitude_map")
