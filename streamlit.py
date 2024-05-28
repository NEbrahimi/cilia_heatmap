import streamlit as st
import cv2
from nd2reader import ND2Reader
import os
import tempfile
import subprocess
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import shutil
from scipy.signal import find_peaks
import pandas as pd

# Set page configuration and styles
st.set_page_config(page_title="Cilia Analysis Tool", layout="wide")
st.markdown(
    """
    <style>
    .css-18e3th9 {background-color: #F8F9FA;}
    .css-1d391kg {background-color: white;}
    </style>
    """, unsafe_allow_html=True)


def convert_nd2_to_video(input_file, output_dir, fps):
    with ND2Reader(input_file) as images:
        height, width = images.metadata['height'], images.metadata['width']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = os.path.splitext(os.path.basename(input_file.name))[0] + '.mp4'
        video_path = os.path.join(output_dir, video_filename)
        out = cv2.VideoWriter(video_path, fourcc, int(fps), (width, height))
        for frame in images:
            frame_8bit = cv2.normalize(frame, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_color = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)
            out.write(frame_color)
        out.release()
        return video_path


def convert_video_for_streamlit(input_path):
    output_path = input_path.replace('.mp4', '_compatible.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast', output_path], check=True)
    return output_path


def pixel_wise_fft_filtered_and_masked(video_path, fps):
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
    phase = np.angle(fft_cube)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(frames), d=1 / fps))
    valid_indices = (freqs > 2) & (freqs < 30)
    magnitude_filtered = magnitude[:, :, valid_indices]
    phase_filtered = phase[:, :, valid_indices]
    freqs_filtered = freqs[valid_indices]

    dominant_freq_indices = np.argmax(magnitude_filtered, axis=2)
    dominant_magnitude = np.max(magnitude_filtered, axis=2)
    dominant_phase = np.take_along_axis(phase_filtered, dominant_freq_indices[:, :, np.newaxis], axis=2).squeeze()
    dominant_frequencies = freqs_filtered[dominant_freq_indices]

    mask = (dominant_magnitude >= 300).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 10  # Minimum area threshold to keep a contour

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    mask_path = os.path.join(tempfile.gettempdir(), 'mask.png')
    cv2.imwrite(mask_path, filtered_mask)

    plt.figure(figsize=(15, 8))
    im = plt.imshow(dominant_frequencies, cmap='jet', interpolation='nearest', vmin=0, vmax=50)
    plt.colorbar(im, label='Dominant Frequency (Hz)')
    plt.title('Ciliary Beat Frequency Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    frequency_map_path = os.path.join(tempfile.gettempdir(), 'frequency_map.png')
    plt.savefig(frequency_map_path)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(dominant_magnitude, cmap='hot')
    plt.colorbar()
    plt.title('Dominant Frequency Magnitude (3-30 Hz)')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    magnitude_path = os.path.join(tempfile.gettempdir(), 'magnitude_map.png')
    plt.savefig(magnitude_path)
    plt.close()

    return mask_path, frequency_map_path, magnitude_path


def apply_mask_to_video(video_path, mask_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Mask image not found or unable to read.")
        return

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ret, frame = cap.read()
    if not ret:
        print("Unable to read video frame.")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, 333, (width, height))

    while ret:
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        out_video.write(masked_frame)
        ret, frame = cap.read()

    cap.release()
    out_video.release()


def pixel_wise_fft(video_path, fps, freq_min, freq_max):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Error opening video file")

    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()
    if not ret:
        capture.release()
        raise ValueError("Unable to read video frame")

    frame_height, frame_width = frame.shape[:2]
    pixel_time_series = np.zeros((frame_height, frame_width, num_frames), dtype=np.float32)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(num_frames):
        ret, frame = capture.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixel_time_series[:, :, i] = gray_frame

    capture.release()

    all_power_spectrum = np.abs(np.fft.fft(pixel_time_series.reshape(-1, num_frames), axis=-1))**2
    power_threshold = np.percentile(all_power_spectrum, 95)

    cbf_map = np.zeros((frame_height, frame_width))
    max_power_map = np.zeros((frame_height, frame_width))
    freq_amp_data = []

    freq_bins = np.fft.fftfreq(n=num_frames, d=1 / fps)
    amplitude_distribution = np.zeros_like(freq_bins)
    power_distribution = np.zeros_like(freq_bins)

    for i in range(frame_height):
        for j in range(frame_width):
            intensity_series = pixel_time_series[i, j, :]
            fft_result = np.fft.fft(intensity_series)
            fft_frequencies = np.fft.fftfreq(n=num_frames, d=1 / fps)
            positive_frequencies = fft_frequencies > 0
            power_spectrum = np.abs(fft_result[positive_frequencies]) ** 2
            amplitude_spectrum = np.abs(fft_result[positive_frequencies])

            significant_power_mask = power_spectrum > power_threshold
            significant_frequencies = fft_frequencies[positive_frequencies][significant_power_mask]
            significant_power = power_spectrum[significant_power_mask]
            significant_amplitude = amplitude_spectrum[significant_power_mask]

            freq_range_mask = (significant_frequencies >= freq_min) & (significant_frequencies <= freq_max)
            filtered_frequencies = significant_frequencies[freq_range_mask]
            filtered_power = significant_power[freq_range_mask]

            if filtered_frequencies.size > 0:
                max_power_idx = np.argmax(filtered_power)
                cbf_map[i, j] = filtered_frequencies[max_power_idx]
                max_power_map[i, j] = filtered_power[max_power_idx]

            for freq, power, amplitude in zip(significant_frequencies, significant_power, significant_amplitude):
                if freq_min <= freq <= freq_max:
                    freq_idx = np.argmin(np.abs(freq_bins - freq))
                    amplitude_distribution[freq_idx] += amplitude
                    power_distribution[freq_idx] += power

        max_amplitude_freq = freq_bins[np.argmax(amplitude_distribution)]
        max_power_freq = freq_bins[np.argmax(power_distribution)]

        return cbf_map, max_power_map, pd.DataFrame(freq_amp_data), max_amplitude_freq, max_power_freq


def calculate_statistics(valid_cbfs):
    if valid_cbfs.size > 0:
        median_cbf = np.median(valid_cbfs)
        std_cbf = np.std(valid_cbfs)
        p25_cbf = np.percentile(valid_cbfs, 25)
        p75_cbf = np.percentile(valid_cbfs, 75)

        return {
            'Mean': np.mean(valid_cbfs),
            'Median': median_cbf,
            '25%': p25_cbf,
            '75%': p75_cbf,
            'std': std_cbf
        }
    else:
        raise ValueError("No valid CBFs found. Adjust filtering criteria.")


def report_dominant_frequencies(max_amplitude_freq, max_power_freq):
    return max_amplitude_freq, max_power_freq


def save_maps(cbf_map, max_power_map, cbf_map_path, max_power_map_path):
    plt.figure(figsize=(10, 5))
    plt.imshow(cbf_map, cmap='jet')
    plt.colorbar(label='CBF (Hz)')
    plt.title('Ciliary Beat Frequency Map')
    plt.savefig(cbf_map_path)
    plt.close()
    if os.path.exists(cbf_map_path):
        print(f"CBF map saved to {cbf_map_path}")
    else:
        print(f"Failed to save CBF map to {cbf_map_path}")

    plt.figure(figsize=(10, 5))
    plt.imshow(max_power_map, cmap='hot')
    plt.colorbar(label='Power')
    plt.title('Maximum Power Spectrum Map')
    plt.savefig(max_power_map_path)
    plt.close()
    if os.path.exists(max_power_map_path):
        print(f"Max Power map saved to {max_power_map_path}")
    else:
        print(f"Failed to save Max Power map to {max_power_map_path}")


# Input widgets
with st.sidebar:
    st.title("Step 1: File Upload and Review")
    uploaded_file = st.file_uploader("Upload a .nd2 file", type=["nd2"])
    exposure_time = st.number_input("Exposure Time (seconds)", min_value=0.0001, value=0.003, step=0.0001,
                                    format="%.3f")
    run_step_1 = st.button("Run Step 1")

    st.title("Step 2: Masking")
    freq_min = st.number_input("Min Frequency", value=2, help="Set the minimum frequency for filtering.")
    freq_max = st.number_input("Max Frequency", value=30, help="Set the maximum frequency for filtering.")
    mag_threshold = st.number_input("Magnitude Threshold", value=300,
                                    help="Set the threshold for background detection sensitivity.")
    run_step_2 = st.button("Run Step 2")

    st.title("Step 3: Cilia Beat Frequency Analysis")
    video_source = st.radio("Select Video Source", options=['Original', 'Masked'], index=0,
                            help="Choose whether to use the original or masked video for analysis.")
    freq_filter_min = st.number_input("Frequency Filter Min", value=2, help="Minimum frequency for CBF analysis.")
    freq_filter_max = st.number_input("Frequency Filter Max", value=30, help="Maximum frequency for CBF analysis.")
    run_step_3 = st.button("Run Step 3")

# Step 1
if uploaded_file and exposure_time > 0 and run_step_1:
    fps = 1 / exposure_time
    if 'temp_dir' not in st.session_state:
        st.session_state['temp_dir'] = tempfile.mkdtemp()

    video_path = convert_nd2_to_video(uploaded_file, st.session_state['temp_dir'], fps)
    compatible_video_path = convert_video_for_streamlit(video_path)
    st.session_state['original_video_path'] = video_path
    st.session_state['compatible_video_path'] = compatible_video_path
    with open(compatible_video_path, "rb") as file:
        file_data = file.read()
    st.download_button("Download Original Video", file_data, compatible_video_path.split(os.path.sep)[-1])
    col1, col2 = st.columns(2)
    with col1:
        st.video(compatible_video_path, format='video/mp4', start_time=0, loop=True)


# Define the permanent storage path
storage_path = r"C:\Users\z3541106\codes\datasets\images\storage"
os.makedirs(storage_path, exist_ok=True)

# Step 2 processing after user triggers
if 'original_video_path' in st.session_state and run_step_2:
    fps = 1 / exposure_time
    mask_path, frequency_map_path, magnitude_path = pixel_wise_fft_filtered_and_masked(st.session_state['original_video_path'], fps)

    frames_output_dir = tempfile.mkdtemp()
    masked_video_path = os.path.join(frames_output_dir, 'masked_video.mp4')

    apply_mask_to_video(st.session_state['original_video_path'], mask_path, masked_video_path)
    compatible_masked_video_path = convert_video_for_streamlit(masked_video_path)

    original_video_permanent_path = os.path.join(storage_path, 'original_video.mp4')
    masked_video_permanent_path = os.path.join(storage_path, 'masked_video.mp4')
    shutil.copy(st.session_state['original_video_path'], original_video_permanent_path)
    shutil.copy(masked_video_path, masked_video_permanent_path)

    st.session_state['original_video_permanent_path'] = original_video_permanent_path
    st.session_state['masked_video_permanent_path'] = masked_video_permanent_path
    st.session_state['compatible_masked_video_path'] = compatible_masked_video_path

    col1, col2 = st.columns(2)
    with col1:
        st.video(st.session_state['compatible_video_path'], format='video/mp4', start_time=0, loop=True)
        with open(st.session_state['original_video_path'], "rb") as file:
            st.download_button("Download Original Video", file.read(), file_name='Original_Video.mp4', key="download_orig_video")
        st.image(mask_path, caption="Mask Image")
        with open(mask_path, "rb") as file:
            st.download_button("Download Mask Image", file.read(), file_name='Mask.png', key="download_mask_image")

    with col2:
        st.video(st.session_state['compatible_masked_video_path'], format='video/mp4', start_time=0, loop=True)
        with open(masked_video_path, "rb") as file:
            st.download_button("Download Masked Video", file.read(), file_name='Masked_Video.mp4', key="download_masked_video")
        st.image(frequency_map_path, caption="Frequency Map")
        with open(frequency_map_path, "rb") as file:
            st.download_button("Download Frequency Map", file.read(), file_name='Frequency_Map.png', key="download_frequency_map")
        st.image(magnitude_path, caption="Magnitude Map")
        with open(magnitude_path, "rb") as file:
            st.download_button("Download Magnitude Map", file.read(), file_name='Magnitude_Map.png', key="download_magnitude_map")

# Step 3 processing
if 'original_video_permanent_path' in st.session_state and run_step_3:
    fps = 1 / exposure_time
    video_path = st.session_state['original_video_permanent_path']
    if video_source == 'Masked':
        video_path = st.session_state.get('masked_video_permanent_path', video_path)

    cbf_map, max_power_map, freq_amp_df, max_amplitude_freq, max_power_freq = pixel_wise_fft(video_path, fps, freq_filter_min, freq_filter_max)
    valid_cbfs = cbf_map[cbf_map > 0]

    stats = calculate_statistics(valid_cbfs)
    stats['Dominant Frequency (Amplitude)'], stats['Dominant Frequency (Power)'] = report_dominant_frequencies(max_amplitude_freq, max_power_freq)

    cbf_map_path = os.path.join(tempfile.gettempdir(), 'cbf_map.png')
    max_power_map_path = os.path.join(tempfile.gettempdir(), 'max_power_map.png')
    save_maps(cbf_map, max_power_map, cbf_map_path, max_power_map_path)

    freq_amp_df.to_csv(os.path.join(tempfile.gettempdir(), 'freq_amp_data.csv'), index=False)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cbf_map_path, caption="Ciliary Beat Frequency Map")
        st.image(max_power_map_path, caption="Maximum Power Spectrum Map")

    with col2:
        st.table(pd.DataFrame(stats, index=[0]))
        st.write(f"Dominant Frequency (Amplitude): {max_amplitude_freq:.2f} Hz")
        st.write(f"Dominant Frequency (Power): {max_power_freq:.2f} Hz")

    with open(os.path.join(tempfile.gettempdir(), 'freq_amp_data.csv'), "rb") as file:
        st.download_button("Download Frequency and Amplitude Data", file.read(), file_name='freq_amp_data.csv', key="download_freq_amp_data")

    with open(cbf_map_path, "rb") as file:
        st.download_button("Download CBF Map", file.read(), file_name='cbf_map.png', key="download_cbf_map")

    with open(max_power_map_path, "rb") as file:
        st.download_button("Download Max Power Map", file.read(), file_name='max_power_map.png', key="download_max_power_map")






