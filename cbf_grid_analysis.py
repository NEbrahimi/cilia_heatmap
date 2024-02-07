import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram


video_path = 'C:/Users/z3541106/codes/datasets/cilia/heatmap/output/sparse/test_video.mp4'

plot_output_path = 'C:/Users/z3541106/codes/datasets/cilia/heatmap/output/sparse/grid_raw/'

# Define the grid size
grid_size = 7

capture = cv2.VideoCapture(video_path)
fps = 199.56  # Update with the FPS of your video

frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
cell_width = frame_width // grid_size
cell_height = frame_height // grid_size

intensity_values_grid = np.zeros((grid_size, grid_size), dtype=object)

for i in range(grid_size):
    for j in range(grid_size):
        intensity_values_grid[i, j] = []

while True:
    ret, frame = capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute average intensity for each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            roi = gray_frame[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            avg_intensity = np.mean(roi)
            intensity_values_grid[i, j].append(avg_intensity)

capture.release()

# Analyze each grid cell
for i in range(grid_size):
    for j in range(grid_size):
        intensity_array = np.array(intensity_values_grid[i, j])

        # Apply periodogram
        frequencies, power_spectrum = periodogram(intensity_array, fs=fps)

        # Filter out frequencies below a given Hz
        filter_mask = frequencies >= 0
        filtered_frequencies = frequencies[filter_mask]
        filtered_power_spectrum = power_spectrum[filter_mask]

        # Find the CBF for the grid cell
        cbf = filtered_frequencies[np.argmax(filtered_power_spectrum)]

        # Plot
        plt.figure()
        plt.plot(frequencies, power_spectrum)
        plt.title(f'Power Spectral Density - Grid {i},{j}')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.xlim(0, max(frequencies))
        plt.xticks(np.arange(0, max(frequencies) + 1, 10))
        plt.text(max(frequencies) * 0.7, max(power_spectrum) * 0.9, f'CBF: {cbf:.2f} Hz', fontsize=12)

        # Save the plot
        plt.savefig(f'{plot_output_path}grid_{i}_{j}_cbf.png')
        plt.close()

        # Print CBF value
        print(f"Grid {i},{j} - Ciliary Beat Frequency: {cbf} Hz")
