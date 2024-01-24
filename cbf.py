import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

capture = cv2.VideoCapture('C:/Users/z3541106/codes/datasets/cilia/heatmap/static-11-23-116AS-D21-Baseline-F1_001.mp4')

intensity_values = []

while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray_frame

    avg_intensity = np.mean(roi)
    intensity_values.append(avg_intensity)

capture.release()

intensity_array = np.array(intensity_values)

fps = 331.75

# Apply periodogram
frequencies, power_spectrum = periodogram(intensity_array, fs=fps)

plt.plot(frequencies, power_spectrum)
plt.title('Power Spectral Density')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
#plt.show()

# Set x-axis limits and intervals
plt.xlim(0, max(frequencies))  # Adjust the max value as needed
plt.xticks(np.arange(0, max(frequencies) + 1, 10))

cbf = frequencies[np.argmax(power_spectrum)]

# Determine position for text annotation
x_position = max(frequencies) * 0.7  # Adjust as needed
y_position = max(power_spectrum) * 0.9  # Adjust as needed

# Add CBF text annotation to the plot
plt.text(x_position, y_position, f'CBF: {cbf:.2f} Hz', fontsize=12)  # Adjust x_position, y_position, and fontsize as needed

# Show and save the plot
plt.savefig('C:/Users/z3541106/codes/datasets/cilia/heatmap/output/static-11-23-116AS-D21-Baseline-F1_001.png')  # Specify your save path here
plt.show()

print(f"Ciliary Beat Frequency: {cbf} Hz")
