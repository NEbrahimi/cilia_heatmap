import cv2
import numpy as np

cap = cv2.VideoCapture('C:/Users/z3541106/codes/datasets/cilia/heatmap/output/mask/test_video_image_100_100_t2.mp4_masked.mp4')

# Define the grid size
grid_size = 7
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cell_width = frame_width // grid_size
cell_height = frame_height // grid_size

# Define the codec and create VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('C:/Users/z3541106/codes/datasets/cilia/heatmap/output/grid/test_video_image_100_100_t2_grid_7.mp4', fourcc, 100.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for i in range(1, grid_size):
        # Vertical lines
        start_point_v = (i * cell_width, 0)
        end_point_v = (i * cell_width, frame_height)
        frame = cv2.line(frame, start_point_v, end_point_v, (0, 255, 255), 1)

        # Horizontal lines
        start_point_h = (0, i * cell_height)
        end_point_h = (frame_width, i * cell_height)
        frame = cv2.line(frame, start_point_h, end_point_h, (0, 255, 255), 1)

        # Write the frame into the file 'output_video.mp4'
    out.write(frame)

# Release everything when the job is finished
cap.release()
out.release()