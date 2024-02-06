import cv2
import numpy as np
import vision
import arguments_parser

def main():
    """
    Whole heatmap pipeline creation with an additional step to mask the original video.
    """
    parser = arguments_parser.prepare_parser()
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video_file)
    background_subtractor = cv2.createBackgroundSubtractorKNN(dist2Threshold=120)

    read_success, video_frame = capture.read()
    if not read_success:
        print("Failed to read the first frame from the video.")
        return

    height, width, _ = video_frame.shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(args.video_output + ".mp4", fourcc, 100.0, (width, height))
    accumulated_image = np.zeros((height, width), np.uint8)
    count = 0
    frame_count = 0  # Initialize frame_count before the loop starts

    while read_success:
        background_filter = background_subtractor.apply(video_frame)
        if frame_count == 0:
            cv2.imwrite('background_subtracted.png', background_filter)
        frame_count += 1

        if count > args.video_skip and count % args.take_every == 0:
            eroded_image = vision.apply_morph(background_filter, morph_type=cv2.MORPH_ERODE, kernel_size=(3, 3))
            accumulated_image = vision.add_images(accumulated_image, eroded_image)

        count += 1
        read_success, video_frame = capture.read()

    cv2.imwrite('background_subtracted.png', background_filter)

    # Normalize and apply heatmap colors
    normalized_image = vision.normalize_image(accumulated_image)
    heatmap_image = vision.apply_heatmap_colors(normalized_image)

    # Save the final heatmap image
    heatmap_png_path = args.video_output + ".png"
    cv2.imwrite(heatmap_png_path, normalized_image)

    capture.release()
    video.release()
    cv2.destroyAllWindows()

    # Apply the heatmap as a mask to the original video
    apply_heatmap_mask(args.video_file, heatmap_png_path, args.video_output + "_masked.mp4")


def apply_heatmap_mask(video_path, mask_image_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Mask image not found or unable to read.")
        return

    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite('thresholded_mask.png', mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 0  # No area threshold, keep all contours

    # Remove small contours
    for contour in contours:
        if cv2.contourArea(contour) < area_threshold:
            cv2.drawContours(mask, [contour], -1, (0,), thickness=cv2.FILLED)

    # Save the final mask to verify its correctness
    mask_image_debug_path = output_video_path.replace("_masked.mp4", "_mask.png")
    cv2.imwrite(mask_image_debug_path, mask)

    ret, frame = cap.read()
    if not ret:
        print("Unable to read video frame.")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    out_video = cv2.VideoWriter(output_video_path, fourcc, 100.0, (width, height))

    while ret:
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        out_video.write(masked_frame)
        ret, frame = cap.read()

    cap.release()
    out_video.release()

if __name__ == '__main__':
    main()
