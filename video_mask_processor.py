import cv2
import numpy as np
import vision
import arguments_parser


def main():
    """
    Whole heatmap pipeline creation with additional masked video output.
    """
    parser = arguments_parser.prepare_parser()
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video_file)
    background_subtractor = cv2.createBackgroundSubtractorKNN(dist2Threshold=70)

    read_success, video_frame = capture.read()
    if not read_success:
        print("Error reading video")
        return

    height, width, _ = video_frame.shape
    frames_number = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    video_heatmap = cv2.VideoWriter(args.video_output + "_heatmap.mp4", fourcc, 60.0, (width, height))
    video_masked = cv2.VideoWriter(args.video_output + "_masked.mp4", fourcc, 60.0, (width, height))

    accumulated_image = np.zeros((height, width), np.uint8)
    count = 0

    while read_success:
        read_success, video_frame = capture.read()
        if read_success:
            background_filter = background_subtractor.apply(video_frame)

            if count > args.video_skip and count % args.take_every == 0:
                erodated_image = vision.apply_morph(background_filter, morph_type=cv2.MORPH_ERODE, kernel_size=(3, 3))
                accumulated_image = vision.add_images(accumulated_image, erodated_image)
                normalized_image = vision.normalize_image(accumulated_image)
                heatmap_image = vision.apply_heatmap_colors(normalized_image)
                frames_merged = vision.superimpose(heatmap_image, video_frame, args.video_alpha)

                # Write to heatmap video
                video_heatmap.write(frames_merged)

                # Create and write to masked video
                _, mask = cv2.threshold(erodated_image, 1, 255, cv2.THRESH_BINARY)
                masked_frame = cv2.bitwise_and(video_frame, video_frame, mask=mask)
                video_masked.write(masked_frame)

                if not args.video_disable:
                    cv2.imshow("Main", frames_merged)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if count % 100 == 0:
                    print(f"Progress: {count}/{frames_number}")

            count += 1

    # Save the final heatmap image and release resources
    cv2.imwrite(args.video_output + ".png", heatmap_image)
    capture.release()
    video_heatmap.release()
    video_masked.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
