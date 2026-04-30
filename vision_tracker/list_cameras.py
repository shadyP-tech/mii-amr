import cv2

import camera
import config


def main():
    max_index = 5

    print("Testing OpenCV/AVFoundation camera indices.")
    print("Press any key in each image window to continue.\n")

    for index in range(max_index):
        if config.CAMERA_FORCE_AVFOUNDATION:
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            print(f"{index}: not available")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        if config.CAMERA_FOURCC:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        frame = None
        ok = False
        for _ in range(max(1, config.CAMERA_WARMUP_FRAMES)):
            ok, next_frame = cap.read()
            if ok:
                frame = next_frame

        if not ok or frame is None:
            print(f"{index}: opened but no frame")
            cap.release()
            continue

        mean_saturation, channel_diff = camera.color_stats(frame)
        color_state = "color" if camera.looks_like_color(frame) else "grayscale-looking"
        print(
            f"{index}: {frame.shape[1]}x{frame.shape[0]} "
            f"{color_state} sat={mean_saturation:.1f} diff={channel_diff:.1f}"
        )

        cv2.imshow(f"camera {index}: {color_state}", frame)
        cv2.waitKey(0)
        cv2.destroyWindow(f"camera {index}: {color_state}")
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
