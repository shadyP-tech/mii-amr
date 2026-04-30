import cv2
import numpy as np
import config
import camera

def nothing(x):
    pass

def main():
    try:
        cap = camera.open_camera()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    cv2.namedWindow('Trackbars')
    
    # Initialize trackbars with current config values
    cv2.createTrackbar('HMin', 'Trackbars', config.HSV_LOWER[0], 179, nothing)
    cv2.createTrackbar('SMin', 'Trackbars', config.HSV_LOWER[1], 255, nothing)
    cv2.createTrackbar('VMin', 'Trackbars', config.HSV_LOWER[2], 255, nothing)
    
    cv2.createTrackbar('HMax', 'Trackbars', config.HSV_UPPER[0], 179, nothing)
    cv2.createTrackbar('SMax', 'Trackbars', config.HSV_UPPER[1], 255, nothing)
    cv2.createTrackbar('VMax', 'Trackbars', config.HSV_UPPER[2], 255, nothing)

    print("=====================================================")
    print("Adjust the sliders until the 3 green circles appear")
    print("as solid white blobs in the 'mask' window, and the")
    print("background is completely black.")
    print("Press ESC when you are happy with the mask.")
    print("=====================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read current positions of all trackbars
        h_min = cv2.getTrackbarPos('HMin', 'Trackbars')
        s_min = cv2.getTrackbarPos('SMin', 'Trackbars')
        v_min = cv2.getTrackbarPos('VMin', 'Trackbars')
        
        h_max = cv2.getTrackbarPos('HMax', 'Trackbars')
        s_max = cv2.getTrackbarPos('SMax', 'Trackbars')
        v_max = cv2.getTrackbarPos('VMax', 'Trackbars')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Cleanup mask slightly for easier viewing
        kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask_clean)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("\n✅ Update config.py lines 23-24 with these values:")
            print(f"HSV_LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"HSV_UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
