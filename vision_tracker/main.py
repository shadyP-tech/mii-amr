"""Publish the latest external camera pose for the real-run start gate."""

import cv2

import calibration
import camera
import config
import pose_estimator
import tracker


def _write_invalid_latest_pose(num_detected=0):
    missing = float("nan")
    pose_estimator.write_latest_pose(
        missing, missing, missing, valid_pose=False, num_detected=num_detected
    )


def main():
    _write_invalid_latest_pose()
    cap = None
    try:
        homography = calibration.load_homography()
        if homography is None:
            raise RuntimeError("No homography found. Run vision_tracker/calibration.py first.")
        cap = camera.open_camera()
        print(f"Publishing {config.LATEST_TRACKER_POSE_FILE}; Ctrl+C to stop.")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped returning frames.")
            frame = cv2.resize(frame, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE)
            centers, _ = tracker.detect_markers(frame)
            classified = pose_estimator.classify_markers([
                (calibration.pixel_to_world((x, y), homography), radius)
                for x, y, radius in centers
            ])
            if classified is None:
                _write_invalid_latest_pose(len(centers))
            else:
                pose_estimator.write_latest_pose(
                    *pose_estimator.estimate_pose(classified),
                    valid_pose=True,
                    num_detected=len(centers),
                )
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        if cap is not None:
            cap.release()
        _write_invalid_latest_pose()


if __name__ == "__main__":
    raise SystemExit(main())
