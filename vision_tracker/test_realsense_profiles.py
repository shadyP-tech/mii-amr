import subprocess
import sys
import textwrap


PROFILES = [
    ("rgb8", 640, 480, 30),
    ("bgr8", 640, 480, 30),
    ("rgb8", 1280, 720, 30),
    ("bgr8", 1280, 720, 30),
    ("rgb8", 424, 240, 30),
    ("bgr8", 424, 240, 30),
]


CHILD_CODE = r"""
import pyrealsense2 as rs

fmt_name, width, height, fps = {profile!r}

pipeline = rs.pipeline()
config = rs.config()
fmt = getattr(rs.format, fmt_name)
config.enable_stream(rs.stream.color, width, height, fmt, fps)

profile = pipeline.start(config)
try:
    for _ in range(30):
        frames = pipeline.wait_for_frames(5000)
        color = frames.get_color_frame()
        if color:
            data = color.get_data()
            print(
                f"ok shape={{color.get_width()}}x{{color.get_height()}} "
                f"bytes={{len(data)}}"
            )
            break
    else:
        raise RuntimeError("no color frame received")
finally:
    pipeline.stop()
"""


def run_profile(profile):
    code = CHILD_CODE.format(profile=profile)
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=20,
    )
    return result


def main():
    print(f"Python: {sys.executable}")
    print("Testing RealSense color profiles in child processes.\n")

    for profile in PROFILES:
        fmt_name, width, height, fps = profile
        label = f"{fmt_name} {width}x{height}@{fps}"
        try:
            result = run_profile(profile)
        except subprocess.TimeoutExpired:
            print(f"{label}: TIMEOUT")
            continue

        if result.returncode == 0:
            print(f"{label}: OK {result.stdout.strip()}")
        else:
            print(f"{label}: FAILED returncode={result.returncode}")
            stderr = result.stderr.strip()
            if stderr:
                print(textwrap.indent(stderr, "    "))


if __name__ == "__main__":
    main()
