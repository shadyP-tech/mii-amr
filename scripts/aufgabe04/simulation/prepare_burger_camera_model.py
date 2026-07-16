#!/usr/bin/env python3
"""Generate a simulation-only Burger camera SDF with valid pinhole optics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizontal-fov-rad", type=float, default=1.3962634)
    args = parser.parse_args()
    if not 0.1 < args.horizontal_fov_rad < 3.0:
        parser.error("horizontal FOV must be between 0.1 and 3.0 radians")
    text = args.source.read_text()
    patched, count = re.subn(
        r"<horizontal_fov>[^<]+</horizontal_fov>",
        f"<horizontal_fov>{args.horizontal_fov_rad:.7f}</horizontal_fov>",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("source SDF does not contain exactly one camera horizontal_fov")
    patched, sensor_count = re.subn(
        r'<sensor name="camera" type="wideanglecamera">',
        '<sensor name="camera" type="camera">',
        patched,
        count=1,
    )
    if sensor_count != 1:
        raise SystemExit("source SDF does not contain the expected wide-angle camera sensor")
    patched, visualize_count = re.subn(
        r'(<sensor name="camera" type="camera">.*?<visualize>)true(</visualize>)',
        r"\1false\2",
        patched,
        count=1,
        flags=re.DOTALL,
    )
    if visualize_count != 1:
        raise SystemExit("source SDF does not contain the expected camera visualize flag")
    patched, lens_count = re.subn(
        r"\s*<lens>.*?</lens>",
        "",
        patched,
        count=1,
        flags=re.DOTALL,
    )
    if lens_count != 1:
        raise SystemExit("source SDF does not contain the expected custom lens block")
    patched, width_count = re.subn(r"<width>320</width>", "<width>640</width>", patched, count=1)
    patched, height_count = re.subn(r"<height>240</height>", "<height>480</height>", patched, count=1)
    if width_count != 1 or height_count != 1:
        raise SystemExit("source SDF does not contain the expected 320x240 camera image")
    ground_truth_plugin = """
    <plugin name="aufgabe04_gazebo_ground_truth" filename="libgazebo_ros_p3d.so">
      <ros><remapping>odom:=/gazebo_ground_truth</remapping></ros>
      <body_name>base_footprint</body_name>
      <frame_name>world</frame_name>
      <update_rate>30</update_rate>
      <xyz_offset>0 0 0</xyz_offset>
      <rpy_offset>0 0 0</rpy_offset>
      <gaussian_noise>0</gaussian_noise>
    </plugin>
"""
    patched, model_count = re.subn(
        r"</model>",
        ground_truth_plugin + "  </model>",
        patched,
        count=1,
    )
    if model_count != 1:
        raise SystemExit("source SDF does not contain exactly one model closing tag")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
