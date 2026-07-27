#!/usr/bin/env python3
"""Generate a simulation-only Burger camera SDF with valid pinhole optics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _resolve_model_resource_uris(text: str, resource_root: Path | None) -> str:
    """Replace Gazebo model URIs with validated absolute simulation assets."""

    if resource_root is None:
        return text
    resolved_root = resource_root.resolve()
    common_root = (resolved_root / "turtlebot3_common").resolve()
    if not common_root.is_dir():
        raise SystemExit(
            "TurtleBot model resource root does not contain turtlebot3_common: "
            f"{resolved_root}"
        )

    def replace(match: re.Match[str]) -> str:
        relative = Path(match.group(1))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(
                "unsafe turtlebot3_common model resource URI: "
                f"{match.group(0)}"
            )
        packaged_resource = common_root / relative
        if not packaged_resource.is_file():
            raise SystemExit(
                "TurtleBot model resource does not exist: "
                f"{packaged_resource}"
            )
        # Colcon's install tree may expose each mesh as a symlink to the source
        # workspace. Resolve it only after the sanitized package-relative path
        # has been validated so gzserver receives the actual readable file.
        resource = packaged_resource.resolve()
        return f"<uri>{resource.as_uri()}</uri>"

    resolved, count = re.subn(
        r"<uri>model://turtlebot3_common/([^<]+)</uri>",
        replace,
        text,
    )
    if count == 0:
        raise SystemExit(
            "source SDF does not contain turtlebot3_common model resource URIs"
        )
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizontal-fov-rad", type=float, default=1.3962634)
    parser.add_argument(
        "--model-resource-root",
        type=Path,
        default=None,
        help=(
            "Optional Gazebo models directory used to replace "
            "model://turtlebot3_common URIs with validated absolute file URIs."
        ),
    )
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
    if args.model_resource_root is not None:
        patched = _resolve_model_resource_uris(patched, args.model_resource_root)
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
