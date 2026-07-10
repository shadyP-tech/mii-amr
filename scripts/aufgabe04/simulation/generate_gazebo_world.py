"""Generate a self-contained Gazebo Classic world containing Aufgabe 04 stands.

The generated stands use the same station pose convention as the pure station
layout code: the station yaw points from the stand towards the robot's final
approach direction.  The QR face is the local +x face, so the stand model is
rotated by the station yaw.

The QR geometry is deliberately built from SDF boxes instead of an external
texture.  This keeps the world portable and makes the QR panel visible to a
simulated camera without requiring a Gazebo model path or image-material
installation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


QR_SIZE = 21
QR_DATA_CODEWORDS = 19
QR_EC_CODEWORDS = 7
GREEN = "0.0 0.65 0.20 1"
GREEN_DARK = "0.0 0.36 0.10 1"
WHITE = "0.96 0.96 0.96 1"
BLACK = "0.01 0.01 0.01 1"
WALL = "0.55 0.55 0.55 1"

# Uniformly scale the original physical-stand model to a TurtleBot3
# Burger-sized envelope.  Keeping one scale factor for every axis avoids the
# broad, compressed silhouette produced when only the height is reduced.
STAND_HEIGHT_M = 0.20
PHYSICAL_STAND_HEIGHT_M = 1.43
STAND_SCALE = STAND_HEIGHT_M / PHYSICAL_STAND_HEIGHT_M


def _scaled(value: float) -> float:
    return value * STAND_SCALE


BOARD_CENTER_Z_M = _scaled(1.18)
BOARD_HEIGHT_M = _scaled(0.50)
BOARD_WIDTH_M = _scaled(0.50)
QR_PANEL_SIZE_M = _scaled(0.38)


def _bch_remainder(value: int, polynomial: int) -> int:
    """Return the BCH remainder used by QR format information."""

    value <<= polynomial.bit_length() - 1
    while value.bit_length() >= polynomial.bit_length():
        value ^= polynomial << (value.bit_length() - polynomial.bit_length())
    return value


def _format_bits(mask: int = 0) -> int:
    # Version 1-L uses format error-correction bits 01.
    value = (0b01 << 3) | mask
    return ((value << 10) | _bch_remainder(value, 0b10100110111)) ^ 0x5412


def _gf256_multiply(left: int, right: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        left = ((left << 1) ^ (0x11D if left & 0x80 else 0)) & 0xFF
    return result


def _reed_solomon_remainder(data: Sequence[int], generator: Sequence[int]) -> list[int]:
    remainder = [0] * (len(generator) - 1)
    for codeword in data:
        factor = codeword ^ remainder[0]
        remainder = remainder[1:] + [0]
        for index, coefficient in enumerate(generator[1:]):
            remainder[index] ^= _gf256_multiply(coefficient, factor)
    return remainder


def _qr_codewords(payload: str) -> list[int]:
    encoded = payload.encode("utf-8")
    if len(encoded) > 17:
        raise ValueError("Version 1-L QR payload must be at most 17 UTF-8 bytes")
    bits = [0, 1, 0, 0]  # byte mode
    bits.extend((len(encoded) >> shift) & 1 for shift in range(7, -1, -1))
    for byte in encoded:
        bits.extend((byte >> shift) & 1 for shift in range(7, -1, -1))
    capacity = QR_DATA_CODEWORDS * 8
    bits.extend([0] * min(4, capacity - len(bits)))
    bits.extend([0] * ((8 - len(bits) % 8) % 8))
    data = [
        sum(bits[offset + bit] << (7 - bit) for bit in range(8))
        for offset in range(0, len(bits), 8)
    ]
    pads = (0xEC, 0x11)
    pad_index = 0
    while len(data) < QR_DATA_CODEWORDS:
        data.append(pads[pad_index % 2])
        pad_index += 1
    generator = (1, 87, 229, 146, 149, 238, 102, 21)
    return data + _reed_solomon_remainder(data, generator)


def qr_matrix(payload: str, mask: int = 0) -> tuple[tuple[int, ...], ...]:
    """Build a Version 1-L QR matrix for a short station payload.

    The result includes finder/timing patterns, format information, masking,
    and Reed-Solomon error correction.  ``1`` denotes a black module.
    """

    if not 0 <= mask <= 7:
        raise ValueError("QR mask must be between 0 and 7")
    matrix: list[list[int | None]] = [[None] * QR_SIZE for _ in range(QR_SIZE)]

    def set_module(row: int, col: int, value: int) -> None:
        if 0 <= row < QR_SIZE and 0 <= col < QR_SIZE:
            matrix[row][col] = 1 if value else 0

    def reserve_finder(top: int, left: int) -> None:
        for row in range(top - 1, top + 8):
            for col in range(left - 1, left + 8):
                if 0 <= row < QR_SIZE and 0 <= col < QR_SIZE:
                    # The one-module white separator is part of the reserved
                    # function area and remains white.
                    inside = top <= row < top + 7 and left <= col < left + 7
                    if inside:
                        edge = row in (top, top + 6) or col in (left, left + 6)
                        set_module(row, col, int(edge or (top + 2 <= row <= top + 4 and left + 2 <= col <= left + 4)))
                    else:
                        set_module(row, col, 0)

    reserve_finder(0, 0)
    reserve_finder(0, QR_SIZE - 7)
    reserve_finder(QR_SIZE - 7, 0)

    for index in range(8, QR_SIZE - 8):
        set_module(6, index, index % 2 == 0)
        set_module(index, 6, index % 2 == 0)

    # Reserve the format-information locations.  They are filled after the
    # data mask is applied.
    # Reserve the two standard copies of format information.  The ordering
    # mirrors the QR placement algorithm: the first eight bits go to the
    # upper-left vertical and upper-right horizontal runs; the remaining
    # seven go to the lower-left vertical and upper-left horizontal runs.
    format_coordinates: list[tuple[int, int]] = []
    for index in range(15):
        if index < 6:
            format_coordinates.append((index, 8))
        elif index < 8:
            format_coordinates.append((index + 1, 8))
        else:
            format_coordinates.append((QR_SIZE - 15 + index, 8))
        if index < 8:
            format_coordinates.append((8, QR_SIZE - index - 1))
        elif index == 8:
            format_coordinates.append((8, 7))
        else:
            format_coordinates.append((8, 15 - index - 1))
    for row, col in format_coordinates:
        matrix[row][col] = None
    matrix[QR_SIZE - 8][8] = 1  # fixed dark module

    codewords = _qr_codewords(payload)
    data_bits = [
        (codeword >> shift) & 1
        for codeword in codewords
        for shift in range(7, -1, -1)
    ]
    bit_index = 0
    col = QR_SIZE - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(QR_SIZE - 1, -1, -1) if upward else range(QR_SIZE)
        for row in rows:
            for current_col in (col, col - 1):
                if matrix[row][current_col] is not None:
                    continue
                bit = data_bits[bit_index] if bit_index < len(data_bits) else 0
                bit_index += 1
                if mask == 0 and (row + current_col) % 2 == 0:
                    bit ^= 1
                matrix[row][current_col] = bit
        upward = not upward
        col -= 2

    format_value = _format_bits(mask)
    format_bits = [(format_value >> index) & 1 for index in range(15)]
    for index, bit in enumerate(format_bits):
        vertical_coordinate = format_coordinates[index * 2]
        horizontal_coordinate = format_coordinates[index * 2 + 1]
        # The QR bit stream is placed least-significant bit first.
        matrix[vertical_coordinate[0]][vertical_coordinate[1]] = bit
        matrix[horizontal_coordinate[0]][horizontal_coordinate[1]] = bit

    return tuple(tuple(int(value or 0) for value in row) for row in matrix)


def _pose(x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f} 0 0 {yaw:.6f}"


def _material(color: str) -> str:
    return f"""<material>
          <ambient>{color}</ambient>
          <diffuse>{color}</diffuse>
          <specular>0.08 0.08 0.08 1</specular>
        </material>"""


def _box_visual(name: str, pose: str, size: tuple[float, float, float], color: str) -> str:
    sx, sy, sz = size
    return f"""<visual name=\"{name}\">
        <pose>{pose}</pose>
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
        {_material(color)}
      </visual>"""


def _box_collision(name: str, pose: str, size: tuple[float, float, float]) -> str:
    sx, sy, sz = size
    return f"""<collision name=\"{name}\">
        <pose>{pose}</pose>
        <geometry><box><size>{sx:.6f} {sy:.6f} {sz:.6f}</size></box></geometry>
      </collision>"""


def _qr_visuals(station_id: str) -> str:
    matrix = qr_matrix(station_id)
    module = QR_PANEL_SIZE_M / 29.0
    visuals = [
        _box_visual(
            "qr_white_panel",
            _pose(_scaled(0.023), 0.0, BOARD_CENTER_Z_M),
            (_scaled(0.004), QR_PANEL_SIZE_M, QR_PANEL_SIZE_M),
            WHITE,
        )
    ]
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            if not value:
                continue
            # Viewed from the QR-facing local +x side, image-right must follow
            # increasing QR columns. The previous sign mirrored every code and
            # moved the lower-left finder to the lower-right.
            y = ((col + 4) - 10.5) * module
            z = BOARD_CENTER_Z_M + (10.5 - (row + 4)) * module
            visuals.append(
                _box_visual(
                    f"qr_{row:02d}_{col:02d}",
                    _pose(_scaled(0.026), y, z),
                    (_scaled(0.002), module, module),
                    BLACK,
                )
            )
    return "\n      ".join(visuals)


def _station_model(station: Mapping[str, object]) -> str:
    station_id = str(station["station_id"]).strip().upper()
    x = float(station["x_m"])
    y = float(station["y_m"])
    yaw = float(station["yaw_rad"])
    visuals = [
        _box_visual(
            "base_center",
            _pose(0, 0, _scaled(0.025)),
            (_scaled(0.30), _scaled(0.30), _scaled(0.05)),
            GREEN,
        ),
        _box_visual(
            "neck",
            _pose(0, 0, _scaled(0.105)),
            (_scaled(0.16), _scaled(0.16), _scaled(0.16)),
            GREEN_DARK,
        ),
        _box_visual(
            "stem",
            _pose(0, 0, _scaled(0.53)),
            (_scaled(0.065), _scaled(0.065), _scaled(0.82)),
            GREEN,
        ),
        _box_visual(
            "head_board",
            _pose(0, 0, BOARD_CENTER_Z_M),
            (_scaled(0.04), BOARD_WIDTH_M, BOARD_HEIGHT_M),
            GREEN,
        ),
    ]
    collisions = [
        _box_collision(
            "base_center",
            _pose(0, 0, _scaled(0.025)),
            (_scaled(0.30), _scaled(0.30), _scaled(0.05)),
        ),
        _box_collision(
            "stem",
            _pose(0, 0, _scaled(0.53)),
            (_scaled(0.065), _scaled(0.065), _scaled(0.82)),
        ),
        _box_collision(
            "head_board",
            _pose(0, 0, BOARD_CENTER_Z_M),
            (_scaled(0.04), BOARD_WIDTH_M, BOARD_HEIGHT_M),
        ),
    ]
    for index, arm_yaw in enumerate((0.785398, 2.356194, 3.926991, 5.497787)):
        arm_pose = _pose(0.0, 0.0, _scaled(0.025), arm_yaw)
        arm_size = (_scaled(0.68), _scaled(0.115), _scaled(0.05))
        visuals.append(_box_visual(f"base_arm_{index}", arm_pose, arm_size, GREEN))
        collisions.append(_box_collision(f"base_arm_{index}", arm_pose, arm_size))
    visuals.append(_qr_visuals(station_id))
    return f"""<model name=\"station_{station_id}\">
    <pose>{_pose(x, y, 0.0, yaw)}</pose>
    <static>true</static>
    <self_collide>false</self_collide>
    <link name=\"stand_link\">
      {' '.join(visuals)}
      {' '.join(collisions)}
    </link>
  </model>"""


def world_sdf(stations: Iterable[Mapping[str, object]]) -> str:
    stations = tuple(stations)
    station_models = "\n  ".join(_station_model(station) for station in stations)
    return f"""<?xml version=\"1.0\" ?>
<sdf version=\"1.6\">
  <world name=\"aufgabe04_stands\">
    <gravity>0 0 -9.81</gravity>
    <physics name=\"default_physics\" type=\"ode\">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.75 0.75 0.75 1</background>
      <shadows>true</shadows>
    </scene>
    <light name=\"sun\" type=\"directional\">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 4 0.35 -0.45 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.3 0.2 -1.0</direction>
    </light>
    <model name=\"arena_floor\">
      <static>true</static>
      <link name=\"floor_link\">
        <visual name=\"floor_visual\">
          <pose>0 0 -0.025 0 0 0</pose>
          <geometry><box><size>3.9 1.898 0.05</size></box></geometry>
          {_material("0.16 0.16 0.16 1")}
        </visual>
        <collision name=\"floor_collision\">
          <pose>0 0 -0.025 0 0 0</pose>
          <geometry><box><size>3.9 1.898 0.05</size></box></geometry>
        </collision>
      </link>
    </model>
    <model name=\"arena_walls\">
      <static>true</static>
      <link name=\"walls_link\">
        {_box_visual("wall_left", _pose(-1.95, 0, 0.20), (0.04, 1.898, 0.40), WALL)}
        {_box_visual("wall_right", _pose(1.95, 0, 0.20), (0.04, 1.898, 0.40), WALL)}
        {_box_visual("wall_front", _pose(0, -0.949, 0.20), (3.9, 0.04, 0.40), WALL)}
        {_box_visual("wall_back", _pose(0, 0.949, 0.20), (3.9, 0.04, 0.40), WALL)}
        {_box_collision("wall_left", _pose(-1.95, 0, 0.20), (0.04, 1.898, 0.40))}
        {_box_collision("wall_right", _pose(1.95, 0, 0.20), (0.04, 1.898, 0.40))}
        {_box_collision("wall_front", _pose(0, -0.949, 0.20), (3.9, 0.04, 0.40))}
        {_box_collision("wall_back", _pose(0, 0.949, 0.20), (3.9, 0.04, 0.40))}
      </link>
    </model>
  {station_models}
  </world>
</sdf>
"""


def _stations_from_layout(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    stations = payload.get("stations")
    if not isinstance(stations, list) or not stations:
        raise ValueError(f"layout must contain a non-empty stations list: {path}")
    required = {"station_id", "x_m", "y_m", "yaw_rad"}
    for index, station in enumerate(stations):
        if not isinstance(station, dict) or not required.issubset(station):
            raise ValueError(f"stations[{index}] is missing one of {sorted(required)}")
    return stations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path("results/aufgabe04/layouts/random_station_layout.json"),
        help="Station layout JSON produced by generate_random_station_layout.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("simulation/gazebo/worlds/aufgabe04_stands.world"),
        help="Output SDF world path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(world_sdf(_stations_from_layout(args.layout)))
    print(f"Wrote {output} with layout {args.layout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
