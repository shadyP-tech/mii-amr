import math
import struct
import zlib
from pathlib import Path

from .constants import CHI2_95_2D
from .geometry import mat_vec, rotation_matrix, vec_add
from .statistics import ellipse_parameters


def write_fallback_plot(path, groups, ellipses=None, polylines=None):
    ellipses = ellipses or []
    polylines = polylines or []
    width = 900
    height = 700
    margin = 70
    white = (255, 255, 255)
    pixels = [[white for _ in range(width)] for _ in range(height)]

    all_points = []
    for points, _color in groups:
        all_points.extend(points)
    for points, _color in polylines:
        all_points.extend(points)
    for mu, sigma, _color in ellipses:
        all_points.extend(ellipse_sample_points(mu, sigma))

    if not all_points:
        all_points = [[0.0, 0.0], [1.0, 1.0]]

    to_pixel = plot_transform(all_points, width, height, margin)

    zero = to_pixel([0.0, 0.0])
    draw_line(pixels, (margin, zero[1]), (width - margin, zero[1]), (220, 220, 220))
    draw_line(pixels, (zero[0], margin), (zero[0], height - margin), (220, 220, 220))

    for points, color in polylines:
        pixel_points = [to_pixel(point) for point in points]
        for start, end in zip(pixel_points, pixel_points[1:]):
            draw_line(pixels, start, end, color)
        for point in pixel_points:
            draw_circle(pixels, point, 4, color)

    for mu, sigma, color in ellipses:
        pixel_points = [to_pixel(point) for point in ellipse_sample_points(mu, sigma)]
        for start, end in zip(pixel_points, pixel_points[1:] + pixel_points[:1]):
            draw_line(pixels, start, end, color)

    for points, color in groups:
        for point in points:
            draw_circle(pixels, to_pixel(point), 4, color)

    write_png(path, pixels)


def ellipse_sample_points(mu, sigma, chi2_value=CHI2_95_2D, count=96):
    params = ellipse_parameters(mu, sigma, chi2_value=chi2_value)
    angle = math.radians(params["orientation_deg"])
    rot = rotation_matrix(angle)
    a = params["semi_major_m"]
    b = params["semi_minor_m"]
    points = []
    for index in range(count):
        t = 2.0 * math.pi * index / count
        local = [a * math.cos(t), b * math.sin(t)]
        points.append(vec_add(mu, mat_vec(rot, local)))
    return points


def plot_transform(points, width, height, margin):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    padding_x = 0.12 * span_x
    padding_y = 0.12 * span_y
    min_x -= padding_x
    max_x += padding_x
    min_y -= padding_y
    max_y += padding_y
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    pixel_center_x = width / 2
    pixel_center_y = height / 2

    def to_pixel(point):
        x = int(round(pixel_center_x + (point[0] - center_x) * scale))
        y = int(round(pixel_center_y - (point[1] - center_y) * scale))
        return x, y

    return to_pixel


def draw_circle(pixels, center, radius, color):
    cx, cy = center
    height = len(pixels)
    width = len(pixels[0])
    for y in range(cy - radius, cy + radius + 1):
        if y < 0 or y >= height:
            continue
        for x in range(cx - radius, cx + radius + 1):
            if x < 0 or x >= width:
                continue
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                pixels[y][x] = color


def draw_line(pixels, start, end, color):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    height = len(pixels)
    width = len(pixels[0])

    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def write_png(path, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend([r, g, b])

    def chunk(chunk_type, data):
        payload = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
