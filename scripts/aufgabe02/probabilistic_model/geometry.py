import math

from .csv_data import finite_float


def rotation_matrix(theta_rad):
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return [[c, -s], [s, c]]


def mat_vec(matrix, vector):
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def local_displacements(rows, prefix):
    displacements = []

    for row in rows:
        start = [
            finite_float(row, f"{prefix}_start_x"),
            finite_float(row, f"{prefix}_start_y"),
        ]
        final = [
            finite_float(row, f"{prefix}_final_x"),
            finite_float(row, f"{prefix}_final_y"),
        ]
        yaw_rad = math.radians(finite_float(row, f"{prefix}_start_yaw_deg"))
        delta_world = vec_sub(final, start)
        delta_local = mat_vec(rotation_matrix(-yaw_rad), delta_world)
        displacements.append(delta_local)

    return displacements


def motion_errors(local_delta, step_distance_m):
    command = [step_distance_m, 0.0]
    return [vec_sub(delta, command) for delta in local_delta]


def normalize_angle_deg(value):
    normalized = (value + 180.0) % 360.0 - 180.0
    if normalized == -180.0 and value > 0:
        return 180.0
    return normalized
