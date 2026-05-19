import math

from .constants import CHI2_95_2D, NEAR_SINGULAR_COND, NEAR_SINGULAR_DET
from .geometry import mat_vec, normalize_angle_deg, vec_sub


def empirical_mean_cov(points):
    points = as_points(points)
    if len(points) < 2:
        raise ValueError("At least 2 valid points are required for covariance")

    mu = [
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    ]

    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for point in points:
        dx = point[0] - mu[0]
        dy = point[1] - mu[1]
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy

    denom = len(points) - 1
    sigma = [[sxx / denom, sxy / denom], [sxy / denom, syy / denom]]
    return mu, sigma


def as_points(points):
    normalized = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Expected 2D points")
        x = float(point[0])
        y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Point array contains non-finite values")
        normalized.append([x, y])
    return normalized


def circular_yaw_summary_deg(values_deg):
    values = [float(value) for value in values_deg]
    if not values:
        raise ValueError("At least one yaw value is required")

    radians = [math.radians(value) for value in values]
    sin_mean = sum(math.sin(value) for value in radians) / len(radians)
    cos_mean = sum(math.cos(value) for value in radians) / len(radians)
    mean_rad = math.atan2(sin_mean, cos_mean)
    resultant = math.hypot(sin_mean, cos_mean)
    resultant = min(max(resultant, 1e-12), 1.0)
    std_rad = math.sqrt(-2.0 * math.log(resultant))

    return {
        "mean_deg": normalize_angle_deg(math.degrees(mean_rad)),
        "std_deg": math.degrees(std_rad),
    }


def symmetric_eigen_2x2(matrix):
    a = float(matrix[0][0])
    b = 0.5 * (float(matrix[0][1]) + float(matrix[1][0]))
    d = float(matrix[1][1])
    trace_half = 0.5 * (a + d)
    diff_half = 0.5 * (a - d)
    radius = math.hypot(diff_half, b)
    eig1 = trace_half + radius
    eig2 = trace_half - radius

    vec1 = eigenvector_for_value(a, b, d, eig1)
    vec2 = [-vec1[1], vec1[0]]
    return [eig1, eig2], [vec1, vec2]


def eigenvector_for_value(a, b, d, eig):
    if abs(b) > 1e-15 or abs(eig - a) > 1e-15:
        vector = [b, eig - a]
    else:
        vector = [1.0, 0.0] if a >= d else [0.0, 1.0]

    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return [1.0, 0.0]
    return [vector[0] / norm, vector[1] / norm]


def covariance_warning(sigma):
    det = determinant_2x2(sigma)
    eigvals, _ = symmetric_eigen_2x2(sigma)
    max_abs = max(abs(value) for value in eigvals)
    min_abs = min(abs(value) for value in eigvals)
    condition = math.inf if min_abs == 0.0 else max_abs / min_abs

    if abs(det) < NEAR_SINGULAR_DET or condition > NEAR_SINGULAR_COND:
        return (
            "Covariance is singular or nearly singular; "
            "using a pseudo-inverse for Mahalanobis distances."
        )

    return None


def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def pseudo_inverse_symmetric_2x2(matrix):
    eigvals, eigvecs = symmetric_eigen_2x2(matrix)
    result = [[0.0, 0.0], [0.0, 0.0]]
    for eigval, eigvec in zip(eigvals, eigvecs):
        if abs(eigval) <= 1e-15:
            continue
        inv = 1.0 / eigval
        result[0][0] += inv * eigvec[0] * eigvec[0]
        result[0][1] += inv * eigvec[0] * eigvec[1]
        result[1][0] += inv * eigvec[1] * eigvec[0]
        result[1][1] += inv * eigvec[1] * eigvec[1]
    return result


def mahalanobis_squared(points, mu, sigma):
    points = as_points(points)
    sigma_inv = pseudo_inverse_symmetric_2x2(sigma)
    distances = []
    for point in points:
        residual = vec_sub(point, mu)
        weighted = mat_vec(sigma_inv, residual)
        distances.append(residual[0] * weighted[0] + residual[1] * weighted[1])
    return distances


def outlier_records(rows, points, mu, sigma, threshold=CHI2_95_2D):
    distances = mahalanobis_squared(points, mu, sigma)
    outliers = []
    for row, distance in zip(rows, distances):
        if distance > threshold:
            outliers.append(
                {
                    "run_id": row.get("run_id", ""),
                    "mahalanobis_squared": float(distance),
                }
            )
    return outliers


def ellipse_parameters(mu, sigma, chi2_value=CHI2_95_2D):
    eigvals, eigvecs = symmetric_eigen_2x2(sigma)
    eigvals = [max(value, 0.0) for value in eigvals]
    semi_axes = [math.sqrt(value * chi2_value) for value in eigvals]
    angle_rad = math.atan2(eigvecs[0][1], eigvecs[0][0])

    return {
        "center": [float(mu[0]), float(mu[1])],
        "chi2_value": float(chi2_value),
        "semi_major_m": float(semi_axes[0]),
        "semi_minor_m": float(semi_axes[1]),
        "major_axis_length_m": float(2.0 * semi_axes[0]),
        "minor_axis_length_m": float(2.0 * semi_axes[1]),
        "orientation_deg": normalize_angle_deg(math.degrees(angle_rad)),
        "area_m2": float(math.pi * semi_axes[0] * semi_axes[1]),
    }


def matrix_std(sigma):
    return [math.sqrt(max(sigma[0][0], 0.0)), math.sqrt(max(sigma[1][1], 0.0))]
