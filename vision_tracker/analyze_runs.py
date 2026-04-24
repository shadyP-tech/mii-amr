"""
analyze_runs.py — Compare vision tracking data against odometry.

Usage:
    python3 analyze_runs.py data/vision_*.csv data/odom_*.csv
    python3 analyze_runs.py data/vision_20260424_120000.csv data/odom_20260424_120000.csv

Produces:
    - X-Y trajectory overlay (vision vs odom)
    - Yaw over time
    - Position error over time
    - Final-position error summary

Supports multiple vision+odom pairs for building a probabilistic model
of the final position (Aufgabe 2).
"""

import csv
import sys
import math
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found — plots will be skipped.")
    print("         Install with:  pip install matplotlib\n")


def load_vision_csv(path):
    """Load a vision CSV and return arrays of valid poses.

    Returns
    -------
    dict with keys: t, x, y, yaw  (numpy arrays, only valid_pose==1 rows)
    """
    t, x, y, yaw = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["valid_pose"]) != 1:
                continue
            t.append(float(row["timestamp"]))
            x.append(float(row["pose_x"]))
            y.append(float(row["pose_y"]))
            yaw.append(float(row["pose_yaw"]))
    return {
        "t": np.array(t),
        "x": np.array(x),
        "y": np.array(y),
        "yaw": np.array(yaw),
    }


def load_odom_csv(path):
    """Load an odometry CSV.

    Returns
    -------
    dict with keys: t, x, y, yaw  (numpy arrays)
    """
    t, x, y, yaw = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["timestamp"]))
            x.append(float(row["odom_x"]))
            y.append(float(row["odom_y"]))
            yaw.append(float(row["odom_yaw"]))
    return {
        "t": np.array(t),
        "x": np.array(x),
        "y": np.array(y),
        "yaw": np.array(yaw),
    }


def align_by_time(vision, odom):
    """Align odom data to vision timestamps using nearest-neighbor.

    Returns
    -------
    (vision, odom_aligned) with same length arrays.
    """
    aligned_x, aligned_y, aligned_yaw = [], [], []

    for vt in vision["t"]:
        idx = np.argmin(np.abs(odom["t"] - vt))
        aligned_x.append(odom["x"][idx])
        aligned_y.append(odom["y"][idx])
        aligned_yaw.append(odom["yaw"][idx])

    odom_aligned = {
        "t": vision["t"].copy(),
        "x": np.array(aligned_x),
        "y": np.array(aligned_y),
        "yaw": np.array(aligned_yaw),
    }
    return vision, odom_aligned


def compute_errors(vision, odom):
    """Compute per-timestep position and yaw errors.

    Returns
    -------
    dict with keys: pos_error, yaw_error  (numpy arrays)
    """
    dx = vision["x"] - odom["x"]
    dy = vision["y"] - odom["y"]
    pos_error = np.sqrt(dx ** 2 + dy ** 2)

    # Wrap yaw difference to [-π, π]
    yaw_diff = vision["yaw"] - odom["yaw"]
    yaw_error = np.abs(np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff)))

    return {
        "pos_error": pos_error,
        "yaw_error": yaw_error,
    }


def print_summary(vision, odom, errors):
    """Print a text summary of the comparison."""
    print("=" * 60)
    print("  Vision vs Odometry — Summary")
    print("=" * 60)

    if len(vision["t"]) == 0:
        print("  No valid vision data.")
        return

    t_rel = vision["t"] - vision["t"][0]

    print(f"  Duration:         {t_rel[-1]:.1f} s")
    print(f"  Valid frames:     {len(vision['t'])}")
    print()

    # Final positions
    print(f"  Vision  final:    x={vision['x'][-1]:.4f}  y={vision['y'][-1]:.4f}  "
          f"yaw={math.degrees(vision['yaw'][-1]):.1f}°")
    print(f"  Odom    final:    x={odom['x'][-1]:.4f}  y={odom['y'][-1]:.4f}  "
          f"yaw={math.degrees(odom['yaw'][-1]):.1f}°")

    final_err = errors["pos_error"][-1]
    final_yaw_err = math.degrees(errors["yaw_error"][-1])
    print(f"\n  Final pos error:  {final_err:.4f} m")
    print(f"  Final yaw error:  {final_yaw_err:.1f}°")

    print(f"\n  Mean pos error:   {np.mean(errors['pos_error']):.4f} m")
    print(f"  Max pos error:    {np.max(errors['pos_error']):.4f} m")
    print(f"  Mean yaw error:   {math.degrees(np.mean(errors['yaw_error'])):.1f}°")
    print("=" * 60)


def plot_comparison(vision, odom, errors):
    """Create a 3-panel comparison plot."""
    if not HAS_MPL:
        return

    t_rel = vision["t"] - vision["t"][0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) X-Y trajectory
    ax = axes[0]
    ax.plot(vision["x"], vision["y"], "g.-", label="Vision", alpha=0.7)
    ax.plot(odom["x"], odom["y"], "b.-", label="Odometry", alpha=0.7)
    ax.plot(vision["x"][0], vision["y"][0], "ko", ms=8, label="Start")
    ax.plot(vision["x"][-1], vision["y"][-1], "rs", ms=8, label="Vision end")
    ax.plot(odom["x"][-1], odom["y"][-1], "bs", ms=8, label="Odom end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2) Yaw over time
    ax = axes[1]
    ax.plot(t_rel, np.degrees(vision["yaw"]), "g-", label="Vision")
    ax.plot(t_rel, np.degrees(odom["yaw"]), "b-", label="Odometry")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw (°)")
    ax.set_title("Heading")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Position error
    ax = axes[2]
    ax.plot(t_rel, errors["pos_error"] * 100, "r-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Vision–Odom error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/comparison_plot.png", dpi=150)
    print(f"Plot saved to data/comparison_plot.png")
    plt.show()


def main():
    if len(sys.argv) < 3:
        print("Usage:  python3 analyze_runs.py <vision.csv> <odom.csv>")
        print("        python3 analyze_runs.py data/vision_*.csv data/odom_*.csv")
        sys.exit(1)

    vision_path = sys.argv[1]
    odom_path = sys.argv[2]

    print(f"Vision: {vision_path}")
    print(f"Odom:   {odom_path}\n")

    vision = load_vision_csv(vision_path)
    odom = load_odom_csv(odom_path)

    if len(vision["t"]) == 0:
        print("ERROR: No valid vision data in CSV.")
        sys.exit(1)
    if len(odom["t"]) == 0:
        print("ERROR: No odom data in CSV.")
        sys.exit(1)

    vision, odom_aligned = align_by_time(vision, odom)
    errors = compute_errors(vision, odom_aligned)

    print_summary(vision, odom_aligned, errors)
    plot_comparison(vision, odom_aligned, errors)


if __name__ == "__main__":
    main()
