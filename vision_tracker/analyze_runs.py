#!/usr/bin/env python3

import csv
import math
import statistics as stats

CSV_PATH = "results/real_scripted_drive_runs.csv"


def f(row, key):
    return float(row[key])


with open(CSV_PATH, newline="") as file:
    rows = list(csv.DictReader(file))

# Use only clean rows
rows = [r for r in rows if r["run_id"] in {
    "run_real_06",
    "run_real_07",
    "run_real_08",
    "run_real_09",
    "run_real_10",
}]

print(f"Number of runs: {len(rows)}")

tracker_final_x = [f(r, "tracker_final_x") for r in rows]
tracker_final_y = [f(r, "tracker_final_y") for r in rows]
tracker_final_yaw = [f(r, "tracker_final_yaw_deg") for r in rows]

tracker_start_x = [f(r, "tracker_start_x") for r in rows]
tracker_start_y = [f(r, "tracker_start_y") for r in rows]
tracker_start_yaw = [f(r, "tracker_start_yaw_deg") for r in rows]

odom_final_x = [f(r, "odom_final_x") for r in rows]
odom_final_y = [f(r, "odom_final_y") for r in rows]
odom_final_yaw = [f(r, "odom_final_yaw_deg") for r in rows]


def summarize(name, values):
    print(f"{name}:")
    print(f"  mean = {stats.mean(values):.6f}")
    print(f"  std  = {stats.stdev(values):.6f}" if len(values) > 1 else "  std  = 0.000000")


print("\nTracker start pose:")
summarize("start_x", tracker_start_x)
summarize("start_y", tracker_start_y)
summarize("start_yaw_deg", tracker_start_yaw)

print("\nTracker final pose:")
summarize("final_x", tracker_final_x)
summarize("final_y", tracker_final_y)
summarize("final_yaw_deg", tracker_final_yaw)

print("\nOdometry final pose:")
summarize("odom_final_x", odom_final_x)
summarize("odom_final_y", odom_final_y)
summarize("odom_final_yaw_deg", odom_final_yaw)

mean_x = stats.mean(tracker_final_x)
mean_y = stats.mean(tracker_final_y)

print("\nPer-run tracker final deviation from mean:")
for r in rows:
    dx = f(r, "tracker_final_x") - mean_x
    dy = f(r, "tracker_final_y") - mean_y
    dist = math.sqrt(dx * dx + dy * dy)
    print(f"{r['run_id']}: dx={dx:.6f}, dy={dy:.6f}, distance={dist:.6f}")

print("\nNet tracker displacement per run:")
for r in rows:
    dx = f(r, "tracker_final_x") - f(r, "tracker_start_x")
    dy = f(r, "tracker_final_y") - f(r, "tracker_start_y")
    dist = math.sqrt(dx * dx + dy * dy)
    dyaw = f(r, "tracker_final_yaw_deg") - f(r, "tracker_start_yaw_deg")
    print(f"{r['run_id']}: dx={dx:.6f}, dy={dy:.6f}, distance={dist:.6f}, dyaw={dyaw:.3f} deg")