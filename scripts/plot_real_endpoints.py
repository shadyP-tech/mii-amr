#!/usr/bin/env python3

import csv
import statistics as stats

import matplotlib.pyplot as plt

CSV_PATH = "results/real_scripted_drive_runs.csv"

with open(CSV_PATH, newline="") as file:
    rows = list(csv.DictReader(file))

rows = [r for r in rows if r["run_id"] in {
    "run_real_06",
    "run_real_07",
    "run_real_08",
    "run_real_09",
    "run_real_10",
}]

xs = [float(r["tracker_final_x"]) for r in rows]
ys = [float(r["tracker_final_y"]) for r in rows]
labels = [r["run_id"] for r in rows]

mean_x = stats.mean(xs)
mean_y = stats.mean(ys)

plt.figure()
plt.scatter(xs, ys, label="real tracker final positions")
plt.scatter([mean_x], [mean_y], marker="x", s=100, label="mean")

for x, y, label in zip(xs, ys, labels):
    plt.text(x, y, label)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Final position after scripted drive - real TurtleBot")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.savefig("results/real_endpoint_plot.png", dpi=200, bbox_inches="tight")

print("Saved plot to results/real_endpoint_plot.png")