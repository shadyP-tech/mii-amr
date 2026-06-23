# Aufgabe 04 Perception Debug Viewer

This tool is debug-only. It visualizes camera frames, HSV masks, masked previews, and ROI color classification results for stand-color threshold tuning.

It does not command robot motion, does not publish `/cmd_vel`, does not send Nav2 goals, and does not execute station approach behavior.

Viewer output is not real-parkour validation evidence. Real-robot Aufgabe 04 claims require the separate real-parkour checklist, run logs, and recorded mission evidence.

Keep all OpenCV window, camera, and live-debug code in this debug package. Pure perception logic belongs in `scripts/aufgabe04/perception/color_classifier.py` and should be covered by offline tests only.

## Usage

Run this only when the robot is stationary or physically secured. Confirm no autonomous mission, Nav2 goal, custom follower, or station-route runner is active.

```bash
python3 -m scripts.aufgabe04.perception.debug.color_mask_viewer \
  --camera-index 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --resize 0.7 \
  --color green \
  --roi 420,220,220,180 \
  --tune
```

Useful keys:

- `p`: print the active `ColorRange(...)` threshold.
- `s`: save frame, mask, and preview snapshots when `--save-snapshot` is set.
- `q` or `Esc`: quit.

Tune thresholds in the actual lighting where the stands will be seen. Prefer selecting a stand ROI over classifying the full frame, because full-frame classification dilutes confidence with background pixels.

