# Aufgabe 04 Gazebo stands

This directory contains a portable Gazebo Classic world with static station
stands that match the physical green QR stands: four-foot green base, stem,
green head board, white QR panel, and black QR modules.

For Gazebo, the stand is intentionally scaled to the TurtleBot3 Burger
envelope: its top is 0.20 m above the floor and the QR panel is centred at
0.16 m. This is the robot-height simulation variant, not the metre-tall
physical stand shown in the reference photos.

The world is generated from the same station-layout JSON used by the offline
route planner. This keeps stand poses, station IDs, and approach targets in one
source of truth.

Generate or refresh the world from the checked-in `A/B/C` layout:

```bash
cd /Users/stephpark/Documents/stephsWorld/mii-amr
python3 -m scripts.aufgabe04.simulation.generate_gazebo_world
```

Start the static world:

```bash
gazebo --verbose simulation/gazebo/worlds/aufgabe04_stands.world
```

The world intentionally does not spawn a TurtleBot. Start the TurtleBot3
Gazebo launch used by your ROS 2 Humble installation with this world as its
world argument, or spawn a robot separately. The stand models are static and
their base, stem, board, and arena walls have collision geometry; QR modules
are visual-only so they cannot create artificial collision hits.

The QR payloads are the station IDs (`A`, `B`, and `C`) encoded as Version 1-L
QR codes. The stand's local +x direction is its QR-facing direction, and the
layout yaw rotates that face toward the final approach side.

To use a new random layout, generate it first and then pass it to the world
generator:

```bash
python3 scripts/aufgabe04/navigation/generate_random_station_layout.py \
  --station-count 3 --seed 42 \
  --output results/aufgabe04/layouts/random_station_layout.json
python3 -m scripts.aufgabe04.simulation.generate_gazebo_world \
  --layout results/aufgabe04/layouts/random_station_layout.json
```
