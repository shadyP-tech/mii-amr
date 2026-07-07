---
name: grill-me
description: >
  Rigorously question and align mii-amr implementation plans, project setup,
  ROS2/TurtleBot run protocols, data collection, testing, and report evidence
  before coding or running commands. Use when the user asks to be grilled,
  challenged, sanity-checked, aligned, or forced to clarify assumptions.
---

# Grill Me

## Purpose

Use this skill as an alignment gate before implementation or experiment work in
`mii-amr`. The goal is to make the coding agent and user agree on objective,
scope, environment, risks, validation, and evidence before changing code,
running robot commands, or committing to a protocol.

When this skill is invoked, ask pointed questions first. Do not edit files or
propose real robot motion until critical ambiguities are answered or explicitly
converted into stated assumptions.

## Style

- Be direct and rigorous, not adversarial.
- Prefer concrete yes/no or short-answer questions.
- Prioritize blockers over curiosity.
- Keep the question set bounded: normally 5 blocking questions and up to 8
  risk/quality questions.
- If a safe default exists, state it as a proposed assumption instead of asking.
- If the request is already clear, say what is clear and ask only the remaining
  high-value questions.

## Output Contract

For a plan, setup, or implementation idea, respond with:

1. `Alignment Snapshot`: summarize what the user appears to want in 2-4 bullets.
2. `Blocking Questions`: ask the questions that must be answered before work.
3. `Risk Questions`: ask questions that improve quality, safety, or evidence.
4. `Proposed Defaults`: list assumptions the agent will use if the user does
   not care.
5. `Definition Of Done`: state what success should look like.

For a very small task, compress this into a short checklist and 1-3 questions.

## Project-Specific Grill Checklist

### Scope And Intent

- What exact behavior should change, and what should remain unchanged?
- Is this an implementation task, a diagnostic explanation, a run protocol, a
  report/evidence task, or a code review?
- Is the goal a one-off experiment workaround or a durable repo pattern?
- Which files, scripts, commands, topics, or artifacts are in scope?
- What would be an unacceptable side effect?

### Environment And Runtime

- Which environment owns the work: `[MacBook]`, `[workstation/container]`, or
  `[UTM Ubuntu]`?
- Is ROS2/rclpy available, or should the work stay ROS-free/unit-testable?
- For TurtleBot/Nav2 work, are `/scan`, `/odom`, `/amcl_pose`, `/initialpose`,
  `/cmd_vel`, TF, and `/navigate_to_pose` expected to exist?
- Which `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, `TURTLEBOT3_MODEL`, and
  `LDS_MODEL` are expected?
- Is the current repo path `/workspace/mii-amr`, the MacBook project path, or
  another checkout?

### Safety And Motion

- Is any physical robot motion intended, or should this remain read-only/dry-run?
- If real motion is intended, who is operating the robot, where is the safety
  stop, and should `/cmd_vel` safety-stop terminal instructions be included?
- Are there competing `/cmd_vel` publishers or active Nav2 goals?
- Should visualization publishers be disabled during real motion if they affect
  timing?

### Aufgabe 03 Navigation

- Is the route the original waypoint CSV, a late-obstacle route, or a newly
  generated path?
- Should LiDAR replan happen at startup or only after a live obstacle appears?
  For delayed-obstacle validation, default to
  `--run-local-map-initial-scan-mode none`.
- For the current arena prior, is `--heater-wall-side=+x`,
  `--arena-force-short-wall-side axis_positive`, and
  `--arena-force-short-wall-type heater` the intended setup?
- Should arena-active RViz topics be published by the real runner, by the
  standalone read-only helper, or not at all?
- Which frame should RViz use: `map` for Nav2/static route or `odom` for
  arena-active temporary maps before AMCL has `map -> odom`?

### Data And Evidence

- Which generated files should prove success: CSV logs, run-local maps,
  waypoint/path CSVs, PPMs, RViz screenshots, bag files, or pasted terminal logs?
- Are pasted terminal logs acceptable as evidence, or must artifacts be copied
  into `results/`?
- Which CSV schema/version should the analysis expect?
- Should generated `results/` files be preserved, ignored, or committed?
- What report claim should this evidence support?

### Implementation Quality

- What is the narrowest code path that can satisfy the goal?
- What existing helper, model, CLI flag, or test pattern should be reused?
- What backwards compatibility matters for old commands or logs?
- What edge cases must fail loudly instead of silently falling back?
- Which unit tests, dry runs, or command help checks should verify the change?

## Red Flags To Surface

- The user asks for real robot motion without a safety stop and operator plan.
- The plan mixes MacBook-only work with ROS2/rclpy execution assumptions.
- A visualization helper publishes on the same topics as a real runner during
  motion.
- A late-obstacle test accidentally uses startup full-scan replanning.
- Forced arena heater/clean flags can mirror the localization prior.
- The report claim depends on pasted logs but no CSV/artifact evidence exists.
- The implementation changes generated result files or user-modified files
  without a clear reason.

## Closing The Grill

After the user answers, produce a short decision record:

- `Decisions`: what is now fixed.
- `Assumptions`: what remains assumed.
- `Plan`: the smallest safe implementation or run sequence.
- `Validation`: how success will be checked.

Then proceed only if the user asked for implementation or a run protocol.
