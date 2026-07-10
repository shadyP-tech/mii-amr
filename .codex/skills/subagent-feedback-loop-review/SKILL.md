---
name: subagent-feedback-loop-review
description: Iterative read-only review workflow for mii-amr ideas and plans using the repo-local reviewer subagents. Use when the user asks to review, sanity-check, approve, or revise an implementation idea, run protocol, robotics plan, Aufgabe 04 design, or codebase placement through local subagents before coding.
---

# Subagent Feedback Loop Review

## Purpose

Use this skill to run a bounded feedback loop over an idea or plan before any
implementation. The loop gathers read-only subagent reviews, revises the plan
minimally, and then performs a final placement review only after the plan is
approved or conditionally approved.

Do not implement code, modify files, run robot motion, or change generated
artifacts while using this skill.

## Reviewer Set

Default to the local project reviewers that are available in this checkout:

1. `aufgabe04_task_mapper`: requirements/protocol fit for Aufgabe 04.
2. `aufgabe04_architecture_reviewer`: module boundaries, codebase shape, testability, and bloat risks.
3. `nav_safety_reviewer`: navigation safety, Nav2 handoff, waypoint following, and real-robot motion risks.
4. `turtlebot_platform_expert`: ROS2 Humble/TurtleBot topic, frame, namespace, and hardware-readiness risks.
5. `test_evidence_auditor`: tests, logs, evidence, and report-readiness.

Add focused reviewers when the idea touches their scope:

- `qr_camera_auditor`: QR scanning, onboard camera, station order parsing, camera integration.
- `fleet_coordination_reviewer`: two-robot behavior, station locks, right-before-left policy, collision rules.

If the user provides an explicit reviewer list, use that list only for agent
roles that are callable in the current session. Do not simulate missing
subagents. If a required reviewer is unavailable, report the missing role and
mark the loop `BLOCKED` unless the user explicitly authorizes an available
replacement.

## Verdicts

Require every reviewer to return exactly one verdict:

- `PASS`: no required changes.
- `CONDITIONAL_PASS`: acceptable if guardrails, tests, artifact checks, or TODOs are recorded.
- `NEEDS_REVISION`: directionally valid but needs concrete changes.
- `NEEDS_REDESIGN`: core design is wrong or incoherent.
- `BLOCKED`: protocol, safety, leakage, math, claim-boundary, or placement violation.

Prefer the stricter verdict when reviewers disagree. Treat missing safety,
requirements, or evidence checks as conditional items at minimum.

## Iteration Workflow

### Iteration 1

Spawn all required reviewers in parallel when possible. Give each reviewer the
same idea or plan and require this compact response shape:

```text
verdict:
blocking issue:
required revision:
conditional guardrail/test/TODO:
notes:
```

Summarize every verdict, blocking issue, required revision, and conditional
guardrail. Revise the plan only for feedback with verdicts
`NEEDS_REVISION`, `NEEDS_REDESIGN`, or `BLOCKED`.

Keep revisions minimal and traceable. Preserve the user's intent unless a
reviewer identifies a concrete violation or incoherent design choice.

### Iterations 2 And 3

Always rerun `aufgabe04_task_mapper` as the project protocol/requirements gate.
Rerun only the reviewers whose feedback caused a revision. Do not rerun
unaffected reviewers.

Stop after approval criteria are met or after 3 total iterations.

## Placement Review

If the review loop is approved or conditionally approved, create the final
revised plan first. Then run final placement review.

For this checkout, use `aufgabe04_architecture_reviewer` as the codebase
placement reviewer unless the session exposes a dedicated
`codebase_placement_reviewer`. The placement pass must focus on exact files,
module ownership, tests, runbooks, and artifact locations, not research or task
redesign.

If placement review finds protocol or requirements risk, rerun
`aufgabe04_task_mapper`. If placement review finds architecture risk, rerun
`aufgabe04_architecture_reviewer`. Apply the same 3-iteration cap to the main
loop; placement-triggered reruns should be narrow and must not reopen
unaffected review areas.

## Approval Criteria

Report `approved` only when:

- protocol/requirements gate is `PASS`;
- every other required reviewer is `PASS` or `CONDITIONAL_PASS`;
- placement review is `PASS`;
- all conditional items are converted into guardrails, tests, artifact checks, or TODOs.

Report `conditionally approved` only when:

- protocol/requirements gate is `PASS`;
- no reviewer remains `NEEDS_REVISION`, `NEEDS_REDESIGN`, or `BLOCKED`;
- placement review is `PASS` or `CONDITIONAL_PASS`;
- at least one conditional item has recorded guardrails.

Report `not approved` when:

- protocol/requirements gate is not `PASS`; or
- any reviewer remains `NEEDS_REVISION`, `NEEDS_REDESIGN`, or `BLOCKED` after review or placement.

## Final Output

Return exactly these sections:

1. `Final Verdict`
2. `Final Revised Plan`
3. `Codebase Placement Map`
4. `Implementation Guardrails / Tests / TODOs`

Keep the final answer read-only. Include missing reviewers or unavailable
subagent roles under `Final Verdict` when they affect approval.
