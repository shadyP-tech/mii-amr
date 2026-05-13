import { C, base, callout, metric, source, title } from "./shared.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Problem 2: repeatable starts");
  title(slide, ctx, "The start pose became a control variable.", "Without a gate, small manual placement differences would look like motion-model error.");

  metric(slide, ctx, "0.04 m", "position tolerance before a real run can start", 86, 300, 210, C.teal);
  metric(slide, ctx, "4 deg", "yaw tolerance before a real run can start", 326, 300, 210, C.teal);
  metric(slide, ctx, "1 s", "stable tracker pose required", 566, 300, 210, C.teal);
  metric(slide, ctx, "3", "markers required for a valid camera pose", 806, 300, 210, C.teal);

  ctx.addShape(slide, { left: 82, top: 500, width: 980, height: 2, fill: C.line, line: ctx.line() });
  ctx.addShape(slide, { left: 82, top: 488, width: 26, height: 26, fill: C.teal, line: ctx.line() });
  ctx.addShape(slide, { left: 522, top: 488, width: 26, height: 26, fill: C.amber, line: ctx.line() });
  ctx.addShape(slide, { left: 1036, top: 488, width: 26, height: 26, fill: C.red, line: ctx.line() });
  ctx.addText(slide, {
    text: "accepted starts logged with measured dx/dy/yaw error",
    left: 118,
    top: 484,
    width: 380,
    height: 34,
    fontSize: 18,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addText(slide, {
    text: "stale / invalid / missing marker poses block the run",
    left: 558,
    top: 484,
    width: 430,
    height: 34,
    fontSize: 18,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  ctx.addShape(slide, { left: 90, top: 578, width: 6, height: 64, fill: C.blue, line: ctx.line() });
  ctx.addText(slide, {
    text: "Decision",
    left: 110,
    top: 580,
    width: 180,
    height: 26,
    fontSize: 18,
    color: C.blue,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addText(slide, {
    text: "Make the run script wait for camera-confirmed pose instead of trusting manual placement.",
    left: 110,
    top: 610,
    width: 780,
    height: 36,
    fontSize: 21,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  source(slide, ctx, "vision_tracker/config.py; vision_tracker/start_pose_gate.py; results/real_start_pose_checks.csv");
  return slide;
}
