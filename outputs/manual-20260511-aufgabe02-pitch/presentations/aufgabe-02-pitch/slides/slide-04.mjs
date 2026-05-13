import { C, base, callout, source, title } from "./shared.mjs";

function lane(ctx, slide, label, left, top, color, rows) {
  ctx.addShape(slide, { left, top, width: 500, height: 250, fill: "#FFFFFF", line: { style: "solid", fill: C.line, width: 1.2 } });
  ctx.addShape(slide, { left, top, width: 500, height: 42, fill: color, line: ctx.line() });
  ctx.addText(slide, {
    text: label,
    left: left + 18,
    top: top + 10,
    width: 460,
    height: 24,
    fontSize: 18,
    color: C.white,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  rows.forEach((row, i) => {
    const y = top + 62 + i * 54;
    ctx.addText(slide, {
      text: row[0],
      left: left + 22,
      top: y,
      width: 150,
      height: 26,
      fontSize: 18,
      color,
      bold: true,
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
    ctx.addText(slide, {
      text: row[1],
      left: left + 172,
      top: y,
      width: 300,
      height: 36,
      fontSize: 19,
      color: C.ink,
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
  });
}

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Problem 3: coupling commands");
  title(slide, ctx, "The same /cmd_vel command hid different stop logic.", "That mattered because the assignment asked for real motion coupled to perfect simulation commands.");

  lane(ctx, slide, "Simulation run", 76, 292, C.blue, [
    ["input", "RUN_DISTANCE and RUN_SPEED"],
    ["stop", "drive until odometry reaches target distance"],
    ["validate", "start pose, forward distance, lateral drift, yaw drift"],
  ]);
  lane(ctx, slide, "Real robot run", 704, 292, C.teal, [
    ["input", "same command vocabulary: speed, duration, angle"],
    ["stop", "time / primitive execution on hardware"],
    ["measure", "external tracker start and final pose plus odometry"],
  ]);

  callout(
    slide,
    ctx,
    "Decision",
    "Log both odometry and camera tracker fields, then compare endpoint displacement instead of assuming either source is perfect.",
    120,
    566,
    1040,
    74,
    C.amber,
  );

  source(slide, ctx, "scripts/scripted_drive.py; scripts/real_scripted_drive.py; scripts/run_real_experiment.sh");
  return slide;
}
