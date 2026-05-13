import { C, ROOT, arrow, base, callout, smallBox, source, title } from "./shared.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Problem 1: measuring pose");
  title(slide, ctx, "I could not treat odometry as the answer.", "The task needed final position in the physical experiment, so I needed an external reference frame.");

  callout(
    slide,
    ctx,
    "Decision",
    "Use an overhead camera, three green markers, and a homography to write the latest absolute tracker pose.",
    70,
    276,
    410,
    128,
    C.teal,
  );
  callout(
    slide,
    ctx,
    "Why",
    "It made simulation endpoints, real endpoints, and repeated-run statistics comparable in meters.",
    70,
    444,
    410,
    112,
    C.blue,
  );

  const y = 366;
  smallBox(slide, ctx, "camera frame", 552, y, 150, 82, C.white);
  arrow(slide, ctx, 714, y + 41, 764, C.teal);
  smallBox(slide, ctx, "HSV green marker detection", 774, y, 204, 82, C.white);
  arrow(slide, ctx, 990, y + 41, 1040, C.teal);
  smallBox(slide, ctx, "pixel -> world homography", 1050, y, 172, 82, C.white);

  smallBox(slide, ctx, "classify 3 markers", 610, 500, 172, 76, C.paper2);
  arrow(slide, ctx, 792, 538, 850, C.amber);
  smallBox(slide, ctx, "estimate x, y, yaw", 860, 500, 172, 76, C.paper2);
  arrow(slide, ctx, 1042, 538, 1090, C.amber);
  smallBox(slide, ctx, "latest tracker pose CSV", 1100, 500, 122, 76, C.paper2);

  ctx.addText(slide, {
    text: "Key code path",
    left: 552,
    top: 282,
    width: 320,
    height: 28,
    fontSize: 18,
    color: C.muted,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addText(slide, {
    text: "vision_tracker/main.py -> tracker.py -> calibration.py -> pose_estimator.py",
    left: 552,
    top: 316,
    width: 610,
    height: 28,
    fontSize: 15,
    color: C.muted,
    typeface: ctx.fonts.mono,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  source(slide, ctx, `${ROOT}/vision_tracker/main.py; pose_estimator.write_latest_pose()`);
  return slide;
}
