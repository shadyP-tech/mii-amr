import { C, base, metric, source, title } from "./shared.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Decision 4: model scope");
  title(slide, ctx, "I chose a deliberately small empirical model.", "The task asked for a rough endpoint prediction. Tradeoff: no full trajectory, map uncertainty, or full x/y/yaw covariance.");

  metric(slide, ctx, "30", "valid F30 forward runs", 92, 286, 165, C.blue);
  metric(slide, ctx, "0.302 m", "mean local forward displacement", 294, 286, 230, C.blue);
  metric(slide, ctx, "-0.002 m", "mean lateral displacement", 560, 286, 230, C.blue);
  metric(slide, ctx, "+0.32 deg", "mean yaw drift after F30", 826, 286, 240, C.blue);

  ctx.addShape(slide, { left: 94, top: 442, width: 1080, height: 1.5, fill: C.line, line: ctx.line() });
  metric(slide, ctx, "-85.0 deg", "CW90 measured mean yaw change", 130, 480, 260, C.amber);
  metric(slide, ctx, "+84.6 deg", "CCW90 measured mean yaw change", 510, 480, 280, C.amber);
  metric(slide, ctx, "10k", "Monte Carlo samples for path endpoint", 900, 480, 210, C.amber);

  source(slide, ctx, "results/probabilistic_motion_primitives_model_summary.csv; scripts/build_motion_primitives_model.py");
  return slide;
}
