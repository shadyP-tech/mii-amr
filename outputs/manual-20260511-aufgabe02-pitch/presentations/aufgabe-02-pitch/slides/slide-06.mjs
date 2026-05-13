import { C, ROOT, base, callout, metric, source, title } from "./shared.mjs";

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Problem 5: composed route");
  title(slide, ctx, "The final route exposed the model limit.");

  await ctx.addImage(slide, {
    path: `${ROOT}/results/supervisor_route_prediction.png`,
    left: 60,
    top: 236,
    width: 640,
    height: 400,
    fit: "contain",
    alt: "Supervisor route endpoint prediction plot",
  });

  metric(slide, ctx, "0.149 m", "validation residual magnitude", 750, 272, 230, C.red);
  metric(slide, ctx, "False", "inside the 95% endpoint ellipse", 1006, 272, 190, C.red);
  metric(slide, ctx, "-10.7", "yaw residual in validation run 004, deg", 750, 426, 230, C.red);
  metric(slide, ctx, "0.498 m", "95% ellipse major-axis length", 1006, 426, 190, C.amber);

  callout(
    slide,
    ctx,
    "Decision after seeing this",
    "Use it as uncertainty evidence. The route failure motivates closed-loop localization.",
    750,
    546,
    448,
    100,
    C.teal,
  );

  source(slide, ctx, "results/supervisor_route_prediction_summary.csv; results/supervisor_route_validation_runs.csv");
  return slide;
}
