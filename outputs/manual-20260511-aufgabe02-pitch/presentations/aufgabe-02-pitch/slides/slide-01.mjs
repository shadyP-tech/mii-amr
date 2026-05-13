import { C, base, callout, source } from "./shared.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Pitch spine");

  ctx.addText(slide, {
    text: "The assignment became a measurement problem before it became a modeling problem.",
    left: 54,
    top: 104,
    width: 860,
    height: 156,
    fontSize: 50,
    color: C.ink,
    bold: true,
    typeface: ctx.fonts.title,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  ctx.addText(slide, {
    text: "5-10 minute pitch: the concrete problems, the decisions I made, and why those decisions were defensible.",
    left: 58,
    top: 292,
    width: 760,
    height: 54,
    fontSize: 24,
    color: C.muted,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  callout(
    slide,
    ctx,
    "Problem",
    "The task asks for real final pose, but the robot does not give a clean absolute endpoint alone.",
    76,
    394,
    330,
    118,
    C.red,
  );
  callout(
    slide,
    ctx,
    "Decision",
    "Build an external camera tracker and gate each run against a repeatable start.",
    474,
    394,
    330,
    118,
    C.teal,
  );
  callout(
    slide,
    ctx,
    "Consequence",
    "A simple probabilistic model works for primitives; the composed route exposes the limits.",
    872,
    394,
    330,
    118,
    C.amber,
  );

  source(slide, ctx, "Sources: docs/tasks/Aufgabe_02.pdf; scripts/; vision_tracker/; results/");
  return slide;
}
