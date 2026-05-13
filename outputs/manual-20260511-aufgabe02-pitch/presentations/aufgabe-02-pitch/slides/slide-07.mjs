import { C, base, callout, source } from "./shared.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  base(slide, ctx, "Close");

  ctx.addText(slide, {
    text: "What I would defend in the pitch",
    left: 70,
    top: 112,
    width: 720,
    height: 70,
    fontSize: 52,
    color: C.ink,
    bold: true,
    typeface: ctx.fonts.title,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  callout(
    slide,
    ctx,
    "1. Measurement first",
    "The external camera tracker was not extra polish; it was required to answer the task with physical final positions.",
    94,
    246,
    1040,
    86,
    C.teal,
  );
  callout(
    slide,
    ctx,
    "2. Repeatability before statistics",
    "The start-pose gate made repeated runs comparable by rejecting stale, invalid, or badly placed starts.",
    94,
    374,
    1040,
    86,
    C.blue,
  );
  callout(
    slide,
    ctx,
    "3. The model failed in the useful way",
    "A rough endpoint model is enough to show uncertainty, but the supervisor route demonstrates why mobile robots need localization feedback such as SLAM.",
    94,
    502,
    1040,
    96,
    C.red,
  );

  source(slide, ctx, "Closing line: the work made the need for SLAM visible instead of just stating it.");
  return slide;
}
