export const ROOT = "/Users/stephpark/Documents/stephsWorld/mii-amr";

export const C = {
  paper: "#F7F3E8",
  paper2: "#EEE7D6",
  ink: "#17202A",
  muted: "#706B61",
  line: "#D7D3C8",
  teal: "#0F766E",
  blue: "#2563EB",
  amber: "#D97706",
  red: "#B91C1C",
  white: "#FFFFFF",
  black: "#000000",
};

export function base(slide, ctx, section = "Aufgabe 2") {
  ctx.addShape(slide, { left: 0, top: 0, width: ctx.W, height: ctx.H, fill: C.paper, line: ctx.line() });
  ctx.addShape(slide, {
    left: 54,
    top: 42,
    width: 10,
    height: 10,
    fill: C.teal,
    line: ctx.line(),
    name: "kicker-marker",
  });
  ctx.addText(slide, {
    text: section.toUpperCase(),
    left: 76,
    top: 34,
    width: 320,
    height: 26,
    fontSize: 15,
    color: C.teal,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
    name: "kicker-label",
  });
  ctx.addShape(slide, { left: 54, top: 66, width: 1172, height: 1.5, fill: C.line, line: ctx.line() });
  ctx.addText(slide, {
    text: String(ctx.slideNumber || ""),
    left: 1182,
    top: 654,
    width: 44,
    height: 26,
    fontSize: 16,
    color: C.muted,
    align: "right",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function title(slide, ctx, text, subtitle) {
  ctx.addText(slide, {
    text,
    left: 54,
    top: 88,
    width: 850,
    height: subtitle ? 124 : 132,
    fontSize: 44,
    color: C.ink,
    bold: true,
    typeface: ctx.fonts.title,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  if (subtitle) {
    ctx.addText(slide, {
      text: subtitle,
      left: 56,
      top: 216,
      width: 820,
      height: 58,
      fontSize: 21,
      color: C.muted,
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
  }
}

export function source(slide, ctx, text) {
  ctx.addText(slide, {
    text,
    left: 54,
    top: 654,
    width: 760,
    height: 26,
    fontSize: 13,
    color: C.muted,
    typeface: ctx.fonts.mono,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function pill(slide, ctx, text, left, top, color = C.teal, width = 170) {
  ctx.addShape(slide, { left, top, width, height: 28, fill: color, line: ctx.line() });
  ctx.addText(slide, {
    text,
    left: left + 12,
    top: top + 5,
    width: width - 24,
    height: 20,
    fontSize: 14,
    color: C.white,
    bold: true,
    align: "center",
    valign: "middle",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function callout(slide, ctx, label, body, left, top, width, height, accent = C.teal) {
  ctx.addShape(slide, { left, top, width: 6, height, fill: accent, line: ctx.line() });
  ctx.addText(slide, {
    text: label,
    left: left + 20,
    top: top + 2,
    width: width - 30,
    height: 30,
    fontSize: 18,
    color: accent,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addText(slide, {
    text: body,
    left: left + 20,
    top: top + 38,
    width: width - 30,
    height: height - 42,
    fontSize: 21,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function metric(slide, ctx, value, label, left, top, width, color = C.ink) {
  ctx.addText(slide, {
    text: value,
    left,
    top,
    width,
    height: 62,
    fontSize: 50,
    color,
    bold: true,
    typeface: ctx.fonts.title,
    align: "center",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  ctx.addText(slide, {
    text: label,
    left,
    top: top + 62,
    width,
    height: 56,
    fontSize: 17,
    color: C.muted,
    align: "center",
    insets: { left: 8, right: 8, top: 0, bottom: 0 },
  });
}

export function smallBox(slide, ctx, text, left, top, width, height, fill = C.white, stroke = C.line) {
  ctx.addShape(slide, {
    left,
    top,
    width,
    height,
    fill,
    line: { style: "solid", fill: stroke, width: 1.2 },
  });
  ctx.addText(slide, {
    text,
    left: left + 16,
    top: top + 12,
    width: width - 32,
    height: height - 18,
    fontSize: 19,
    color: C.ink,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function arrow(slide, ctx, x1, y, x2, color = C.line) {
  ctx.addShape(slide, { left: x1, top: y, width: x2 - x1 - 10, height: 2, fill: color, line: ctx.line() });
  ctx.addShape(slide, { left: x2 - 13, top: y - 5, width: 12, height: 12, fill: color, line: ctx.line() });
}
