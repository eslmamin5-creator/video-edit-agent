import React, { useEffect, useState } from "react";
import { continueRender, delayRender, staticFile, useCurrentFrame, useVideoConfig, interpolate, Easing } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

// A short phrase set BEHIND the on-camera subject: this layer is drawn over the
// full frame and the compositor then draws the subject cutout on top of it, so
// the speaker overlaps part of the letters. Everything editorial is decided on
// the Python side (motion/behind_subject.py) and arrives as props: the words
// (the approved edit-plan text, never generated here), the line split, size,
// the position of the phrase as ONE locked group, colour, outline, opacity, and
// the timing (the composition IS the show window: `enterSec` ramps the phrase in,
// `exitSec` ramps it out at the end).
//
// `probe` renders the same layout with every word in its own flat colour, at full
// opacity and without decoration: the planner reads the glyph shapes back from it
// to judge occlusion, so the measurement comes from the very renderer that draws
// the final frames.
export type TextOutline = { color: string; width: number };

const PROBE_COLORS = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"];

const useBrandFonts = (theme: BrandTheme) => {
  const files = theme.fontFiles ?? [];
  const [handle] = useState(() => (files.length > 0 ? delayRender("brand fonts") : null));
  useEffect(() => {
    if (handle === null) return;
    Promise.all(
      files.map((f) =>
        new FontFace(f.family, `url(${staticFile(f.file)})`, { weight: String(f.weight) })
          .load()
          .then((face) => document.fonts.add(face)),
      ),
    )
      .catch(() => undefined)
      .finally(() => continueRender(handle));
  }, [handle]);
};

export const BehindText: React.FC<{
  text: string;
  lines?: string[];
  theme?: BrandTheme;
  color?: string;
  fontPx?: number;
  centerX?: number;
  centerY?: number;
  outline?: TextOutline | null;
  shadow?: string;
  opacity?: number;
  enterSec?: number;
  exitSec?: number;
  probe?: boolean;
}> = ({
  text, lines, theme = defaultTheme, color, fontPx = 200, centerX, centerY, outline, shadow,
  opacity = 1, enterSec = 0.3, exitSec = 0.25, probe = false,
}) => {
  useBrandFonts(theme);
  const frame = useCurrentFrame();
  const { fps, width, height, durationInFrames } = useVideoConfig();
  const t = frame / fps;
  const total = durationInFrames / fps;
  const enter = interpolate(t, [0, Math.max(enterSec, 1e-3)], [0, 1], {
    extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  const leave = interpolate(t, [total - Math.max(exitSec, 1e-3), total], [1, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.in(Easing.quad),
  });
  const alpha = probe ? 1 : Math.min(enter, leave) * opacity;
  const scale = probe ? 1 : interpolate(enter, [0, 1], [0.94, 1]);
  const rise = probe ? 0 : interpolate(enter, [0, 1], [fontPx * 0.06, 0]);
  const shown = lines && lines.length > 0 ? lines : [text];
  const x = typeof centerX === "number" ? centerX : width / 2;
  const y = typeof centerY === "number" ? centerY : height * 0.4;
  let wordIndex = 0;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", direction: theme.rtl ? "rtl" : "ltr" }}>
      <div
        style={{
          position: "absolute", left: x, top: y,
          transform: `translate(-50%, -50%) translateY(${rise}px) scale(${scale})`, transformOrigin: "50% 50%",
          opacity: alpha,
          fontFamily: theme.fontFamily, fontWeight: 900, fontSize: fontPx, lineHeight: 1.05,
          color: color ?? theme.secondary, textAlign: "center", width: "max-content", maxWidth: "94%",
          WebkitTextStroke: !probe && outline ? `${outline.width}px ${outline.color}` : undefined,
          paintOrder: !probe && outline ? "stroke fill" : undefined,
          textShadow: probe ? "none" : (shadow ?? "none"),
        }}
      >
        {shown.map((line, i) => (
          <div key={i} style={{ whiteSpace: "nowrap" }}>
            {line.split(" ").filter(Boolean).map((word, j, all) => {
              const c = probe ? PROBE_COLORS[wordIndex++ % PROBE_COLORS.length] : undefined;
              return (
                <React.Fragment key={j}>
                  <span style={c ? { color: c } : undefined}>{word}</span>
                  {j < all.length - 1 ? " " : ""}
                </React.Fragment>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};
