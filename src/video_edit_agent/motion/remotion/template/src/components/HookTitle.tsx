import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

// Legibility treatment computed by the Python side (motion/legibility.py) from
// the Brand Profile and the footage: foreground colour, an optional backing
// plate, outline/shadow, and a subject-aware vertical position. Every field is
// optional; without them the title falls back to the centred brand look.
export type TitlePlate = { color: string; opacity: number; padding: number; radius: number };
export type TitleOutline = { color: string; width: number };

const hexToRgba = (hex: string, alpha: number): string => {
  const c = hex.replace("#", "");
  const full = c.length === 3 ? c.split("").map((ch) => ch + ch).join("") : c;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
};

export const HookTitle: React.FC<{
  text: string;
  theme?: BrandTheme;
  color?: string;
  centerY?: number;
  maxWidth?: number;
  plate?: TitlePlate | null;
  outline?: TitleOutline | null;
  shadow?: string;
}> = ({ text, theme = defaultTheme, color, centerY, maxWidth, plate, outline, shadow }) => {
  const frame = useCurrentFrame();
  const { fps, height } = useVideoConfig();
  const scale = spring({ frame, fps, config: { damping: 14, stiffness: 140 } });
  const opacity = interpolate(frame, [0, fps * 0.3], [0, 1], { extrapolateRight: "clamp" });
  const positioned = typeof centerY === "number";

  return (
    <div
      style={{
        position: "relative", width: "100%", height: "100%",
        display: positioned ? "block" : "flex", alignItems: "center", justifyContent: "center",
        direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      <div
        style={{
          ...(positioned
            ? { position: "absolute", left: "50%", top: Math.min(Math.max(centerY as number, 0), height), transformOrigin: "50% 50%" }
            : {}),
          transform: positioned ? `translate(-50%, -50%) scale(${scale})` : `scale(${scale})`,
          opacity, fontFamily: theme.fontFamily, fontWeight: 800, fontSize: 88, lineHeight: 1.25,
          color: color ?? theme.secondary, textAlign: "center",
          width: positioned && maxWidth ? "max-content" : undefined,
          maxWidth: maxWidth ?? undefined,
          padding: plate ? `${plate.padding}px` : "0 60px",
          background: plate ? hexToRgba(plate.color, plate.opacity) : undefined,
          borderRadius: plate ? plate.radius : undefined,
          WebkitTextStroke: outline ? `${outline.width}px ${outline.color}` : undefined,
          paintOrder: outline ? "stroke fill" : undefined,
          textShadow: shadow ?? "0 4px 24px rgba(0,0,0,0.5)",
          boxSizing: "border-box",
        }}
      >
        {text}
      </div>
    </div>
  );
};
