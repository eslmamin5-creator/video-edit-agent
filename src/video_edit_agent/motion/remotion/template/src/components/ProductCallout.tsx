import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const ProductCallout: React.FC<{
  label: string; x: number; y: number; theme?: BrandTheme;
}> = ({ label, x, y, theme = defaultTheme }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const opacity = interpolate(frame, [0, fps * 0.2], [0, 1], { extrapolateRight: "clamp" });
  const lineLength = interpolate(frame, [fps * 0.1, fps * 0.4], [0, 60], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <div style={{ position: "absolute", left: x, top: y, opacity, direction: theme.rtl ? "rtl" : "ltr" }}>
      <div style={{ width: 12, height: 12, borderRadius: "50%", background: theme.accent }} />
      <svg width="80" height="4" style={{ position: "absolute", top: 4, left: 12, overflow: "visible" }}>
        <line x1="0" y1="0" x2={lineLength} y2="0" stroke={theme.accent} strokeWidth={2} />
      </svg>
      <div
        style={{
          position: "absolute", top: -14, left: 12 + lineLength + 8, whiteSpace: "nowrap",
          background: theme.primary, color: theme.secondary, fontFamily: theme.fontFamily,
          fontWeight: 600, fontSize: 18, padding: "6px 14px", borderRadius: 6,
        }}
      >
        {label}
      </div>
    </div>
  );
};
