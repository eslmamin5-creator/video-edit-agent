import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const MetricHighlight: React.FC<{ metric: string; value: string; trend?: "up" | "down" | "flat"; theme?: BrandTheme }> = ({
  metric, value, trend = "flat", theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const opacity = interpolate(frame, [0, fps * 0.25], [0, 1], { extrapolateRight: "clamp" });
  const y = interpolate(frame, [0, fps * 0.3], [30, 0], { extrapolateRight: "clamp" });
  const trendGlyph = trend === "up" ? "▲" : trend === "down" ? "▼" : "▬";
  const trendColor = trend === "up" ? "#4ADE80" : trend === "down" ? "#F87171" : theme.secondary;

  return (
    <div
      style={{
        display: "flex", flexDirection: "column", alignItems: theme.rtl ? "flex-end" : "flex-start",
        opacity, transform: `translateY(${y}px)`, direction: theme.rtl ? "rtl" : "ltr",
        background: theme.primary, padding: "20px 28px", borderRadius: 10,
      }}
    >
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 500, fontSize: 20, color: theme.secondary, opacity: 0.75 }}>
        {metric}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <div style={{ fontFamily: theme.fontFamily, fontWeight: 900, fontSize: 56, color: theme.accent }}>{value}</div>
        <div style={{ fontSize: 22, color: trendColor }}>{trendGlyph}</div>
      </div>
    </div>
  );
};
