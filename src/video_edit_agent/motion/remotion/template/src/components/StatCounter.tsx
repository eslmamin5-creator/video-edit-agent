import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const StatCounter: React.FC<{ value: number; label: string; suffix?: string; theme?: BrandTheme }> = ({
  value, label, suffix = "", theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const countEnd = Math.max(durationInFrames - fps * 0.3, fps);
  const current = Math.round(interpolate(frame, [0, countEnd], [0, value], { extrapolateRight: "clamp" }));
  const opacity = interpolate(frame, [0, fps * 0.2], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", opacity, direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 900, fontSize: 140, color: theme.accent }}>
        {current.toLocaleString()}
        {suffix}
      </div>
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 500, fontSize: 32, color: theme.secondary, marginTop: 8 }}>
        {label}
      </div>
    </div>
  );
};
