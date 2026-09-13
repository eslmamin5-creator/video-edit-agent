import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

type Side = { title: string; items: string[] };

export const Comparison: React.FC<{ left: Side; right: Side; theme?: BrandTheme }> = ({
  left, right, theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const opacity = interpolate(frame, [0, fps * 0.3], [0, 1], { extrapolateRight: "clamp" });

  const Column: React.FC<{ side: Side; accent: string }> = ({ side, accent }) => (
    <div style={{ flex: 1, padding: 32 }}>
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 800, fontSize: 34, color: accent, marginBottom: 16 }}>
        {side.title}
      </div>
      {side.items.map((item, i) => (
        <div key={i} style={{ fontFamily: theme.fontFamily, fontSize: 22, color: theme.secondary, marginBottom: 10 }}>
          {theme.rtl ? `${item} •` : `• ${item}`}
        </div>
      ))}
    </div>
  );

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", flexDirection: theme.rtl ? "row-reverse" : "row",
        opacity, direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      <Column side={left} accent={theme.accent} />
      <div style={{ width: 2, background: theme.secondary, opacity: 0.3 }} />
      <Column side={right} accent={theme.secondary} />
    </div>
  );
};
