import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

type Milestone = { label: string; date: string };

export const TimelineGraphic: React.FC<{ milestones: Milestone[]; theme?: BrandTheme }> = ({
  milestones, theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = interpolate(frame, [fps * 0.2, fps * 1.2], [0, 100], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });
  const opacity = interpolate(frame, [0, fps * 0.2], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "center",
        padding: "0 80px", opacity, direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      <div style={{ position: "relative", height: 4, background: theme.secondary, opacity: 0.3, marginBottom: 30 }}>
        <div
          style={{
            position: "absolute", top: 0, [theme.rtl ? "right" : "left"]: 0, height: 4,
            width: `${progress}%`, background: theme.accent,
          }}
        />
      </div>
      <div style={{ display: "flex", flexDirection: theme.rtl ? "row-reverse" : "row", justifyContent: "space-between" }}>
        {milestones.map((m, i) => (
          <div key={i} style={{ textAlign: "center", flex: 1 }}>
            <div style={{ fontFamily: theme.fontFamily, fontWeight: 700, fontSize: 20, color: theme.secondary }}>
              {m.label}
            </div>
            <div style={{ fontFamily: theme.fontFamily, fontWeight: 400, fontSize: 16, color: theme.accent, marginTop: 4 }}>
              {m.date}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
