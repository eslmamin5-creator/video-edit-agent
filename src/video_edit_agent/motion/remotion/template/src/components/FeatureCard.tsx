import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const FeatureCard: React.FC<{ title: string; description: string; theme?: BrandTheme }> = ({
  title, description, theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scale = spring({ frame, fps, config: { damping: 16, stiffness: 120 } });
  const opacity = interpolate(frame, [0, fps * 0.25], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center",
        direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      <div
        style={{
          transform: `scale(${scale})`, opacity, background: theme.primary, borderRadius: 16,
          padding: "40px 48px", maxWidth: 640, borderTop: `4px solid ${theme.accent}`,
        }}
      >
        <div style={{ fontFamily: theme.fontFamily, fontWeight: 800, fontSize: 40, color: theme.secondary, marginBottom: 12 }}>
          {title}
        </div>
        <div style={{ fontFamily: theme.fontFamily, fontWeight: 400, fontSize: 24, color: theme.secondary, opacity: 0.85 }}>
          {description}
        </div>
      </div>
    </div>
  );
};
