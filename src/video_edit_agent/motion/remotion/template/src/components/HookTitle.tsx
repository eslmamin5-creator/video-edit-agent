import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const HookTitle: React.FC<{ text: string; theme?: BrandTheme }> = ({ text, theme = defaultTheme }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scale = spring({ frame, fps, config: { damping: 14, stiffness: 140 } });
  const opacity = interpolate(frame, [0, fps * 0.3], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center",
        direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      <div
        style={{
          transform: `scale(${scale})`, opacity, fontFamily: theme.fontFamily, fontWeight: 800,
          fontSize: 88, color: theme.secondary, textAlign: "center", padding: "0 60px",
          textShadow: "0 4px 24px rgba(0,0,0,0.5)",
        }}
      >
        {text}
      </div>
    </div>
  );
};
