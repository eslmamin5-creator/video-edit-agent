import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const Quote: React.FC<{ text: string; attribution?: string; theme?: BrandTheme }> = ({
  text, attribution = "", theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const opacity = interpolate(frame, [0, fps * 0.4], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", opacity, direction: theme.rtl ? "rtl" : "ltr", padding: "0 100px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontFamily: theme.fontFamily, fontWeight: 600, fontStyle: "italic", fontSize: 52,
          color: theme.secondary, lineHeight: 1.4,
        }}
      >
        {theme.rtl ? `”${text}“` : `“${text}”`}
      </div>
      {attribution ? (
        <div style={{ fontFamily: theme.fontFamily, fontWeight: 500, fontSize: 26, color: theme.accent, marginTop: 24 }}>
          {theme.rtl ? `— ${attribution}` : `— ${attribution}`}
        </div>
      ) : null}
    </div>
  );
};
