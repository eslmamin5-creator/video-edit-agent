import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const CTA: React.FC<{ text: string; actionLabel?: string; theme?: BrandTheme }> = ({
  text, actionLabel = "", theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const pulse = 1 + 0.04 * Math.sin(frame / fps * Math.PI * 2);
  const opacity = interpolate(frame, [0, fps * 0.3], [0, 1], { extrapolateRight: "clamp" });
  const enter = spring({ frame, fps, config: { damping: 15, stiffness: 130 } });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", opacity, direction: theme.rtl ? "rtl" : "ltr",
        transform: `scale(${enter})`,
      }}
    >
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 800, fontSize: 46, color: theme.secondary, marginBottom: 20 }}>
        {text}
      </div>
      {actionLabel ? (
        <div
          style={{
            transform: `scale(${pulse})`, background: theme.accent, color: theme.primary,
            fontFamily: theme.fontFamily, fontWeight: 700, fontSize: 28, padding: "16px 40px", borderRadius: 999,
          }}
        >
          {actionLabel}
        </div>
      ) : null}
    </div>
  );
};
