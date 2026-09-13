import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const LowerThird: React.FC<{ name: string; subtitle?: string; theme?: BrandTheme }> = ({
  name, subtitle = "", theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const x = interpolate(frame, [0, fps * 0.4], [-80, 0], { extrapolateRight: "clamp" });
  const opacity = interpolate(frame, [0, fps * 0.3], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        position: "absolute", bottom: 120, left: theme.rtl ? undefined : 60, right: theme.rtl ? 60 : undefined,
        transform: `translateX(${theme.rtl ? -x : x}px)`, opacity,
        display: "flex", flexDirection: "column", direction: theme.rtl ? "rtl" : "ltr",
        background: theme.primary, borderLeft: theme.rtl ? "none" : `6px solid ${theme.accent}`,
        borderRight: theme.rtl ? `6px solid ${theme.accent}` : "none",
        padding: "14px 24px", borderRadius: 4, maxWidth: 520,
      }}
    >
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 700, fontSize: 34, color: theme.secondary }}>
        {name}
      </div>
      {subtitle ? (
        <div style={{ fontFamily: theme.fontFamily, fontWeight: 400, fontSize: 22, color: theme.accent, marginTop: 2 }}>
          {subtitle}
        </div>
      ) : null}
    </div>
  );
};
