import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { BrandTheme, defaultTheme } from "../theme";

export const LogoReveal: React.FC<{ logoUrl?: string; brandName: string; theme?: BrandTheme }> = ({
  logoUrl, brandName, theme = defaultTheme,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scale = spring({ frame, fps, config: { damping: 12, stiffness: 100 } });
  const opacity = interpolate(frame, [0, fps * 0.35], [0, 1], { extrapolateRight: "clamp" });

  return (
    <div
      style={{
        width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", opacity, transform: `scale(${scale})`, direction: theme.rtl ? "rtl" : "ltr",
      }}
    >
      {logoUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logoUrl} alt={brandName} style={{ maxWidth: 320, maxHeight: 320, objectFit: "contain" }} />
      ) : (
        <div
          style={{
            width: 220, height: 220, borderRadius: "50%", background: theme.accent,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontFamily: theme.fontFamily, fontWeight: 900, fontSize: 64, color: theme.primary,
          }}
        >
          {brandName.slice(0, 1).toUpperCase()}
        </div>
      )}
      <div style={{ fontFamily: theme.fontFamily, fontWeight: 700, fontSize: 30, color: theme.secondary, marginTop: 20 }}>
        {brandName}
      </div>
    </div>
  );
};
