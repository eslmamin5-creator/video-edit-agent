// Brand tokens are injected at render time via --props; these are neutral
// fallbacks only (spec section 19: "do not hardcode client branding").
export type BrandTheme = {
  primary: string;
  secondary: string;
  accent: string;
  fontFamily: string;
  rtl: boolean;
};

export const defaultTheme: BrandTheme = {
  primary: "#111111",
  secondary: "#FFFFFF",
  // Neutral: the Brand Profile supplies the real accent. Never a strong
  // arbitrary color here.
  accent: "#FFFFFF",
  fontFamily: "Arial, 'Segoe UI', sans-serif",
  rtl: false,
};
