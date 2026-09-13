import React from "react";
import { Composition } from "remotion";
import { HookTitle } from "./components/HookTitle";
import { LowerThird } from "./components/LowerThird";
import { StatCounter } from "./components/StatCounter";
import { Quote } from "./components/Quote";
import { Comparison } from "./components/Comparison";
import { FeatureCard } from "./components/FeatureCard";
import { CTA } from "./components/CTA";
import { LogoReveal } from "./components/LogoReveal";
import { MetricHighlight } from "./components/MetricHighlight";
import { ProductCallout } from "./components/ProductCallout";
import { TimelineGraphic } from "./components/TimelineGraphic";
import { defaultTheme } from "./theme";

// Maps AnimationKind values (core/schemas.py) to Remotion compositions.
// The Python-side adapter (motion/remotion/adapter.py) selects the
// composition id and passes matching --props at render time.
const FPS = 30;
const DEFAULT_DURATION = FPS * 5;
const DEFAULT_SIZE = { width: 1080, height: 1920 };

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="hook_title"
        component={HookTitle}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ text: "Hook Title", theme: defaultTheme }}
      />
      <Composition
        id="lower_third"
        component={LowerThird}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ name: "Name", subtitle: "Title", theme: defaultTheme }}
      />
      <Composition
        id="stat_counter"
        component={StatCounter}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ value: 100, label: "Label", suffix: "%", theme: defaultTheme }}
      />
      <Composition
        id="quote"
        component={Quote}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ text: "Quote text", attribution: "", theme: defaultTheme }}
      />
      <Composition
        id="comparison"
        component={Comparison}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{
          left: { title: "Before", items: ["Item 1"] },
          right: { title: "After", items: ["Item 1"] },
          theme: defaultTheme,
        }}
      />
      <Composition
        id="feature_card"
        component={FeatureCard}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ title: "Feature", description: "Description", theme: defaultTheme }}
      />
      <Composition
        id="cta"
        component={CTA}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ text: "Call to action", actionLabel: "Learn more", theme: defaultTheme }}
      />
      <Composition
        id="logo_reveal"
        component={LogoReveal}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ brandName: "Brand", theme: defaultTheme }}
      />
      <Composition
        id="metric_highlight"
        component={MetricHighlight}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ metric: "Metric", value: "42", trend: "up" as const, theme: defaultTheme }}
      />
      <Composition
        id="product_callout"
        component={ProductCallout}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ label: "Callout", x: 200, y: 300, theme: defaultTheme }}
      />
      <Composition
        id="timeline_graphic"
        component={TimelineGraphic}
        durationInFrames={DEFAULT_DURATION}
        fps={FPS}
        {...DEFAULT_SIZE}
        defaultProps={{ milestones: [{ label: "Step 1", date: "Q1" }], theme: defaultTheme }}
      />
    </>
  );
};
