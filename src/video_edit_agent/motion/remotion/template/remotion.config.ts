import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("png");
Config.setOverwriteOutput(true);
// Transparent output so compositions can be overlaid on the base timeline
// (spec section 16's layering order). VP8/WebM is the cross-platform choice;
// ProRes 4444 is used automatically instead when rendering on macOS.
Config.setCodec("vp8");
Config.setPixelFormat("yuva420p");
