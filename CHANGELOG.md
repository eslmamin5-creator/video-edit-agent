# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - Unreleased

### Added
- Initial implementation of `video-edit-agent` / `videoedit`: Arabic-first,
  model-agnostic AI video editing agent.
- Transcription provider router with no-API-key fallback to local
  Faster-Whisper (Gemini and ElevenLabs supported when configured).
- Editorial pipeline: silence trimming, repetition/take detection, false-start
  filtering, producing a re-renderable JSON Edit Decision List (EDL).
- Arabic-aware caption engine (RTL shaping, word-highlight karaoke styling,
  safe-zone margins) with ASS/SRT export.
- Brand Profiles system (folder-based, zero core-code changes to add a
  brand).
- Motion graphics router (HyperFrames/Remotion/Manim/simple fallback) with a
  Remotion component library.
- B-roll planner with provider router and fallback priority.
- Project memory (`project.md`) and multi-layer QA with auto-repair.
- CLI (`videoedit edit|doctor|providers|setup|config|brand|project`) with
  localized (Arabic/English) output.
- Test suite covering schemas, transcription routing/fallback, Arabic
  language/dialect detection, the dialect-preservation guard, EDL validation,
  captions, brand profiles, the editorial planner, an end-to-end render
  smoke test, and CLI smoke tests.
- Project documentation: `README.md` / `README.ar.md`,
  `THIRD_PARTY_NOTICES.md`, `CONTRIBUTING.md`, `SECURITY.md`, agent
  instruction files (`agent/core_instructions.md` and adapters).
