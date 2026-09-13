# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.1] - 2026-09-13

Hardening pass: verify Arabic, motion, and provider integrations before
Phase 2. No product redesign, no new agents/architecture — see
`V1.1 Hardening Prompt.md` for the full scope.

### Added
- Real, non-mocked Arabic acceptance test (`scripts/arabic_acceptance.py`,
  13/13 checks): RTL layout, shaping, karaoke highlighting, mixed Arabic/
  English, punctuation, brand-driven fonts, and an actual ffmpeg render
  verified via ffprobe and frame pixel-content checks.
- Automated dialect regression tests for Egyptian/Gulf/mixed-dialect
  sentences.
- Motion engine fallback traceability: `MotionPlanItem.fallback_log` records
  every engine tried and rejected before the one that actually rendered, so a
  substitution is never silent (`tests/test_motion_router.py`).
- Real HyperFrames acceptance script (`scripts/hyperframes_acceptance.py`)
  that checks for a genuine SDK (not just a same-named import) before
  attempting a render, and honestly reports `BLOCKED BY ENVIRONMENT` rather
  than fabricating success.
- Real Behind-Subject acceptance script and tests
  (`scripts/behind_subject_acceptance.py`, `tests/test_behind_subject.py`):
  genuine mediapipe segmentation against a synthetic (non-photographic)
  test fixture, plus a proven mask-cache reuse on a repeated identical call.
- Gated live-provider integration tests for Gemini/ElevenLabs
  (`tests/test_provider_live_acceptance.py`), auto-skipped when no API key
  is present so normal test runs never depend on cloud availability.
- `core/verification_store.py`: a small local record of genuine acceptance
  runs, so `videoedit doctor` can distinguish "installed" from "verified by
  a real render" instead of ever reporting a false positive.
- Honest capability-status matrix in `README.md`/`README.ar.md`.

### Fixed
- `subject/detect.py` crashed with `AttributeError` under mediapipe>=1.0,
  which removed the legacy `mediapipe.solutions` API this project uses.
  Detection and availability checks now correctly identify this and degrade
  gracefully to a plain overlay instead of crashing.
- `capability_router.detect_hyperframes()` previously reported "available"
  on a bare successful `import hyperframes`, which would falsely pass for
  the unrelated, same-named PyPI package that has no rendering API. It now
  checks for the actual `render_from_spec` entry point.

### Known limitations (see README for full detail)
- Behind-Subject *video compositing* (an element actually rendered visibly
  behind a subject in the final output) is not implemented:
  `render/composition.py` never reads `Overlay.behind_subject`. Only mask
  detection and caching are real and verified.
- Real Arabic speech/ASR acceptance against actual recorded audio remains
  unverified in this environment (no fixture/voice available); dialect-
  preservation logic itself is covered by automated regression tests.
- Gemini and ElevenLabs live API calls remain unverified in this environment
  (no API keys configured); adapters and gated tests are in place and will
  run automatically once a key is set.

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
