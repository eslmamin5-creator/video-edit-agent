# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.2] - 2026-09-15

Arabic-first UX, product onboarding, and user guide only -- Editor / Creator
/ Assembler, rendering, provider routing, and bootstrap/runtime architecture
are unchanged. Goal: a first-time user should understand what this agent
does, how Editor/Creator/Assembler differ, and how Arabic support works
within a minute, without reading source code.

### Added
- Agent instructions (`agent/core_instructions.md`, `SKILL.md`) now
  explicitly separate **interaction language** (follows the user, Arabic in
  / Arabic out, English in / English out) from **media/transcript
  language** (governed only by the dialect-preservation rule, never
  adjusted by interaction language), and document a natural-language
  routing table so users can say "عدل الفيديو ده" / "Edit this video"
  instead of memorizing `videoedit edit|create|assemble`.
- `docs/USER_GUIDE.md` and `docs/USER_GUIDE.ar.md` — user-facing (not
  developer/architecture) guides covering the three workflows, feature
  overview, Arabic-first experience, offline-vs-cloud table, install path,
  and first-conversation examples.
- `docs/video-edit-agent-user-guide.pdf` and `-ar.pdf`, generated
  reproducibly from the Markdown guides via `scripts/build_user_guide.py`
  (`pip install -e ".[docs]"`; PDF tooling is documentation-only, never a
  runtime dependency).
- `README.md` / `README.ar.md` restructured product-first: hero, language
  switch, what the agent does, Editor/Creator/Assembler, feature guide,
  Arabic-first capabilities, offline-vs-cloud, install, first-conversation
  examples, then technical CLI usage.
- Localization keys `onboarding_invitation` and `readiness_summary` in
  `localization/messages/{en,ar}.json` for the first-use invitation and a
  localized readiness summary (Arabic readiness responses no longer default
  to an English capability report).
- Tests for interaction-language selection on capability questions and for
  README/user-guide link integrity.

### Unchanged
No changes to Editor, Creator, Assembler, MasterTimeline, rendering logic,
provider routing, or bootstrap/runtime architecture in this release.

## [0.2.1] - 2026-09-15

Installation and first-run UX only -- Editor/Creator/Assembler, Brand
Profiles, MasterTimeline, and behind-subject compositing are unchanged.
Goal: installation should feel like "install the Skill -> ask Claude to set
it up -> start making videos," not "become a Python developer first."

### Added
- New `src/video_edit_agent/bootstrap/` package: OS detection, ordered
  Python interpreter selection (prefers 3.11, falls back through
  3.12/3.10, then anything meeting `requires-python`, always with an
  explanation), a private per-project runtime at `.runtime/venv` (never
  the global/system Python), named install profiles (`core`, `local`,
  `subject`, `motion`, `full-local`), ffmpeg/ffprobe detection with exact
  OS/package-manager-specific install guidance (winget/choco/scoop,
  Homebrew, apt/dnf/yum/pacman/zypper/apk) that never installs anything
  automatically, Node/npm/Remotion detection, and non-secret setup-state
  persistence (`.runtime/setup_state.json`).
- `videoedit setup --profile full-local|local|subject|motion|core`,
  `videoedit setup --check` (read-only, makes no changes), and
  `videoedit setup --repair` (force-rebuilds the private runtime from
  scratch, even if it currently appears healthy, without touching anything
  outside `.runtime/`). Re-running setup is
  idempotent and fast (pip no-ops on an already-satisfied profile).
- `videoedit doctor` now also reports the last `videoedit setup` run
  (profile, Python version, verification timestamp) when setup state
  exists.

### Fixed
- **mediapipe dependency bound was unsafe.** `mediapipe<1.0` alone is not
  sufficient: fresh-install testing found that mediapipe 0.10.35 (still
  `<1.0`) has already dropped the legacy `mediapipe.solutions.
  selfie_segmentation` API this project's behind-subject adapter needs.
  Pinned to the exact verified-working version, `mediapipe==0.10.21`,
  across the `subject`, `full-local`, and `all` extras.
- **`full-local` could fail to install on a clean machine.** It previously
  pulled in `manim`, whose wheel build requires system Cairo/Pango
  libraries (`pangocairo >= 1.30.0`) that are commonly absent on fresh
  Linux/Windows installs. `manim` is not load-bearing for "local motion"
  -- the built-in simple motion engine already covers it fully offline --
  so it was removed from `full-local` (still available via the explicit
  `motion`/`all` profiles for users who want it and have the system deps).
- **`videoedit setup`'s final readiness report reflected the wrong
  interpreter.** After building the private runtime and installing a
  profile into it, the capability check (faster-whisper, mediapipe, ...)
  ran in-process -- i.e. in whichever interpreter invoked `videoedit
  setup`, not the newly-built `.runtime/venv` -- so a fresh install could
  report just-installed packages as still missing. The check (and the
  `doctor` table shown at the end of `setup`) now runs inside the private
  runtime's own interpreter via subprocess, with an in-process fallback
  and an explicit note if that subprocess check itself fails.

### Changed
- `mediapipe>=0.10` bumped to the exact pin above in `subject`/`full-local`/
  `all` extras (see Fixed).
- Package/CLI version bumped to `0.2.1`.

## [0.2.0] - 2026-09-14

Phase 2 finalization: Creator and Assembler agents, real rendered video
transitions, and real loudness normalization, on top of the multi-agent
production engine added in Phase 2. No new agents beyond Editor/Creator/
Assembler, no product-scope expansion, no GUI/NLE work, no Veo/cloud
generation — see the Phase 2 Finalization spec for full scope.

### Added
- Real, rendered video transitions in the Assembler's finished film: hard
  cut, short crossfade, and dissolve are genuinely rendered via ffmpeg
  `xfade`/`acrossfade` (not just planned). A transition is only ever
  reported `applied=true` when the renderer actually produced that effect.
- Configurable, conservative transition durations (0.15s-0.5s band) via
  `core/transition_math.py::clamp_transition_duration()`, with timeline
  duration math that correctly accounts for overlapping crossfade regions
  (no A/V desync).
- Audio crossfade (`acrossfade`) accompanying a visual crossfade whenever
  both neighboring clips have real audio; falls back to a plain concat when
  only one side does, instead of forcing a crossfade onto silence.
- Real, conservative loudness normalization in the Assembler
  (`agents/assembler/loudness.py`) using ffmpeg `loudnorm`: only scenes that
  deviate from the target level are normalized; true digital silence is
  left untouched via a `SILENCE_FLOOR_DB` guard (avoids a `loudnorm` NaN
  failure mode on pure silence). Every decision records whether
  normalization was required/applied and at what target level.
- New acceptance/regression tests (`tests/test_phase2_transitions_loudness.py`):
  crossfade output-duration and blended-frame verification, dissolve/
  crossfade equivalence, audio-crossfade gating, loudness planning
  (including the silence-floor edge case), and a full Assembler finish
  acceptance with at least one real applied transition and no clipping.
- Real, offline, end-to-end acceptance runs for all three workflows: Editor
  (`edit --offline`), Creator (`create --offline`, exercised against the
  `acme_test` Brand Profile), and Assembler (`assemble --rough --offline`
  then `--finish --offline`).

### Fixed
- ffmpeg `xfade`/`acrossfade` timebase mismatch between a freshly-trimmed
  clip and an already-concatenated accumulator, which previously made
  crossfade rendering fail outright (`render/composition.py`, via explicit
  `settb=AVTB`/`asettb=AVTB` before every `xfade`/`acrossfade`).
- QA (`agents/assembler/qa.py::check_timeline_continuity`) previously
  flagged any negative timeline gap as an "impossible overlap" error, which
  broke once real crossfades started legitimately overlapping clip
  boundaries; now compares the observed overlap against the boundary
  clip's own declared `transition_duration_s`.

### Fixed (release-gate audit, 2026-09-15)
- Corrected stale documentation (`README.md`, `README.ar.md`, this file) that
  claimed Behind-Subject video compositing was unimplemented. It was already
  fully implemented and wired in `74335bb` (the Phase 2 foundational slice,
  an ancestor of this release): `core/pipeline.py` renders a real RGBA
  subject cutout and places it after the graphic overlay in `plan.overlays`,
  and `render/composition.py` composites overlays in list order via
  ffmpeg's alpha-aware `overlay` filter — no special-casing needed. Verified
  with 8/8 real (non-mocked, non-skipped) tests in `tests/test_behind_subject.py`.

### Known limitations
- J-Cut/L-Cut (audio-lead/audio-trail edits) remain conceptual only — the
  Transition Director never proposes one. Deferred intentionally; not a
  release blocker.
- HyperFrames motion engine remains unavailable/unverified in this
  environment (see 0.1.1 notes); the router's traceable fallback to a
  working engine is unaffected and remains verified.

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
