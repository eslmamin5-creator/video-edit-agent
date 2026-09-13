# Agent instructions — video-edit-agent

Canonical instructions for any AI coding agent (Claude, Codex, Copilot, or a
generic LLM agent) working in this repository. Tool-specific adapter files in
`agent/adapters/` add only what differs for that tool; everything else lives
here so it never has to be duplicated or drift out of sync.

## What this project is

`video-edit-agent` (CLI: `videoedit`) is an Arabic-first, model-agnostic AI
video editing agent. It works fully offline with zero API keys (falling back
to local Faster-Whisper transcription) and is enhanced, but never gated, by
optional cloud providers (Gemini, ElevenLabs). See the root `README.md` /
`README.ar.md` for user-facing docs and the original build spec at
`D:\Projects\Master Build Prompt — General AI Video Editing Agent.md` for the
full design rationale.

## Non-negotiable rule: dialect preservation

The single most important constraint in this codebase: **never translate,
rewrite, formalize, or localize spoken Arabic into a different dialect.**
Transcription is verbatim; the source audio has final authority. Any code
that touches transcript text before it reaches captions/EDL must go through
or respect `src/video_edit_agent/language/dialect_guard.py`
(`check_no_dialect_substitution()` / `enforce()`). If you are unsure whether a
change could alter dialect content, run `tests/test_dialect_guard.py` and
treat any failure as a hard blocker, not a warning.

## Architecture map

- `core/` — config layering, media (ffmpeg subprocess wrapper), schemas
  (`Transcript`/`Segment`/`Word`, `EDL`/`EDLClip`), pipeline orchestration,
  capability detection.
- `transcription/` — provider ABC + router with no-key fallback chain
  (Gemini → ElevenLabs → local Faster-Whisper → openai-whisper → whisper.cpp).
- `editorial/` — silence/repetition/false-start detection → `build_edl()`.
- `captions/` — Arabic-aware ASS/SRT generation (RTL shaping, karaoke word
  highlight, safe-zone margins).
- `render/` — ffmpeg composition/export, always via argument arrays.
- `brand/` — Brand Profiles (folder + `brand.yaml`) loader/validator.
- `motion/`, `broll/`, `subject/`, `providers/`, `qa/`, `language/`,
  `localization/`, `cli/` — see their own module docstrings.

## Conventions to follow

- **Never build an ffmpeg command as a shell string.** Always pass an
  argument list through `core.media.run()`. This is a security requirement
  (command-injection prevention), not a style preference.
- **Graceful degradation everywhere.** Every optional dependency (a cloud
  provider, a motion engine, arabic-reshaper, mediapipe, ...) must raise its
  own `*Unavailable` exception on failure, and callers must catch it and fall
  back — never let an optional feature crash the pipeline.
- **Read before you write.** Before adding code that calls into an existing
  module, read that module's actual current source — signatures drift from
  what a spec or an old memory implies.
- **Secrets** come only from `GEMINI_API_KEY` / `ELEVENLABS_API_KEY`
  environment variables. Never print a key value in any CLI output, log line,
  or error message. `tests/test_cli.py` has regression tests for this —
  keep them passing.
- **No API keys required.** Any feature you add must have a working
  no-key/offline path, even if degraded.
- Keep the CLI localized: user-facing strings go through
  `localization/interface.py`'s `t(key, lang, **kwargs)`, not hardcoded
  English.

## Working in this repo

- Install: `pip install -e ".[dev]"` inside the project's `.venv`.
- Run tests: `pytest tests/ -v` (ffmpeg-dependent tests skip cleanly if
  ffmpeg/ffprobe aren't on PATH).
- Lint: `ruff check src tests`.
- Before finishing a change, run the CLI smoke test
  (`videoedit doctor`, `videoedit providers`) to confirm nothing is broken
  end-to-end, not just at the unit-test level.

## Commit and scope discipline

- Keep commits small and logical; don't bundle unrelated changes.
- Don't rename the project away from the neutral `video-edit-agent` /
  `videoedit` names, and never introduce a client/company name into the
  codebase — see `core/config.py` for the one place branding is configured.
