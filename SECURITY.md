# Security Policy

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities.
Instead, report privately to the maintainers (see the repository's contact
details on GitHub, e.g. a private security advisory via
"Security" → "Report a vulnerability" on the repo page).

Include:
- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof-of-concept if available.
- The version/commit you tested against.

We aim to acknowledge reports within a few days.

## Scope and design notes relevant to security review

- `video-edit-agent` never persists API keys — they are read only from the
  `GEMINI_API_KEY` / `ELEVENLABS_API_KEY` environment variables and are never
  written to disk, logs, or CLI output. `tests/test_cli.py` contains
  regression tests asserting no command prints a key value even when one is
  set.
- All FFmpeg invocations use argument-array subprocess calls
  (`core/media.py`'s `run()`), never shell strings, to eliminate command
  injection from filenames or user-supplied text.
- The project works fully offline; no telemetry or network calls are made
  unless a cloud provider (Gemini/ElevenLabs) is explicitly configured and
  used.

## Supported versions

This project is pre-1.0; only the latest released version receives fixes.
