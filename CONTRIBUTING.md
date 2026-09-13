# Contributing

Thanks for considering a contribution to `video-edit-agent`.

## Setup

```bash
git clone <your-fork-url>
cd video-edit-agent
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

FFmpeg must be on your `PATH` to run the render/EDL integration tests
(unit tests that don't need it will still run and pass without it).

## Running tests

```bash
pytest tests/ -v
ruff check src tests
```

All tests must pass with **zero API keys set** — the no-key/offline path is
a first-class requirement, not a fallback to tolerate. If you add a feature
that depends on an optional dependency or cloud provider, it must degrade
gracefully (raise a dedicated `*Unavailable` exception, caught by the
caller) rather than crash.

## Ground rules

- Read [`agent/core_instructions.md`](agent/core_instructions.md) before your
  first PR — it documents the architecture and the non-negotiable
  dialect-preservation rule.
- Never build ffmpeg commands as shell strings; always pass argument lists
  through `core.media.run()`.
- Never introduce a client/company brand name into the codebase. The project
  stays neutral (`video-edit-agent` / `videoedit`); branding is configured
  per-user via Brand Profiles, not hardcoded.
- Keep PRs focused — one logical change per PR, with tests for new
  behavior.
- User-facing CLI strings go through `localization/interface.py`, not
  hardcoded English (the project is Arabic-first).

## Reporting bugs / requesting features

Open an issue with steps to reproduce, your OS/Python version, and whether
you're running with or without API keys configured.

## Security issues

Do not open a public issue for a security vulnerability — see
[`SECURITY.md`](SECURITY.md).
