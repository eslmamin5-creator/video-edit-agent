# Adapter: Codex / generic OpenAI-style coding agents

Read [`../core_instructions.md`](../core_instructions.md) first — everything
below is additive, not a replacement.

- Root `AGENTS.md` in this repository re-exports the core instructions; that
  is the file this class of agent conventionally looks for.
- This project has no network access requirement for its test suite — all
  cloud-provider tests run with keys deliberately unset (see
  `tests/test_transcription_router.py`), so a sandboxed/offline agent can run
  the full suite without special network allowances.
- Apply patches as minimal diffs; do not reformat files you are not
  otherwise changing (see `[tool.ruff]` in `pyproject.toml` for the house
  style if you need to check).
