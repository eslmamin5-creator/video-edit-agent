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

`TRANSCRIBE, DON'T TRANSLATE / REWRITE / FORMALIZE / DIALECT-CONVERT`

Egyptian, Gulf, Saudi, MSA, and code-switched Arabic/English speech must all
be preserved exactly as spoken. This applies regardless of what language the
user is typing to the agent in — see "Interaction language: follow the user,
not the transcript" below for that distinction. Transcription is verbatim;
the source audio has final authority. Any code
that touches transcript text before it reaches captions/EDL must go through
or respect `src/video_edit_agent/language/dialect_guard.py`
(`check_no_dialect_substitution()` / `enforce()`). If you are unsure whether a
change could alter dialect content, run `tests/test_dialect_guard.py` and
treat any failure as a hard blocker, not a warning.

## Interaction language: follow the user, not the transcript

**Interaction language is independent from media/transcript language.** These
are two separate things and must never be conflated:

- **Interaction language** — the language you (the agent) use to talk to the
  user: onboarding, status updates, questions, explanations, error messages,
  completion reports. Follow whatever language the user is currently writing
  in. If they write Arabic, respond in Arabic; if they switch to English,
  switch with them. Arabic responses should be clear, natural, everyday
  Arabic — not stiff, overly formal MSA translation-ese.
- **Media/transcript language and dialect** — whatever was actually spoken in
  the source video. This is governed entirely by the non-negotiable rule
  below and is never adjusted based on which language the user happens to be
  typing in. A user typing in Arabic about an English-language source video
  does not mean the transcript becomes Arabic, and a user typing in English
  about an Egyptian-dialect video does not mean the transcript gets
  translated or formalized.

When a user's very first message is a capability question — variants of
"إنت بتعمل إيه؟" / "ممكن تعمل إيه؟" / "What can you do?" / "How do I use
this?" — give the short three-workflow onboarding explanation (Editor /
Creator / Assembler; see `README.md` "What can video-edit-agent do?" section
for the canonical wording) in the user's language, then end with an
invitation to upload a video, send a script, or point at a scenes folder.
Don't front-load CLI flags or package internals in that first answer.

## Natural-language workflow routing

There is no NLU/intent-classification code in this repository — routing a
user's plain-language request to `videoedit edit` / `videoedit create` /
`videoedit assemble` is something you (the agent) do by reading their
request, not something to build as a parser. Use judgment, not exact string
matching, and don't require the user to know CLI syntax:

| User says (Arabic) | User says (English) | Route to |
|---|---|---|
| `عدل الفيديو ده` / `نضف الفيديو ده وحط كابشن` | "Edit this video" / "clean this up and add captions" | **Editor** (`videoedit edit`) — one existing raw take |
| `اعمل فيديو من السكربت ده` / `حوّل السكربت ده لفيديو` | "Create a video from this script" | **Creator** (`videoedit create`) — a script or idea, no footage yet |
| `اجمع المشاهد دي` / `اعمل rough cut للمشاهد دي` | "Assemble these scenes into one film" | **Assembler** (`videoedit assemble`) — a folder of multiple pre-shot clips |

If intent is genuinely ambiguous (e.g. the user has both a script and
existing footage and doesn't say which they want), ask one concise
clarifying question rather than guessing. Don't expose internal CLI syntax
in the answer unless it's actually useful to the user (e.g. they asked a
technical question, or you're reporting the exact command you ran).

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
