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

## Transcript review is chat-first

After transcription the user reviews the transcript **in the conversation**,
never by opening WAV/JSON/ASS files. Flow: transcribe -> show the transcript
in chat -> the user approves or corrects -> corrections are saved -> continue
to edit-plan review.

- Run `videoedit review-chat <edit dir> --open` once to start the review
  (low-confidence segments become `unresolved`), then pass **the user's
  message as typed** as the second argument for every later turn:
  `videoedit review-chat <edit dir> "<message>"`. Show the printed text in the
  chat as-is; there is no command syntax for the user to learn.
- The user sees 1-based numbers (`[09] 00:32.82-00:34.74`), the current text,
  a `Status:` line and the `Low-confidence:` words. Never show internal ids
  such as `s8`. The default view is suspicious-first; the user can ask for
  all segments, only suspicious ones, a range, specific numbers, next/previous.
  Long videos are paged (8 per page), never dumped.
- Understood in Arabic and English: `1 صح` / `approve 1`, `الجملة 5: <text>`,
  `الجملة 14 كلها المفروض تكون: <text>`, `في الجملة 9 غير كلمة X إلى Y`,
  `غير service إلى سيرفس`, `اسمعني الجملة 15`, `وريني بس الجمل اللي فيها شك`,
  `راجعلي 9 و14 و15 بس`, `اعتمد الباقي`.
- Text the user dictates is stored **verbatim** (never translated, never
  formalized). A word-level fix that is ambiguous (the word appears twice, or
  nowhere) is not guessed: the reply asks which one.
- Segment status: `approved`, `needs_review` (low-confidence, advisory),
  `corrected_pending_approval`, `unresolved`. A whole-sentence replacement or
  "N صح" approves that segment; a word/phrase fix leaves it
  `corrected_pending_approval` until the user says it is right.
  `unresolved` and `corrected_pending_approval` block `review-approve`.
- Audio is a fallback, cut **only** when the user asks (`اسمعني الجملة 9`).
  The reply prints `AUDIO: <path>`; send that file to the user in the
  conversation (e.g. `SendUserFile`) instead of telling them where it is.
- Corrections live in `review/transcript_corrections.json` and review state in
  `review/review_state.json`; the raw `transcript_unified.json` is never
  modified. Project-specific corrections belong only in that state.
- **Transcript approval is not hook-copy approval.** Approving a segment does
  not make it the on-screen hook: hook copy stays `pending_review` until the
  user approves the exact words (`review-copy`). A rewrite you suggest is
  recorded as a *proposed rewrite* (`proposed_copy`) and is never rendered
  until approved. Raw ASR is never used as hook copy.

## Edit-plan / B-roll review is chat-first too

Order: EDIT PLAN -> show the treatments in chat -> the user approves or
changes -> decisions saved -> only then source/generate assets -> quality gate
-> final composition. The user must never meet a B-roll or motion decision for
the first time in a finished render. Generated B-roll is one treatment among
several (stay on speaker, punch-in, kinetic typography, behind-subject text,
motion graphic, local B-roll, generated B-roll, no treatment), never the default.

- Decisions come from `review/broll_editorial.json`; `videoedit review-plan
  <edit dir> --open [--settled TREATMENT=reason]` builds `review/edit_plan.json`
  (slots already approved elsewhere, e.g. hook or punch-in, are settled and not
  asked). Then pass the user's message as typed: `videoedit review-plan
  <edit dir> "<message>"` and show the printed text as-is (no JSON/EDL).
- Understood: `1 موافق`, `2 خليه speaker`, `3 بدل generated اعمله motion
  graphic`, `4 استخدم B-roll محلي`, `5 اختار الخيار B` / `5 النص: <text>`,
  `3 ولّد`, `2 بلاش`, `اعتمد الباقي`, `وريني بس الحاجات اللي محتاجة asset`,
  `وريني الـgenerated فقط`, `وريني الكل`.
- Per-slot status: `pending_review`, `approved`, `changed`, `rejected`,
  `asset_required`, `generation_approved`. Approving a treatment **never**
  approves AI generation: that needs its own explicit answer for that slot, and
  until then the slot's fallback is what composition uses. Missing local
  footage never blocks (fallback). On-screen text (behind-subject/kinetic) comes
  only from the approved transcript and is approved verbatim before the slot is.
- `review-approve` is refused while any edit-plan slot is `pending_review`.
  Nothing in this review generates assets, renders a cutout or sets
  `ready_for_final_render`.

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
