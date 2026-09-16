# SKILL.md

Skill/capability summary for agents or tools that discover repositories via a
`SKILL.md` convention. Full instructions: [`agent/core_instructions.md`](agent/core_instructions.md).

**What this repo builds:** `videoedit` — an Arabic-first, model-agnostic CLI
that turns a raw video + optional Brand Profile into an edited, captioned,
re-renderable result. No API keys required; cloud providers (Gemini,
ElevenLabs) enhance it when configured but are never load-bearing.

## Setting this up as a Claude Code Skill

If you (the agent) have been asked to "set up video-edit-agent and verify
it," or the user just installed this repo into their Claude Code skills
directory, run the bootstrap and stop there — do not start editing videos
until it reports ready:

1. `cd` into this repo.
2. Run `videoedit setup --profile full-local` if `videoedit` is already on
   PATH; otherwise run `python3 -m pip install -e .` (or `py -3.11 -m pip
   install -e .` on Windows) once to get the base CLI, then run
   `videoedit setup --profile full-local`.
3. This builds a private runtime at `.runtime/venv` (never the system
   Python), installs the offline profile, checks ffmpeg/ffprobe and
   Node/npm, and prints a plain readiness report ending in `Ready.` or
   `Not fully ready yet` with exact next steps (e.g. an OS-specific ffmpeg
   install command). It never installs OS-level software itself and never
   stores any API key.
4. Re-running `videoedit setup` is always safe (idempotent) — use it to
   verify readiness at any time. `videoedit setup --check` does the same
   check without changing anything. `videoedit setup --repair` force-rebuilds
   the private runtime from scratch, even if it currently appears healthy.
5. Once it reports ready, use the private runtime's own CLI going forward:
   `.runtime/venv/bin/videoedit` (`.runtime\venv\Scripts\videoedit.exe` on
   Windows) — or just `videoedit` if that path is on PATH.

Full bootstrap implementation: `src/video_edit_agent/bootstrap/`
(`detect.py`, `runtime.py`, `dependencies.py`, `ffmpeg.py`, `node.py`,
`capabilities.py`, `report.py`, `setup.py`). `videoedit doctor` after setup
shows the full capability matrix plus the last setup run's summary.

**Key entry points:**
- CLI: `src/video_edit_agent/cli/main.py` (`videoedit edit|doctor|providers|setup|config|brand|project`)
- Bootstrap/install UX: `src/video_edit_agent/bootstrap/`
- Pipeline orchestration: `src/video_edit_agent/core/pipeline.py`
- Tests: `pytest tests/ -v`

**Hard constraint:** dialect-preserving transcription only — see
`src/video_edit_agent/language/dialect_guard.py` and the "Non-negotiable
rule" section of `agent/core_instructions.md`.
`TRANSCRIBE, DON'T TRANSLATE / REWRITE / FORMALIZE / DIALECT-CONVERT`.

## Talking to the user

This is a conversational agent, not a CLI the user has to memorize. Match
your interaction language (onboarding, status, questions, completion
reports) to whatever language the user is typing in — Arabic in, Arabic out;
English in, English out. This is independent of the transcript/media
language, which is never adjusted and never dialect-converted. See
"Interaction language: follow the user, not the transcript" and
"Natural-language workflow routing" in `agent/core_instructions.md` for the
full rules and the Editor/Creator/Assembler routing table. If a user's first
message is a capability question ("إنت بتعمل إيه؟" / "What can you do?"),
give the short three-workflow explanation from `README.md` before doing
anything else.
