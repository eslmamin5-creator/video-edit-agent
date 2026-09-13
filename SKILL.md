# SKILL.md

Skill/capability summary for agents or tools that discover repositories via a
`SKILL.md` convention. Full instructions: [`agent/core_instructions.md`](agent/core_instructions.md).

**What this repo builds:** `videoedit` — an Arabic-first, model-agnostic CLI
that turns a raw video + optional Brand Profile into an edited, captioned,
re-renderable result. No API keys required; cloud providers (Gemini,
ElevenLabs) enhance it when configured but are never load-bearing.

**Key entry points:**
- CLI: `src/video_edit_agent/cli/main.py` (`videoedit edit|doctor|providers|setup|config|brand|project`)
- Pipeline orchestration: `src/video_edit_agent/core/pipeline.py`
- Tests: `pytest tests/ -v`

**Hard constraint:** dialect-preserving transcription only — see
`src/video_edit_agent/language/dialect_guard.py` and the "Non-negotiable
rule" section of `agent/core_instructions.md`.
