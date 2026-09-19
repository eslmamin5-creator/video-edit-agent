# Adapter: Claude

Read [`../core_instructions.md`](../core_instructions.md) first — everything
below is additive, not a replacement.

- Root `CLAUDE.md` in this repository re-exports the core instructions; that
  is what Claude Code loads automatically at session start.
- When using tools that shell out (`Bash`), prefer the project's own
  `pytest`/`ruff` invocations shown in the core instructions over ad-hoc
  commands.
- If asked to extend a provider or motion engine, follow the existing
  ABC/adapter pattern in that module rather than introducing a new
  abstraction style.
- Transcript review: use `videoedit review-chat` (see "Transcript review is
  chat-first" in the core instructions). Paste the command's output into your
  reply, pass the user's next message to it unchanged, and when it prints
  `AUDIO: <path>` deliver that clip with `SendUserFile` rather than pointing
  the user at `edit/review/audio/`. Ask one thing per turn.
