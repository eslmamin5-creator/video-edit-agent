# AGENTS.md

This file is the entry point that Codex-style and other generic coding
agents look for in this repository. The canonical, tool-agnostic
instructions live in [`agent/core_instructions.md`](agent/core_instructions.md)
— read that file in full before making changes. See
[`agent/adapters/codex.md`](agent/adapters/codex.md) for adapter-specific
notes.

In short: this is an Arabic-first, offline-capable AI video editing agent.
The one rule that overrides everything else is **never translate or
formalize spoken Arabic dialect** — see the "Non-negotiable rule" section in
`agent/core_instructions.md`.
