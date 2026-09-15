"""Installation/first-run bootstrap layer (v0.2.1 installation UX).

This package is the single place that knows how to take a fresh clone (or a
fresh Claude Code Skill install) to a working `videoedit` environment: OS
detection, Python interpreter selection, a private per-project runtime,
dependency installation by profile, and ffmpeg/Node capability checks --
without ever touching global Python, storing secrets, or silently installing
OS-level software.

`videoedit setup` (see `cli/setup.py`) is a thin CLI wrapper around
`bootstrap.setup.run_setup`; nothing here is Typer/Rich-specific so it can be
unit tested directly and reused by a Claude Code Skill driver script.
"""
