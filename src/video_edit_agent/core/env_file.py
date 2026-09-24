"""Project-local `.env` support (v0.2.3, spec section 26 addendum).

Lets a normal user configure GEMINI_API_KEY / ELEVENLABS_API_KEY by dropping
a `.env` file next to their project instead of learning OS environment
variables. Kept separate from `core.config` so the file-discovery and
merge/write logic is independently unit-testable and has no Rich/Typer
dependency.

Precedence (spec section 26): an existing OS/process environment variable
always wins; `.env` is consulted only when that variable is unset or blank.
This module never mutates `os.environ` -- callers resolve credentials
through `resolve_secret`/`core.config.get_*_key` at the point of use instead.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

SUPPORTED_KEYS = ("GEMINI_API_KEY", "ELEVENLABS_API_KEY")


def find_project_env_file(start: Path | None = None) -> Path | None:
    """Walks upward from `start` (default: cwd) through parent directories
    looking for the nearest `.env`, so `videoedit` resolves the same project
    `.env` whether invoked from the project root or a subdirectory."""
    directory = (start if start is not None else Path.cwd()).resolve()
    for candidate_dir in (directory, *directory.parents):
        candidate = candidate_dir / ".env"
        if candidate.is_file():
            return candidate
    return None


def _is_blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def resolve_secret(
    name: str,
    *,
    start: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolves a single secret by name: OS/process env wins if set and
    non-blank; otherwise falls back to the nearest project `.env`; a blank
    value in either source counts as missing."""
    env = environ if environ is not None else os.environ
    os_value = env.get(name)
    if not _is_blank(os_value):
        return os_value

    env_file = find_project_env_file(start)
    if env_file is None:
        return None

    file_value = dotenv_values(env_file).get(name)
    return None if _is_blank(file_value) else file_value


def write_env_values(env_path: Path, values: Mapping[str, str]) -> Path:
    """Merges `values` into `env_path`'s `.env` file: existing `KEY=...`
    lines for keys we're updating are rewritten in place (order and
    unrelated lines/comments preserved); keys not already present are
    appended; the file is created if it doesn't exist. Restricts file
    permissions to owner-only on POSIX after writing (best-effort; skipped
    entirely on Windows, which has no equivalent chmod bits)."""
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(values)

    output_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if ("=" in stripped and not stripped.startswith("#")) else None
        if key is not None and key in remaining:
            output_lines.append(f"{key}={remaining.pop(key)}")
        else:
            output_lines.append(line)
    for key, value in remaining.items():
        output_lines.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if os.name == "posix":
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass  # best-effort only; never fail setup over a permission bit

    return env_path
