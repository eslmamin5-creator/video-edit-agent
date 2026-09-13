"""Shared local B-roll library location (spec section 20). This is a plain
folder of stock-style footage the user (or a brand profile) supplies; no
network access, no API keys, works fully offline.
"""
from __future__ import annotations

import os
from pathlib import Path


def default_library_dir() -> Path:
    """`~/.videoedit/broll_library` by default, overridable via the
    `VIDEOEDIT_BROLL_LIBRARY` environment variable."""
    override = os.environ.get("VIDEOEDIT_BROLL_LIBRARY")
    if override:
        return Path(override)
    return Path.home() / ".videoedit" / "broll_library"


def ensure_library_dir() -> Path:
    directory = default_library_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
