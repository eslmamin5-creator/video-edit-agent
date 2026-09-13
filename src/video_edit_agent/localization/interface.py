"""Localized CLI output (spec section 10).

Uses the OS locale / persisted config to pick a message language, and falls
back to English if a key or the whole locale file is missing — never crashes
the CLI just because a translation is absent.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_MESSAGES_DIR = Path(__file__).parent / "messages"


@lru_cache(maxsize=4)
def _load(lang: str) -> dict[str, str]:
    path = _MESSAGES_DIR / f"{lang}.json"
    if not path.exists():
        path = _MESSAGES_DIR / "en.json"
    return json.loads(path.read_text(encoding="utf-8"))


def t(key: str, lang: str = "en", **kwargs) -> str:
    messages = _load(lang)
    template = messages.get(key) or _load("en").get(key, key)
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template
