"""Offline-mode compliance (Creator spec section 13): `agents/creator/*` must
never import or reference a cloud provider (Gemini, ElevenLabs, Veo), so the
offline prohibition holds structurally, not just via a runtime flag check.
"""
from __future__ import annotations

import ast
from pathlib import Path

import video_edit_agent.agents.creator as creator_pkg

_FORBIDDEN_TOKENS = ("gemini", "elevenlabs", "veo")


def _creator_source_files() -> list[Path]:
    root = Path(creator_pkg.__file__).parent
    return sorted(root.glob("*.py"))


def test_creator_module_files_exist():
    files = _creator_source_files()
    assert len(files) >= 8


def test_creator_source_never_imports_cloud_providers():
    for path in _creator_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.lower() for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").lower()]
            else:
                continue
            for name in names:
                for token in _FORBIDDEN_TOKENS:
                    assert token not in name, f"{path.name} imports forbidden provider module: {name}"


def test_creator_source_never_mentions_cloud_provider_call_sites():
    """A looser textual check on top of the import check: no direct reference
    to cloud provider client classes/functions anywhere in Creator source."""
    banned_symbols = ("GeminiClient", "ElevenLabsClient", "VeoClient", "call_gemini", "call_elevenlabs", "call_veo")
    for path in _creator_source_files():
        text = path.read_text(encoding="utf-8")
        for symbol in banned_symbols:
            assert symbol not in text, f"{path.name} references forbidden symbol: {symbol}"
