"""Offline-mode compliance (spec section 32): `agents/assembler/*` must
never import or reference a cloud provider (Gemini, ElevenLabs, Veo), so the
offline prohibition holds structurally, not just via a runtime flag check --
mirrors `test_creator_offline_compliance.py`'s pattern for Creator.
"""
from __future__ import annotations

import ast
from pathlib import Path

import video_edit_agent.agents.assembler as assembler_pkg

_FORBIDDEN_TOKENS = ("gemini", "elevenlabs", "veo")


def _assembler_source_files() -> list[Path]:
    root = Path(assembler_pkg.__file__).parent
    return sorted(root.glob("*.py"))


def test_assembler_module_files_exist():
    files = _assembler_source_files()
    assert len(files) >= 10


def test_assembler_source_never_imports_cloud_providers():
    for path in _assembler_source_files():
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


def test_assembler_source_never_mentions_cloud_provider_call_sites():
    banned_symbols = ("GeminiClient", "ElevenLabsClient", "VeoClient", "call_gemini", "call_elevenlabs", "call_veo")
    for path in _assembler_source_files():
        text = path.read_text(encoding="utf-8")
        for symbol in banned_symbols:
            assert symbol not in text, f"{path.name} references forbidden symbol: {symbol}"
