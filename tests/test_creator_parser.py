"""Script ingestion tests (Creator spec section 3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from video_edit_agent.agents.creator.parser import ScriptParseError, parse_script


def test_parse_txt_returns_raw_text(tmp_path: Path):
    p = tmp_path / "script.txt"
    p.write_text("Hello world.\nSecond line.", encoding="utf-8")
    assert parse_script(p) == "Hello world.\nSecond line."


def test_parse_md_returns_raw_text(tmp_path: Path):
    p = tmp_path / "script.md"
    p.write_text("# Title\nBody text.", encoding="utf-8")
    assert parse_script(p) == "# Title\nBody text."


def test_parse_unsupported_suffix_raises(tmp_path: Path):
    p = tmp_path / "script.xyz"
    p.write_text("data", encoding="utf-8")
    with pytest.raises(ScriptParseError):
        parse_script(p)


def test_parse_missing_file_raises(tmp_path: Path):
    p = tmp_path / "missing.txt"
    with pytest.raises(ScriptParseError):
        parse_script(p)


def test_parse_docx_without_dependency_raises_clear_error(tmp_path: Path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("no module named docx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = tmp_path / "script.docx"
    p.write_bytes(b"not a real docx")
    with pytest.raises(ScriptParseError, match="python-docx"):
        parse_script(p)


def test_parse_pdf_without_dependency_raises_clear_error(tmp_path: Path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no module named pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = tmp_path / "script.pdf"
    p.write_bytes(b"not a real pdf")
    with pytest.raises(ScriptParseError, match="pypdf"):
        parse_script(p)
