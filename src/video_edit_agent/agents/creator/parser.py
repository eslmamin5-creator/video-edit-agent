"""Script ingestion (Creator spec section 3).

`.txt` and `.md` work unconditionally with zero extra dependencies. `.docx`
and `.pdf` are supported only through reliable text extraction and degrade
gracefully (a clear `ScriptParseError`, never a crash) when the optional
library isn't installed. No OCR.
"""
from __future__ import annotations

from pathlib import Path

_SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}


class ScriptParseError(Exception):
    pass


def parse_script(path: Path) -> str:
    """Returns the script's plain text content."""
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise ScriptParseError(
            f"Unsupported script format '{suffix}'. Supported: {sorted(_SUPPORTED_SUFFIXES)}"
        )
    if not path.exists():
        raise ScriptParseError(f"Script file not found: {path}")

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ScriptParseError(f"Unsupported script format '{suffix}'")  # unreachable


def _parse_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError as exc:
        raise ScriptParseError(
            "Reading .docx scripts requires the optional 'python-docx' package "
            "(pip install python-docx). Use a .txt/.md script instead, or install it."
        ) from exc
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ScriptParseError(
            "Reading .pdf scripts requires the optional 'pypdf' package "
            "(pip install pypdf). Use a .txt/.md script instead, or install it."
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
