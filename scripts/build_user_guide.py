"""Build the user-facing PDF guides from docs/USER_GUIDE*.md.

Reproducible, documentation-only tooling (spec v0.2.2 section 12): this
script and its dependencies (reportlab, arabic-reshaper, python-bidi) are
never imported by the runtime pipeline -- install them with
``pip install -e ".[docs]"``.

Usage:
    python scripts/build_user_guide.py            # builds both PDFs
    python scripts/build_user_guide.py --en-only
    python scripts/build_user_guide.py --ar-only

Source of truth is Markdown (`docs/USER_GUIDE.md` / `docs/USER_GUIDE.ar.md`)
-- never hand-edit the generated PDFs.

Arabic font: this script needs a TTF font with Arabic glyphs to render the
Arabic PDF with correct shaping. No font file is committed to this
repository (spec section 11: "no proprietary font files committed"). On
first run it downloads the open-license (SIL OFL) Noto Naskh Arabic font
into a local, gitignored cache (`.fonts_cache/`) if not already present. If
there is no network access and no cached/local font is found, the script
fails with a clear message and exact next steps rather than silently
producing broken Arabic shaping.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
FONT_CACHE_DIR = REPO_ROOT / ".fonts_cache"
ARABIC_FONT_NAME = "NotoNaskhArabic-Regular.ttf"
ARABIC_FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/"
    "notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf"
)


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _require(module_name: str, extra_hint: str = "docs") -> None:
    try:
        __import__(module_name)
    except ImportError:
        _fail(
            f"missing dependency '{module_name}'. Install PDF-build tooling "
            f'with:  pip install -e ".[{extra_hint}]"'
        )


def _ensure_arabic_font() -> Path:
    """Return a usable Arabic TTF path, downloading it once if needed."""
    FONT_CACHE_DIR.mkdir(exist_ok=True)
    cached = FONT_CACHE_DIR / ARABIC_FONT_NAME
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    try:
        print(f"Downloading Arabic font (one-time, OFL-licensed) to {cached} ...")
        urllib.request.urlretrieve(ARABIC_FONT_URL, cached)
        if cached.stat().st_size == 0:
            raise OSError("downloaded file is empty")
        return cached
    except Exception as exc:  # noqa: BLE001 - report clearly, don't crash cryptically
        if cached.exists():
            cached.unlink()
        _fail(
            "could not obtain an Arabic-capable font and none is cached. "
            f"Network download failed ({exc}). Fix by either: (1) running "
            "this script again with network access, or (2) manually placing "
            f"any Arabic-capable TTF file at {cached}."
        )
    raise AssertionError("unreachable")


# --- Minimal Markdown model -------------------------------------------------
# A tiny, purpose-built parser: headers, bullet lists, tables, code fences,
# blockquotes, and inline **bold** -- exactly what docs/USER_GUIDE*.md use.
# Not a general-purpose Markdown engine.

class Block:
    pass


class Heading(Block):
    def __init__(self, level: int, text: str):
        self.level = level
        self.text = text


class Paragraph(Block):
    def __init__(self, text: str):
        self.text = text


class BulletList(Block):
    def __init__(self, items: list[str]):
        self.items = items


class Table(Block):
    def __init__(self, header: list[str], rows: list[list[str]]):
        self.header = header
        self.rows = rows


class CodeBlock(Block):
    def __init__(self, lines: list[str]):
        self.lines = lines


class Quote(Block):
    def __init__(self, text: str):
        self.text = text


_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")


_INLINE_SPAN_RE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_inline(text: str, *, rtl: bool) -> str:
    """Turn Markdown inline syntax into reportlab markup.

    Bidi reshaping (arabic_reshaper + get_display) has no notion of XML
    tags, so it must never run on a string that already contains inserted
    `<b>`/`<font>` tags -- it reorders the tag characters right along with
    the Arabic text and corrupts the markup. Instead we shape each plain
    text run and each tag's inner content independently, in logical order,
    before wrapping the tagged runs in their tags.
    """

    def shape(segment: str) -> str:
        escaped = _escape_xml(segment)
        if rtl and _has_arabic(escaped):
            return _shape_arabic(escaped)
        return escaped

    text = _LINK_RE.sub(r"\1", text)  # PDF is not interactive; keep link text only

    parts: list[str] = []
    pos = 0
    for m in _INLINE_SPAN_RE.finditer(text):
        if m.start() > pos:
            parts.append(shape(text[pos : m.start()]))
        if m.group(1) is not None:
            parts.append(f"<b>{shape(m.group(1))}</b>")
        else:
            parts.append(f"<font face='Courier'>{shape(m.group(2))}</font>")
        pos = m.end()
    if pos < len(text):
        parts.append(shape(text[pos:]))
    return "".join(parts)


def parse_markdown(md_text: str) -> list[Block]:
    lines = md_text.splitlines()
    blocks: list[Block] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            blocks.append(CodeBlock(code_lines))
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            blocks.append(Heading(level, text))
            i += 1
            continue

        if stripped.startswith(">"):
            blocks.append(Quote(stripped.lstrip(">").strip()))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [
                [cell.strip() for cell in tl.strip("|").split("|")]
                for tl in table_lines
            ]
            header = rows[0]
            body = [r for r in rows[2:]] if len(rows) > 2 else []
            blocks.append(Table(header, body))
            continue

        if stripped.startswith(("- ", "* ")):
            items = []
            while i < n and lines[i].strip().startswith(("- ", "* ")):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(BulletList(items))
            continue

        # Plain paragraph: gather until a blank line or a new block starter.
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() and not lines[i].strip().startswith(("#", ">", "|", "```", "- ", "* ")):
            para_lines.append(lines[i].strip())
            i += 1
        blocks.append(Paragraph(" ".join(para_lines)))

    return blocks


# --- PDF rendering -----------------------------------------------------------

def _shape_arabic(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


def _has_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


def render_pdf(blocks: list[Block], output_path: Path, *, rtl: bool, title: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageTemplate,
        Preformatted,
        Spacer,
        TableStyle,
    )
    from reportlab.platypus import (
        Paragraph as RLParagraph,
    )
    from reportlab.platypus import (
        Table as RLTable,
    )

    base_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    align = TA_RIGHT if rtl else TA_LEFT

    if rtl:
        font_path = _ensure_arabic_font()
        pdfmetrics.registerFont(TTFont("ArabicBody", str(font_path)))
        base_font = "ArabicBody"
        bold_font = "ArabicBody"  # variable-weight TTF; bold handled via markup weight fallback

    def maybe_shape(text: str) -> str:
        return _shape_arabic(text) if rtl and _has_arabic(text) else text

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "GuideH1", parent=styles["Title"], fontName=bold_font, alignment=align,
        textColor=colors.HexColor("#1a3c5e"), spaceAfter=14, spaceBefore=6,
    )
    h2 = ParagraphStyle(
        "GuideH2", parent=styles["Heading2"], fontName=bold_font, alignment=align,
        textColor=colors.HexColor("#1a3c5e"), spaceBefore=16, spaceAfter=8,
        borderPadding=0,
    )
    body = ParagraphStyle(
        "GuideBody", parent=styles["BodyText"], fontName=base_font, alignment=align,
        leading=15, spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "GuideBullet", parent=body, leftIndent=0 if rtl else 14,
        rightIndent=14 if rtl else 0, bulletIndent=0,
    )
    quote = ParagraphStyle(
        "GuideQuote", parent=body, textColor=colors.HexColor("#3a6ea5"),
        backColor=colors.HexColor("#eef4fa"), borderPadding=8, spaceAfter=10,
    )

    story = []
    for block in blocks:
        if isinstance(block, Heading):
            text = _render_inline(block.text, rtl=rtl)
            style = h1 if block.level == 1 else h2
            story.append(RLParagraph(text, style))
        elif isinstance(block, Paragraph):
            text = _render_inline(block.text, rtl=rtl)
            story.append(RLParagraph(text, body))
        elif isinstance(block, Quote):
            text = _render_inline(block.text, rtl=rtl)
            story.append(RLParagraph(text, quote))
        elif isinstance(block, BulletList):
            for item in block.items:
                text = _render_inline(item, rtl=rtl)
                marker = "• " if not rtl else " •"
                text = f"{marker}{text}" if not rtl else f"{text}{marker}"
                story.append(RLParagraph(text, bullet))
            story.append(Spacer(1, 6))
        elif isinstance(block, CodeBlock):
            story.append(Preformatted("\n".join(block.lines), styles["Code"]))
            story.append(Spacer(1, 8))
        elif isinstance(block, Table):
            header = [maybe_shape(c) for c in block.header]
            rows = [[maybe_shape(c) for c in r] for r in block.rows]
            data = [header] + rows
            tbl = RLTable(data, hAlign="RIGHT" if rtl else "LEFT", repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), base_font),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ALIGN", (0, 0), (-1, -1), "RIGHT" if rtl else "LEFT"),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 10))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"video-edit-agent — {title} — {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=title,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_footer)])
    doc.build(story)


def build_guide(md_path: Path, output_path: Path, *, rtl: bool, title: str) -> None:
    if not md_path.exists():
        _fail(f"source Markdown file not found: {md_path}")
    blocks = parse_markdown(md_path.read_text(encoding="utf-8"))
    render_pdf(blocks, output_path, rtl=rtl, title=title)
    print(f"Built {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-only", action="store_true")
    parser.add_argument("--ar-only", action="store_true")
    args = parser.parse_args()

    _require("reportlab")

    build_en = not args.ar_only
    build_ar = not args.en_only

    if build_en:
        build_guide(
            DOCS_DIR / "USER_GUIDE.md",
            DOCS_DIR / "video-edit-agent-user-guide.pdf",
            rtl=False,
            title="User Guide",
        )
    if build_ar:
        _require("arabic_reshaper")
        _require("bidi", extra_hint="docs")
        build_guide(
            DOCS_DIR / "USER_GUIDE.ar.md",
            DOCS_DIR / "video-edit-agent-user-guide-ar.pdf",
            rtl=True,
            title="دليل المستخدم",
        )


if __name__ == "__main__":
    main()
