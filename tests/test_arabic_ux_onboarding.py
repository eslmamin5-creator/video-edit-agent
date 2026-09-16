"""Arabic-first UX tests for v0.2.2 (spec sections 2, 3, 7, 8, 12, 17).

These cover interaction-language selection independent of transcript/dialect
language, that no dialect-rewriting is introduced by the new localization
keys, that the user-facing docs' relative links resolve to real files, and
that the PDF user guide build tooling produces valid non-empty output.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from video_edit_agent.language.detector import detect_interface_language
from video_edit_agent.localization.interface import t

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Interaction language selection (independent of transcript language) ---


def test_interaction_language_follows_arabic_input():
    assert detect_interface_language("إنت بتعمل إيه؟") == "ar"


def test_interaction_language_follows_english_input():
    assert detect_interface_language("What can you do?") == "en"


def test_onboarding_invitation_is_localized_not_translated_literally():
    ar = t("onboarding_invitation", "ar")
    en = t("onboarding_invitation", "en")
    assert ar == "ارفع فيديو، أو ابعت سكربت، أو حدّد فولدر المشاهد وقلّي عايز تعمل إيه."
    assert en == (
        "Upload a video, send a script, or point me at a folder of scenes "
        "and tell me what you want."
    )
    assert ar != en


def test_readiness_summary_localized_arabic_matches_spec_wording():
    ar = t("readiness_summary", "ar")
    assert ar == (
        "جاهز للاستخدام. Editor وCreator وAssembler متاحين. "
        "التشغيل المحلي يعمل بدون مفاتيح API، والميزات السحابية اختيارية."
    )


def test_localization_lookup_falls_back_to_english_for_unknown_language():
    # Interaction language selection must never crash on an unsupported
    # locale code -- it degrades to English rather than erroring.
    assert t("onboarding_invitation", "fr") == t("onboarding_invitation", "en")


# --- No dialect rewriting introduced by new interaction-language keys ---


def test_new_localization_keys_do_not_touch_dialect_guard_module():
    # The interaction-language keys added for onboarding must live purely in
    # localization/messages/*.json, never in the transcript/dialect-guard
    # path -- guard against someone wiring translation into transcription.
    from video_edit_agent.language import dialect_guard

    source = Path(dialect_guard.__file__).read_text(encoding="utf-8")
    assert "onboarding_invitation" not in source
    assert "readiness_summary" not in source


# --- Docs / link integrity ---


_MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _local_link_targets(markdown_path: Path) -> list[str]:
    text = markdown_path.read_text(encoding="utf-8")
    targets = []
    for match in _MD_LINK_RE.finditer(text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target)
    return targets


@pytest.mark.parametrize(
    "doc_path",
    [
        "README.md",
        "README.ar.md",
        "docs/USER_GUIDE.md",
        "docs/USER_GUIDE.ar.md",
    ],
)
def test_local_markdown_links_resolve_to_real_files(doc_path):
    full_path = REPO_ROOT / doc_path
    assert full_path.exists(), f"{doc_path} is missing"
    for target in _local_link_targets(full_path):
        resolved = (full_path.parent / target).resolve()
        assert resolved.exists(), f"{doc_path} links to missing file: {target}"


def test_readme_links_to_both_user_guides():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/USER_GUIDE.md" in readme
    assert "docs/USER_GUIDE.ar.md" in readme
    assert "docs/video-edit-agent-user-guide.pdf" in readme
    assert "docs/video-edit-agent-user-guide-ar.pdf" in readme


def test_readme_ar_links_to_both_user_guides():
    readme_ar = (REPO_ROOT / "README.ar.md").read_text(encoding="utf-8")
    assert "docs/USER_GUIDE.md" in readme_ar
    assert "docs/USER_GUIDE.ar.md" in readme_ar


# --- PDF guide build tooling smoke test ---


def test_build_user_guide_produces_both_pdfs(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi")

    script = REPO_ROOT / "scripts" / "build_user_guide.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    en_pdf = REPO_ROOT / "docs" / "video-edit-agent-user-guide.pdf"
    ar_pdf = REPO_ROOT / "docs" / "video-edit-agent-user-guide-ar.pdf"
    assert en_pdf.exists() and en_pdf.stat().st_size > 1000
    assert ar_pdf.exists() and ar_pdf.stat().st_size > 1000
    # Minimal PDF structural sanity check -- avoids depending on a PDF
    # parsing library just for a smoke test.
    assert en_pdf.read_bytes().startswith(b"%PDF-")
    assert ar_pdf.read_bytes().startswith(b"%PDF-")


def test_inline_shaping_never_corrupts_reportlab_tags_with_mixed_arabic_code():
    from scripts.build_user_guide import _render_inline

    text = "شغّل الأمر `videoedit edit my_take.mp4` وبعدين افتح `project.md`."
    rendered = _render_inline(text, rtl=True)

    assert rendered.count("<font") == rendered.count("</font>") == 2
    # The tag markup itself must survive shaping untouched -- only the
    # Arabic prose around it gets reshaped/bidi-reordered.
    assert "<font face='Courier'>videoedit edit my_take.mp4</font>" in rendered
    assert "<font face='Courier'>project.md</font>" in rendered
