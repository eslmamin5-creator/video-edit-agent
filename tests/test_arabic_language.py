"""Arabic detection / shaping / dialect-marker tests (spec sections 8, 15, 37)."""
from __future__ import annotations

from video_edit_agent.language.arabic import (
    dialect_marker_hits,
    has_code_switch,
    is_arabic_text,
    normalize_for_matching,
    shape_for_display,
)
from video_edit_agent.language.detector import detect_dominant_dialect, detect_interface_language


def test_is_arabic_text_detects_pure_arabic():
    assert is_arabic_text("مرحبا كيف حالك اليوم")


def test_is_arabic_text_rejects_pure_english():
    assert not is_arabic_text("hello how are you today")


def test_is_arabic_text_majority_rule_on_code_switch():
    # Mostly Arabic with one English brand word should still count as Arabic.
    assert is_arabic_text("انا حابب استخدم هاد المنتج جدا كتير Product")


def test_has_code_switch_detects_mixed_script():
    assert has_code_switch("عايز أعمل launch للمنتج بكرة")
    assert not has_code_switch("عايز أعمل كذا بكرة")
    assert not has_code_switch("I want to do this tomorrow")


def test_detect_interface_language_switches_on_input():
    assert detect_interface_language("مرحبا") == "ar"
    assert detect_interface_language("hello") == "en"


def test_egyptian_dialect_markers_detected():
    # Exact-style Egyptian example per spec section 37.
    text = "عايز أعمل الفيديو ده دلوقتي، مش قادر أستنى أكتر"
    hits = dialect_marker_hits(text)
    assert hits["egyptian"] > 0
    assert hits["gulf_saudi"] == 0


def test_gulf_saudi_dialect_markers_detected():
    text = "أبي أسوي الفيديو الحين، وايد أفكار عندي"
    hits = dialect_marker_hits(text)
    assert hits["gulf_saudi"] > 0
    assert hits["egyptian"] == 0


def test_msa_markers_detected():
    text = "أريد أن أفعل هذا الآن، لماذا التأخير"
    hits = dialect_marker_hits(text)
    assert hits["msa"] > 0


def test_detect_dominant_dialect_labels_egyptian_transcript(sample_transcript):
    from video_edit_agent.core.schemas import Segment, Transcript

    t = Transcript(
        provider="test",
        segments=[Segment(id="s0", start=0, end=1, text="عايز أعمل الفيديو ده دلوقتي")],
    )
    assert detect_dominant_dialect(t) == "egyptian"


def test_normalize_for_matching_strips_harakat_and_tatweel():
    with_diacritics = "مَرْحَـبًا"
    normalized = normalize_for_matching(with_diacritics)
    assert "ـ" not in normalized  # tatweel gone
    assert all(0x064B > ord(c) or ord(c) > 0x065F for c in normalized)  # no harakat block chars


def test_shape_for_display_does_not_crash_without_arabic_libs_and_returns_string():
    # Whether or not arabic-reshaper/python-bidi are installed in the test
    # env, shape_for_display must always return a usable string (spec section
    # 43: optional deps degrade gracefully, never raise).
    result = shape_for_display("مرحبا بالجميع")
    assert isinstance(result, str)
    assert len(result) > 0

    result_en = shape_for_display("hello world")
    assert result_en == "hello world"
