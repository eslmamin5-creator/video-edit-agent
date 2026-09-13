"""Dialect preservation guard regression tests (spec sections 8, 37) —
NON-NEGOTIABLE requirement: transcribe only, never translate/rewrite/
formalize/localize into another dialect.
"""
from __future__ import annotations

import pytest

from video_edit_agent.language.dialect_guard import (
    DialectGuardViolation,
    check_no_dialect_substitution,
    enforce,
)


def test_identical_text_always_passes():
    text = "عايز أعمل الفيديو ده دلوقتي"
    result = check_no_dialect_substitution(text, text)
    assert result.ok


def test_verbatim_disfluency_edit_within_same_dialect_passes():
    # Trimming a filler/repetition is fine as long as no NEW dialect's
    # markers appear and the source dialect's own markers survive.
    source = "عايز عايز أعمل الفيديو ده دلوقتي"
    final = "عايز أعمل الفيديو ده دلوقتي"
    result = check_no_dialect_substitution(source, final)
    assert result.ok


def test_egyptian_to_gulf_substitution_is_flagged():
    # "عايز" (Egyptian) silently rewritten to "أبي" (Gulf/Saudi) must be caught.
    source = "عايز أروح دلوقتي"
    final = "أبي أروح الحين"
    result = check_no_dialect_substitution(source, final)
    assert not result.ok
    assert "egyptian" in result.reason
    assert "gulf_saudi" in result.reason


def test_egyptian_to_msa_formalization_is_flagged():
    source = "عايز أعمل كده دلوقتي"
    final = "أريد أن أفعل هذا الآن"
    result = check_no_dialect_substitution(source, final)
    assert not result.ok


def test_text_with_no_dialect_markers_is_not_falsely_flagged():
    # Plain unmarked Arabic (no dialect signal words either way) shouldn't be
    # treated as a violation just because the wording changed.
    source = "هذا فيديو جميل جدا"
    final = "هذا فيديو رائع للغاية"
    result = check_no_dialect_substitution(source, final)
    assert result.ok


def test_enforce_raises_on_violation():
    with pytest.raises(DialectGuardViolation):
        enforce("عايز أروح دلوقتي", "أبي أروح الحين")


def test_enforce_does_not_raise_on_verbatim_match():
    enforce("عايز أروح دلوقتي", "عايز أروح دلوقتي")
