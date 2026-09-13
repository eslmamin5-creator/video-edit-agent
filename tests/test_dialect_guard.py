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


# --- V1.1 hardening: exact spec-section-5 dialect regression cases --------

def test_egyptian_sentence_must_not_shift_to_gulf_dialect():
    """"أنا عايز أعمل الفيديو ده دلوقتي" (Egyptian) must never come out the
    other side as "أنا أبي أسوي الفيديو الحين" (Gulf/Saudi) -- this is the
    exact failure mode the NON-NEGOTIABLE dialect rule exists to prevent."""
    source = "أنا عايز أعمل الفيديو ده دلوقتي"
    forbidden_gulf_rewrite = "أنا أبي أسوي الفيديو الحين"
    result = check_no_dialect_substitution(source, forbidden_gulf_rewrite)
    assert not result.ok
    with pytest.raises(DialectGuardViolation):
        enforce(source, forbidden_gulf_rewrite)


def test_gulf_sentence_must_not_shift_to_egyptian_dialect():
    """The reverse direction: a Saudi/Gulf sentence must not be silently
    "corrected" into Egyptian colloquial either -- the guard is symmetric."""
    source = "أبي أسوي الفيديو الحين"
    forbidden_egyptian_rewrite = "عايز أعمل الفيديو ده دلوقتي"
    result = check_no_dialect_substitution(source, forbidden_egyptian_rewrite)
    assert not result.ok
    with pytest.raises(DialectGuardViolation):
        enforce(source, forbidden_egyptian_rewrite)


def test_mixed_arabic_english_verbatim_text_passes_unchanged():
    """"أنا عايز أراجع الـ Campaign" must survive verbatim -- the embedded
    English term "Campaign" is not a dialect marker, so nothing in this
    project's architecture ever gets a chance to translate/localize it: the
    transcription path is verbatim-only (no translation step exists), and
    this guard confirms the identical mixed-language sentence round-trips
    cleanly through dialect checking without being flagged."""
    source = "أنا عايز أراجع الـ Campaign"
    result = check_no_dialect_substitution(source, source)
    assert result.ok
    assert "Campaign" in source  # sanity: the English term is actually present to preserve


def test_mixed_arabic_english_term_translated_away_is_still_flagged_if_dialect_also_shifts():
    """If a hypothetical transform both translated the English term AND
    shifted the surrounding Arabic to a different dialect, the guard must
    still catch the dialect half of that regression (translating "Campaign"
    alone is a separate, non-dialect concern the architecture prevents by
    having no translation step at all)."""
    source = "أنا عايز أراجع الـ Campaign قبل ما ننشرها"
    bad_rewrite = "أنا أبي أراجع الحملة قبل لا ننشرها"  # dialect shifted Egyptian -> Gulf
    result = check_no_dialect_substitution(source, bad_rewrite)
    assert not result.ok
