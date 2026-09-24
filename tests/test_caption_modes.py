"""Caption backing modes (none / adaptive / plate) and the caption-vs-motion hierarchy.
Synthetic styles and colours only. Timing, chunking and karaoke are never touched."""
from __future__ import annotations

from dataclasses import replace

import pytest

from video_edit_agent.brand.schema import Brand
from video_edit_agent.captions.brand_style import resolve_brand_caption_style
from video_edit_agent.captions.engine import uses_plate
from video_edit_agent.captions.modes import (
    MIN_CONTRAST,
    NORMAL,
    REDUCED,
    CaptionMode,
    apply_behavior,
    apply_mode,
    behavior_for,
    contrast_ratio,
    look_for_chunk,
    mode_of,
    needs_backing,
    parse_mode,
    recommended_mode,
)
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.direction.vocabulary import STRONG_PRIMARY

WHITE, BLACK, YELLOW = (255, 255, 255), (0, 0, 0), (255, 235, 59)


def _style(**kw) -> CaptionStyle:
    base = {"name": "t", "font_size": 64, "word_highlight": True, "background": "brand_box", "outline": 8.0,
            "outline_color": "&H737C0014"}
    base.update(kw)
    return CaptionStyle(**base)


def _brand(**captions) -> Brand:
    return Brand.model_validate({
        "name": "acme", "colors": {"primary": "#0A3D62", "secondary": "#F1F2F6", "accent": "#E58E26"},
        "typography": {"arabic": "Acme Arabic Sans", "latin": "Acme Sans"}, "captions": captions,
    })


# 7. the caption background supports none / adaptive / plate ------------------------------------------


def test_modes_are_none_adaptive_plate():
    assert {m.value for m in CaptionMode} == {"none", "adaptive", "plate"}
    assert parse_mode("PLATE") is CaptionMode.PLATE and parse_mode("nonsense") is None and parse_mode(None) is None


def test_none_draws_stroked_text_without_a_plate():
    out = apply_mode(_style(), "none")
    assert out.background == "outline" and not uses_plate(out)
    assert out.outline >= 2.0  # readability without a plate needs a real stroke
    assert out.outline_color == "&H00000000"  # the plate's tint is not a stroke colour
    assert mode_of(out) is CaptionMode.NONE


def test_adaptive_starts_without_a_plate_too():
    out = apply_mode(_style(), CaptionMode.ADAPTIVE)
    assert out.background == "outline" and not uses_plate(out)


def test_plate_keeps_the_translucent_brand_plate():
    original = _style()
    out = apply_mode(original, "plate")
    assert out == original and uses_plate(out) and mode_of(out) is CaptionMode.PLATE
    assert apply_mode(_style(background="outline"), "plate").background == "box"


def test_no_mode_leaves_the_style_alone():
    original = _style()
    assert apply_mode(original, None) is original
    assert apply_mode(original, "bogus") is original


def test_a_mode_only_changes_how_it_is_drawn_never_timing_or_chunking():
    original = _style(max_chars_per_line=22, line_break_chars=16, max_lines=2)
    for mode in CaptionMode:
        out = apply_mode(original, mode)
        for keep in ("word_highlight", "max_chars_per_line", "line_break_chars", "max_lines", "font_size", "primary_color",
                     "highlight_color"):
            assert getattr(out, keep) == getattr(original, keep), (mode, keep)


def test_none_is_the_default_candidate_for_premium_minimal_looks():
    assert recommended_mode("minimal") is CaptionMode.NONE
    assert recommended_mode("word-highlight") is CaptionMode.NONE
    assert recommended_mode("bold-social", premium=False) is CaptionMode.PLATE


def test_the_brand_can_choose_a_mode_and_defaults_to_the_current_look():
    assert Brand(name="x").captions.mode is None
    keep = resolve_brand_caption_style("word-highlight", _brand(background="brand_box")).style
    assert keep.background == "brand_box"
    none = resolve_brand_caption_style("word-highlight", _brand(background="brand_box", mode="none")).style
    assert none.background == "outline" and not uses_plate(none)
    plate = resolve_brand_caption_style("word-highlight", _brand(background="brand_box", mode="plate")).style
    assert uses_plate(plate)


def test_adaptive_backs_only_where_contrast_requires_it():
    style = _style()
    dark_backdrop = look_for_chunk(style, CaptionMode.ADAPTIVE, WHITE, BLACK)
    assert dark_backdrop.background == "outline" and not dark_backdrop.subtle_backing
    bright_backdrop = look_for_chunk(style, CaptionMode.ADAPTIVE, WHITE, YELLOW)
    assert bright_backdrop.subtle_backing and bright_backdrop.background == "brand_box"
    # never from a guess: an unmeasured backdrop draws no plate
    assert not look_for_chunk(style, CaptionMode.ADAPTIVE, WHITE, None).subtle_backing
    # none never backs, plate always does
    assert look_for_chunk(style, CaptionMode.NONE, WHITE, YELLOW).background == "outline"
    assert look_for_chunk(style, CaptionMode.PLATE, WHITE, BLACK).background == "brand_box"


def test_contrast_maths():
    assert contrast_ratio(WHITE, BLACK) == pytest.approx(21.0, abs=0.05)
    assert contrast_ratio(WHITE, WHITE) == pytest.approx(1.0)
    assert needs_backing(WHITE, YELLOW) and not needs_backing(WHITE, BLACK)
    assert MIN_CONTRAST >= 3.0


# 8. strong motion reduces caption competition without hiding accessibility text ------------------------


def test_strong_motion_reduces_caption_competition_but_never_hides_the_text():
    style = _style(font_size=64, word_highlight=True)
    for treatment in STRONG_PRIMARY:
        behavior = behavior_for(treatment)
        assert behavior is REDUCED and behavior.hidden is False
        out = apply_behavior(style, behavior)
        assert 0 < out.font_size < style.font_size  # quieter, still large enough to read
        assert out.font_size >= 0.8 * style.font_size
        assert out.word_highlight is False  # no second emphasis fighting the primary one
        assert out.primary_color == style.primary_color  # same colour: contrast is preserved
        assert out.max_lines == style.max_lines and out.bold == style.bold
    for treatment in ("speaker_static", "punch_in", "real_broll", None):
        assert behavior_for(treatment) is NORMAL
        assert apply_behavior(style, NORMAL) is style
    assert NORMAL.hidden is False and REDUCED.hidden is False


def test_reduced_behavior_keeps_the_mode_and_the_readable_stroke():
    style = apply_mode(_style(), "none")
    out = apply_behavior(style, REDUCED)
    assert out.background == "outline" and out.outline == replace(style).outline
