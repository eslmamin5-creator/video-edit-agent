"""Regression tests for the generic "Learn more" CTA bug (Review-First
Editing Workflow spec section 11 item D): `build_motion_plan`'s CTA spec must
never rely on Remotion's own hardcoded English `defaultProps.actionLabel`,
and must never silently show English text on Arabic-language content."""
from __future__ import annotations

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import EDL, AnimationKind, EDLClip, Segment, Transcript, Word
from video_edit_agent.motion.director import build_motion_plan


def _transcript(language: str) -> Transcript:
    word_objs = [Word(word=w, start=i * 0.5, end=i * 0.5 + 0.4) for i, w in enumerate(["hello", "world"])]
    segment = Segment(id="seg0", start=0.0, end=1.0, text="hello world", words=word_objs)
    return Transcript(provider="test", language=language, segments=[segment])


def _edl(duration: float = 10.0) -> EDL:
    return EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[EDLClip(source_file="a.mp4", source_in=0.0, source_out=duration, timeline_in=0.0, timeline_out=duration)],
    )


def _cta_spec(specs):
    ctas = [s for s in specs if s.kind == AnimationKind.CTA]
    assert len(ctas) == 1
    return ctas[0]


def test_cta_defaults_to_arabic_text_for_arabic_transcripts():
    spec = _cta_spec(build_motion_plan(_edl(), _transcript("ar"), brand=None))
    assert spec.text
    assert spec.text != "Learn more"
    assert spec.extra.get("actionLabel") == spec.text


def test_cta_defaults_to_english_text_for_non_arabic_transcripts():
    spec = _cta_spec(build_motion_plan(_edl(), _transcript("en"), brand=None))
    assert spec.text == "Learn more"
    assert spec.extra.get("actionLabel") == "Learn more"


def test_cta_prefers_brand_text_default_over_language_default():
    brand = Brand(name="acme")
    brand.cta.text_default = "اطلب الآن"
    spec = _cta_spec(build_motion_plan(_edl(), _transcript("ar"), brand=brand))
    assert spec.text == "اطلب الآن"
    assert spec.extra.get("actionLabel") == "اطلب الآن"


def test_cta_never_has_empty_text():
    for language in ("ar", "en", "auto"):
        spec = _cta_spec(build_motion_plan(_edl(), _transcript(language), brand=None))
        assert spec.text.strip() != ""
