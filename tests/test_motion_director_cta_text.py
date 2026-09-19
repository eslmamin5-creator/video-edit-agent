"""CTA policy: a call to action is NONE by default. `build_motion_plan` never
invents CTA text (no language default, no Remotion `defaultProps.actionLabel`);
a CTA appears only when the caller or the Brand Profile provides its text."""
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


def _ctas(specs):
    return [s for s in specs if s.kind == AnimationKind.CTA]


def test_no_cta_by_default_for_any_language():
    for language in ("ar", "en", "auto"):
        assert _ctas(build_motion_plan(_edl(), _transcript(language), brand=None)) == []


def test_brand_without_cta_text_yields_no_cta():
    assert _ctas(build_motion_plan(_edl(), _transcript("ar"), brand=Brand(name="acme"))) == []


def test_cta_uses_brand_text_when_explicitly_configured():
    brand = Brand(name="acme")
    brand.cta.text_default = "اطلب الآن"
    (spec,) = _ctas(build_motion_plan(_edl(), _transcript("ar"), brand=brand))
    assert spec.text == "اطلب الآن"
    assert spec.extra.get("actionLabel") == "اطلب الآن"


def test_explicit_cta_text_wins_over_brand_text():
    brand = Brand(name="acme")
    brand.cta.text_default = "brand text"
    (spec,) = _ctas(build_motion_plan(_edl(), _transcript("en"), brand=brand, cta_text="explicit"))
    assert spec.text == "explicit"
