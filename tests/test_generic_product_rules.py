"""Generic product rules (Final Pre-Render Review Pass): review-first gate,
B-roll lifecycle, Brand-Profile-driven visuals, logo policy, CTA-none and
timing-preserving transcript corrections.

Every test uses synthetic brands/projects ("acme", "zed") -- the rules must
hold for any Brand Profile, not for one client. The single client test checks
that the shipped client profile *configuration* wires through the generic code.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from tests.conftest import make_transcript, requires_ffmpeg
from video_edit_agent.brand import loader as brand_loader
from video_edit_agent.brand.logo_policy import plan_logo
from video_edit_agent.brand.schema import Brand, BrandColors, LogoMode
from video_edit_agent.broll.treatment import (
    Treatment,
    TreatmentDecision,
    assign_treatments,
)
from video_edit_agent.captions.brand_style import resolve_brand_caption_style
from video_edit_agent.captions.styles import hex_to_ass
from video_edit_agent.core import pipeline as pipeline_mod
from video_edit_agent.core.schemas import (
    EDL,
    AnimationKind,
    BrollPlanItem,
    BrollSourceKind,
    EDLClip,
)
from video_edit_agent.motion.director import build_motion_plan
from video_edit_agent.review import state as review_state
from video_edit_agent.review.corrections import add_correction, apply_corrections, load_corrections
from video_edit_agent.review.schemas import TranscriptCorrection

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "video_edit_agent"


def _brand(**overrides) -> Brand:
    data = {
        "name": "acme",
        "colors": {"primary": "#0A3D62", "secondary": "#F1F2F6", "accent": "#E58E26"},
        "typography": {"arabic": "Acme Arabic Sans", "latin": "Acme Sans"},
    }
    data.update(overrides)
    return Brand.model_validate(data)


def _make_logo(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (200, 80), (255, 255, 255, 255)).save(path)
    return path


# --- review-first gate ------------------------------------------------------


def test_review_gate_defaults_to_not_ready(tmp_path: Path):
    review_dir = tmp_path / "edit" / "review"
    assert review_state.is_ready_for_final_render(review_dir) is False
    state = review_state.load_review_state(review_dir)
    assert state.ready_for_final_render is False
    assert state.bypassed is False
    assert state.broll_generation_approved is False


def test_approval_and_explicit_bypass_are_the_only_ways_through(tmp_path: Path):
    approved = tmp_path / "a"
    review_state.approve(approved, "ok")
    state = review_state.load_review_state(approved)
    assert state.ready_for_final_render and state.broll_generation_approved and not state.bypassed

    bypassed = tmp_path / "b"
    review_state.bypass(bypassed, "--yes")
    state = review_state.load_review_state(bypassed)
    assert state.ready_for_final_render and state.bypassed


@pytest.mark.parametrize("project_name", ["zed", "acme_launch"])
def test_default_run_plans_only_for_any_project(monkeypatch, tmp_path: Path, sample_transcript, sample_edl, project_name):
    """No approval on record -> `review=None` (the default) must not render,
    generate B-roll, or render motion, regardless of project."""
    from video_edit_agent.core.media import MediaInfo

    def boom(name):
        def _raise(*_a, **_k):
            raise AssertionError(f"{name} must not run before approval")

        return _raise

    monkeypatch.setattr(pipeline_mod, "probe", lambda path, *_a, **_k: MediaInfo(
        path=Path(path), width=sample_edl.width, height=sample_edl.height, fps=sample_edl.fps,
        duration=sample_edl.total_duration, has_audio=True, video_codec="h264", audio_codec="aac",
    ))
    monkeypatch.setattr(pipeline_mod, "extract_audio", lambda *_a, **_k: None)

    class Router:
        def __init__(self, *_a, **_k):
            pass

        def transcribe(self, *_a, **_k):
            return sample_transcript

    monkeypatch.setattr(pipeline_mod, "TranscriptionRouter", Router)
    monkeypatch.setattr(pipeline_mod, "save_transcript", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "build_edl", lambda *_a, **_k: (sample_edl, []))
    monkeypatch.setattr(pipeline_mod, "write_takes_packed", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "validate_edl", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "write_captions", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "generate_preview_frames", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("skip")))
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", boom("render_ffmpeg"))
    monkeypatch.setattr(pipeline_mod, "render_motion", boom("render_motion"))
    monkeypatch.setattr(pipeline_mod, "render_subject_cutout", boom("render_subject_cutout"))
    monkeypatch.setattr(pipeline_mod, "compose_with_cards", boom("compose_with_cards"))
    seen: list = []
    monkeypatch.setattr(pipeline_mod, "plan_broll", lambda *_a, **k: seen.append(k["allow_generation"]) or [])

    project = tmp_path / project_name
    project.mkdir()
    source = project / "source.mp4"
    source.write_bytes(b"x")

    result = pipeline_mod.run_pipeline(source, offline=False)  # review=None: the default

    assert result.final_output is None
    assert result.review_dir is not None
    assert seen == [False]
    assert review_state.load_review_state(result.review_dir).ready_for_final_render is False


# --- B-roll lifecycle -------------------------------------------------------


def _slot(start, end, concept="an idea", source=BrollSourceKind.NONE, confidence=0.5) -> BrollPlanItem:
    return BrollPlanItem(
        timeline_start=start, timeline_end=end, purpose="context", spoken_concept=concept,
        recommended_visual="a visual", source=source, confidence=confidence,
    )


def test_generated_broll_is_optional_and_capped():
    items = [_slot(i * 10.0, i * 10.0 + 8.0, confidence=0.5 + i / 100) for i in range(6)]
    assign_treatments(items, None)
    generated = [i for i in items if i.treatment == Treatment.GENERATED_BROLL.value]
    assert 0 < len(generated) <= 2  # never imagery for every sentence
    assert all(i.generate_later for i in generated)
    assert all(not i.generate_later for i in items if i not in generated)
    assert {i.treatment for i in items} - {Treatment.GENERATED_BROLL.value}  # editor chose alternatives


def test_editor_decisions_override_the_heuristic():
    items = [_slot(0.0, 8.0), _slot(10.0, 18.0)]
    decisions = [TreatmentDecision(timeline_start=0.0, timeline_end=8.0, treatment=Treatment.KINETIC_TYPOGRAPHY)]
    assign_treatments(items, decisions)
    assert items[0].treatment == "kinetic_typography"
    assert items[0].generate_later is False


def test_local_asset_slots_use_local_broll():
    items = [_slot(0.0, 8.0, source=BrollSourceKind.LOCAL_LIBRARY)]
    assign_treatments(items, None)
    assert items[0].treatment == "local_broll"


def test_only_generated_treatment_after_approval_reaches_the_generators(monkeypatch, tmp_path: Path):
    from video_edit_agent.broll import planner as planner_mod

    generated_for: list[str] = []
    monkeypatch.setattr(planner_mod.local, "find_broll", lambda item, *_a, **_k: item)
    monkeypatch.setattr(
        planner_mod.gemini_image, "generate_broll_image",
        lambda item, *_a, **_k: generated_for.append(item.treatment) or item,
    )
    monkeypatch.setattr(
        planner_mod.veo, "generate_broll_video",
        lambda item, *_a, **_k: generated_for.append(item.treatment) or item,
    )
    slots = [_slot(0.0, 8.0), _slot(10.0, 18.0)]
    monkeypatch.setattr(planner_mod, "select_broll_moments", lambda *_a, **_k: slots)
    decisions = [
        TreatmentDecision(timeline_start=0.0, timeline_end=8.0, treatment=Treatment.STAY_ON_SPEAKER),
        TreatmentDecision(timeline_start=10.0, timeline_end=18.0, treatment=Treatment.GENERATED_BROLL),
    ]
    edl = EDL(width=1080, height=1920, clips=[
        EDLClip(source_file="s.mp4", source_in=0, source_out=20, timeline_in=0, timeline_out=20, caption_refs=[])
    ])
    tr = make_transcript([("hello", 0.0, 20.0)])

    planner_mod.plan_broll(edl, tr, tmp_path / "local", tmp_path / "gen", allow_generation=False, decisions=decisions)
    assert generated_for == []  # nothing before approval

    slots2 = [_slot(0.0, 8.0), _slot(10.0, 18.0)]
    monkeypatch.setattr(planner_mod, "select_broll_moments", lambda *_a, **_k: slots2)
    planner_mod.plan_broll(edl, tr, tmp_path / "local", tmp_path / "gen", allow_generation=True, decisions=decisions)
    assert generated_for and set(generated_for) == {"generated_broll"}  # never for stay_on_speaker


def test_generated_broll_prompt_forbids_text_logos_and_ai_look():
    from video_edit_agent.broll.prompt import build_broll_prompt

    prompt = build_broll_prompt(_slot(0.0, 5.0)).lower()
    for phrase in ("no readable text", "fake ui", "logos", "real-world lighting", "stock-ai"):
        assert phrase in prompt


# --- Brand Profile drives visuals ------------------------------------------


def test_brand_values_flow_into_caption_style():
    brand = _brand(captions={"word_highlight": True, "use_brand_colors": True, "background": "brand_box"})
    resolved = resolve_brand_caption_style("minimal", brand)
    style = resolved.style
    assert style.font_ar == "Acme Arabic Sans"
    assert style.highlight_color == hex_to_ass("#E58E26")
    assert style.primary_color == hex_to_ass("#FFFFFF")
    assert style.background == "brand_box"
    assert hex_to_ass("#0A3D62")[4:10] in style.outline_color  # box tinted with the brand primary
    assert style.word_highlight is True


def test_missing_brand_accent_does_not_invent_a_color():
    brand = _brand(colors={"primary": "#0A3D62", "secondary": "#DDEEFF"}, captions={"word_highlight": True})
    assert brand.colors.accent is None
    resolved = resolve_brand_caption_style("minimal", brand)
    assert resolved.style.highlight_color == hex_to_ass("#DDEEFF")  # the brand's own secondary
    assert any("MISSING accent" in n for n in resolved.notes)
    assert "FFCC00" not in resolved.style.highlight_color.upper()
    assert BrandColors().accent is None


def test_missing_typography_is_reported_not_invented():
    resolved = resolve_brand_caption_style("minimal", Brand(name="bare"))
    assert any("MISSING typography" in n for n in resolved.notes)


def test_brand_summary_reports_missing_values():
    from video_edit_agent.review.builder import build_brand_summary

    summary = build_brand_summary(Brand(name="bare"), None, None)
    assert summary.cta_status == "NONE"
    assert summary.cta_text is None
    assert summary.missing  # accent / typography / logo are reported, not invented


# --- logo policy -------------------------------------------------------------


def test_logo_mode_defaults_to_end_card_and_bug_is_opt_in(tmp_path: Path):
    brand = _brand()
    assert brand.logo.behavior.mode is LogoMode.END_CARD
    logo = _make_logo(tmp_path / "logo.png")

    default_plan = plan_logo(brand, logo, 1080, 1920, 30)
    assert default_plan.persistent_bug is False
    assert default_plan.end_card is not None and default_plan.intro is None

    opt_in = plan_logo(brand, logo, 1080, 1920, 30, mode_override=LogoMode.PERSISTENT_BUG)
    assert opt_in.persistent_bug is True
    assert opt_in.cards == []


@pytest.mark.parametrize(
    ("mode", "bug", "intro", "end"),
    [
        (LogoMode.NONE, False, False, False),
        (LogoMode.INTRO, False, True, False),
        (LogoMode.END_CARD, False, False, True),
        (LogoMode.INTRO_AND_END, False, True, True),
        (LogoMode.PERSISTENT_BUG, True, False, False),
    ],
)
def test_every_logo_mode(tmp_path: Path, mode, bug, intro, end):
    plan = plan_logo(_brand(), _make_logo(tmp_path / "l.png"), 1080, 1920, 30, mode_override=mode)
    assert (plan.persistent_bug, plan.intro is not None, plan.end_card is not None) == (bug, intro, end)


def test_brand_profile_overrides_logo_mode_and_card_colors(tmp_path: Path):
    brand = _brand(logo={"behavior": {"mode": "intro_and_end", "duration": 1.5, "background": "gradient"}})
    plan = plan_logo(brand, _make_logo(tmp_path / "l.png"), 1080, 1920, 30)
    assert plan.mode is LogoMode.INTRO_AND_END
    card = plan.end_card
    assert card.duration == 1.5
    assert (card.background_top, card.background_bottom) == ("#0A3D62", "#F1F2F6")


def test_missing_logo_is_reported(tmp_path: Path):
    plan = plan_logo(_brand(), None, 1080, 1920, 30)
    assert plan.cards == [] and any("MISSING logo" in w for w in plan.warnings)


@requires_ffmpeg
def test_end_card_renderer_is_brand_driven(tmp_path: Path):
    """Two different brands must produce visibly different cards with the
    background taken from each brand's own primary color."""
    from PIL import Image

    from video_edit_agent.render.end_card import render_card_frame

    logo = _make_logo(tmp_path / "logo.png")
    pixels = []
    for name, primary in (("one", "#0A3D62"), ("two", "#8E1B3A")):
        brand = _brand(name=name, colors={"primary": primary, "secondary": primary})
        spec = plan_logo(brand, logo, 360, 640, 30).end_card
        out = render_card_frame(spec, tmp_path / f"{name}.jpg", tmp_path / f"w_{name}")
        with Image.open(out) as img:
            assert img.size == (360, 640)
            pixels.append(img.convert("RGB").getpixel((5, 5)))
    assert pixels[0] != pixels[1]
    r, g, b = pixels[0]
    assert abs(r - 0x0A) < 12 and abs(g - 0x3D) < 12 and abs(b - 0x62) < 12


# --- word highlight: upcoming words keep the text color -------------------


def test_karaoke_style_paints_unspoken_words_in_the_text_color():
    from video_edit_agent.captions.engine import build_ass
    from video_edit_agent.captions.styles import CaptionStyle

    style = CaptionStyle(
        name="t", word_highlight=True, primary_color=hex_to_ass("#FFFFFF"), highlight_color=hex_to_ass("#00AA33"),
    )
    ass = build_ass(make_transcript([("one two three four", 0.0, 4.0)]), _edl(4.0), style)
    style_line = next(line for line in ass.splitlines() if line.startswith("Style:"))
    fields = style_line.split(",")
    # ASS: PrimaryColour = after the word is sung, SecondaryColour = before.
    assert fields[3] == hex_to_ass("#00AA33")
    assert fields[4] == hex_to_ass("#FFFFFF")

    plain = build_ass(make_transcript([("one two", 0.0, 2.0)]), _edl(2.0), dataclasses.replace(style, word_highlight=False))
    plain_style = next(line for line in plain.splitlines() if line.startswith("Style:")).split(",")
    assert plain_style[3] == hex_to_ass("#FFFFFF")  # non-karaoke text is the text color


# --- CTA is none by default --------------------------------------------------


def _edl(duration: float = 20.0) -> EDL:
    return EDL(width=1080, height=1920, clips=[
        EDLClip(source_file="s.mp4", source_in=0, source_out=duration, timeline_in=0, timeline_out=duration, caption_refs=[])
    ])


def test_cta_defaults_to_none():
    tr = make_transcript([("this is a long enough spoken line for a hook", 0.0, 20.0)])
    specs = build_motion_plan(_edl(), tr, _brand())
    assert all(s.kind is not AnimationKind.CTA for s in specs)
    assert _brand().cta_text is None


def test_cta_only_when_explicit():
    tr = make_transcript([("this is a long enough spoken line for a hook", 0.0, 20.0)])
    from_user = build_motion_plan(_edl(), tr, _brand(), cta_text="Custom CTA")
    assert [s.text for s in from_user if s.kind is AnimationKind.CTA] == ["Custom CTA"]
    from_brand = build_motion_plan(_edl(), tr, _brand(cta={"enabled": True, "text": "Brand CTA"}))
    assert [s.text for s in from_brand if s.kind is AnimationKind.CTA] == ["Brand CTA"]
    disabled = _brand(cta={"enabled": False, "text": "Brand CTA"})
    assert disabled.cta_text is None


# --- transcript corrections --------------------------------------------------


def test_corrections_preserve_timing_and_persist(tmp_path: Path):
    tr = make_transcript([("alpha beta gamma", 0.0, 3.0), ("delta epsilon", 3.0, 5.0)])
    seg = tr.segments[0]
    review_dir = tmp_path / "review"
    add_correction(review_dir, TranscriptCorrection(segment_id=seg.id, corrected_text="BETA", word_index=1))
    corrections = load_corrections(review_dir)  # persisted, not in-memory
    fixed = apply_corrections(tr, corrections)

    new = fixed.segments[0]
    assert [w.word for w in new.words] == ["alpha", "BETA", "gamma"]
    assert [(w.start, w.end) for w in new.words] == [(w.start, w.end) for w in seg.words]
    assert (new.start, new.end) == (seg.start, seg.end)
    assert fixed.segments[1] == tr.segments[1]
    assert tr.segments[0].words[1].word == "beta"  # source transcript untouched


def test_segment_correction_with_different_word_count_keeps_span(tmp_path: Path):
    tr = make_transcript([("one two three", 0.0, 3.0)])
    seg = tr.segments[0]
    fixed = apply_corrections(tr, [TranscriptCorrection(segment_id=seg.id, corrected_text="uno dos")])
    words = fixed.segments[0].words
    assert [w.word for w in words] == ["uno", "dos"]
    assert words[0].start == pytest.approx(seg.start)
    assert words[-1].end == pytest.approx(seg.end)


def test_dialect_and_code_switching_pass_through_verbatim():
    text = "إزاي نعمل الـ branding بتاعنا"
    tr = make_transcript([("x y", 0.0, 2.0)])
    seg = tr.segments[0]
    fixed = apply_corrections(tr, [TranscriptCorrection(segment_id=seg.id, corrected_text=text)])
    assert fixed.segments[0].text == text


# --- shipped brand configuration wires through generic code ------------------


def test_shipped_brand_profile_font_and_logo_path_resolve():
    """The bundled profile that sets an Arabic font must resolve font files,
    a logo, and a brand-driven caption style through the generic loaders."""
    brands_root = REPO_ROOT / "brands"
    candidates = [d for d in brands_root.iterdir() if (d / "brand.yaml").is_file()] if brands_root.is_dir() else []
    with_fonts = [d for d in candidates if (d / "fonts").is_dir() and any((d / "fonts").iterdir())]
    if not with_fonts:
        pytest.skip("no bundled brand profile ships fonts")
    for d in with_fonts:
        brand = brand_loader.load_brand(d.name, root=brands_root)
        assert brand.logo.behavior.mode in set(LogoMode)
        assert brand_loader.resolve_fonts_dir(d.name, root=brands_root) is not None
        style = resolve_brand_caption_style("minimal", brand).style
        assert style.font_ar == brand.arabic_font


def test_ibm_plex_sans_arabic_flows_through_a_brand_profile(tmp_path: Path):
    brands_root = tmp_path / "brands"
    brand_loader.init_brand("plexco", root=brands_root)
    brand_yaml = brands_root / "plexco" / "brand.yaml"
    data = yaml.safe_load(brand_yaml.read_text(encoding="utf-8"))
    data["typography"] = {"arabic": "IBM Plex Sans Arabic", "latin": "IBM Plex Sans Arabic"}
    brand_yaml.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (brands_root / "plexco" / "fonts" / "IBMPlexSansArabic-Regular.ttf").write_bytes(b"font")

    brand = brand_loader.load_brand("plexco", root=brands_root)
    style = resolve_brand_caption_style("minimal", brand).style
    assert style.font_ar == "IBM Plex Sans Arabic"
    assert "Arial" not in (style.font_ar, style.font_en)
    assert brand_loader.resolve_fonts_dir("plexco", root=brands_root) == brands_root / "plexco" / "fonts"


# --- hardcoding audit --------------------------------------------------------

_FORBIDDEN = [
    (re.compile(r"#?0A3D62|#?8E44AD|#?F1C40F", re.IGNORECASE), "client palette hex"),
    (re.compile(r"تواصل معنا"), "generic Arabic CTA"),
    (re.compile(r"[\"'](?:Learn more|Contact us)[\"']"), "generic English CTA literal"),
    (re.compile(r"FFCC00", re.IGNORECASE), "invented default yellow accent"),
    (re.compile(r"[A-Za-z]:[\\/](?:Users|Projects)", re.IGNORECASE), "project-specific absolute path"),
    (re.compile(r"صور محمد|ethics and values", re.IGNORECASE), "this project's paths/phrases"),
]
_SCAN_SUFFIXES = {".py", ".ts", ".tsx", ".yaml", ".yml", ".json"}


def test_no_client_specific_values_in_generic_modules():
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*"):
        if path.suffix not in _SCAN_SUFFIXES or "node_modules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in _FORBIDDEN:
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                line_text = text.splitlines()[line - 1]
                # A comment that documents *why* a value was removed is not a hardcode.
                if label == "invented default yellow accent" and line_text.lstrip().startswith("#"):
                    continue
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{line} [{label}] {line_text.strip()[:80]}")
    assert offenders == []
